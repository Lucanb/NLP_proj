import os
import sys
import argparse
import math
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def pearson_safe(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5: return float("nan")
    return float(pearsonr(a[m], b[m])[0])

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

def build_sequences(df):
    seqs = []
    df = df.sort_values(["user_id", "timestamp"]).copy()
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2: continue

        ts = g["timestamp"].astype("int64") // 10**9
        dt = ts.diff().fillna(0).astype(np.float32).values
        dt_feat = np.log1p(np.clip(dt / (3600.0 * 24.0), 0, 3650)).astype(np.float32)
        v_diff = g["pred_valence"].diff().fillna(0).astype(np.float32).values
        hour = g["timestamp"].dt.hour.astype(np.float32).values
        hour_sin = np.sin(2 * np.pi * hour / 24.0)

        x = np.stack([
            g["pred_valence"].astype(np.float32).values,
            g["pred_arousal"].astype(np.float32).values,
            dt_feat,
            g["collection_phase"].astype(np.float32).values / 7.0,
            g["is_words"].astype(np.float32).values,
            v_diff,
            hour_sin
        ], axis=1)

        y = np.stack([
            g["valence"].shift(-1).values - g["valence"].values,
            g["arousal"].shift(-1).values - g["arousal"].values,
        ], axis=1).astype(np.float32)

        seqs.append((x[:-1], y[:-1], g.iloc[:-1][["user_id", "text_id"]].copy()))
    return seqs

class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return torch.from_numpy(self.seqs[i][0]), torch.from_numpy(self.seqs[i][1]), self.seqs[i][2]

def collate_pad(batch):
    xs, ys, metas = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_l = int(lens.max().item())
    X = torch.zeros((len(xs), max_l, 7), dtype=torch.float32)
    Y = torch.zeros((len(ys), max_l, 2), dtype=torch.float32)
    M = torch.ones((len(xs), max_l), dtype=torch.bool) 
    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]; X[i, :L] = x; Y[i, :L] = y; M[i, :L] = False
    return X, Y, M, lens, metas

class TemporalTransformer(nn.Module):
    def __init__(self, in_dim=7, hid=64, nhead=4, layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hid)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hid) * 0.02)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hid, nhead=nhead, dim_feedforward=hid*4, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hid // 2, 1))

    def forward(self, x, mask_padding):
        B, T, D = x.shape
        x = self.input_proj(x) + self.pos_encoder[:, :T, :]
        out = self.transformer(x, src_key_padding_mask=mask_padding)
        return self.head(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="models/st2a_arousal_ensemble")
    ap.add_argument("--eval_repo", required=False)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)

    ap.add_argument("--top_k", type=int, default=3) 
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    with open("fixed_split.json", "r") as f: split_data = json.load(f)
    train_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(split_data["train"])])), batch_size=8, shuffle=True, collate_fn=collate_pad)
    val_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(split_data["val"])])), batch_size=16, shuffle=False, collate_fn=collate_pad)


    seeds = [42, 123, 7, 2024, 99, 101, 555, 888, 1000, 11, 22, 33, 44, 777, 1234, 4321, 987, 666, 1999, 3000]
    
    results = []

    print(f"\n ENSEMBLE START: Vom testa {len(seeds)} seed-uri si vom combina TOP {args.top_k}...")

    for i, current_seed in enumerate(seeds):
        set_seed(current_seed)
        model = TemporalTransformer(in_dim=7, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)
        loss_fn = nn.SmoothL1Loss(reduction='none')

        best_run_r = -1.0
        best_run_state = None

        for ep in range(1, args.epochs + 1):
            model.train()
            for X, Y, M, _, _ in train_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                opt.zero_grad()
                loss = loss_fn(model(X, M).squeeze(-1), Y[:, :, 1])[~M].mean()
                loss.backward(); opt.step()
            
            model.eval()
            preds, golds = [], []
            with torch.no_grad():
                for X, Y, M, lens, _ in val_dl:
                    p = model(X.to(device), M.to(device)).squeeze(-1).cpu().numpy()
                    g = Y[:, :, 1].numpy()
                    for k in range(len(p)):
                        preds.extend(p[k, :int(lens[k])]); golds.extend(g[k, :int(lens[k])])
            
            r = pearson_safe(golds, preds)
            scheduler.step(r)

            if r > best_run_r:
                best_run_r = r
                best_run_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"[{i+1}/{len(seeds)}] Seed {current_seed}: Best Local r = {best_run_r:.4f}")
        
        if best_run_r > 0.15:
            results.append((best_run_r, current_seed, best_run_state))

    print(f"\n Calculare Ensemble din TOP {args.top_k} modele...")
    
    results.sort(key=lambda x: x[0], reverse=True)
    top_models = results[:args.top_k]

    if not top_models:
        print("Nu am gasit niciun model bun!")
        return

    print("Modele selectate pentru vot:")
    for score, seed, _ in top_models:
        print(f"  -> Seed {seed} (r={score:.4f})")

    all_preds_matrix = []
    gold_truth = []
    
    with torch.no_grad():
        temp_golds = []
        for _, _, _, _, metas in val_dl:
             pass 

    for rank, (score, seed, state) in enumerate(top_models):
        model = TemporalTransformer(in_dim=7, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
        model.load_state_dict(state)
        model.eval()
        
        preds_model = []
        golds_model = []
        
        with torch.no_grad():
            for X, Y, M, lens, _ in val_dl:
                p = model(X.to(device), M.to(device)).squeeze(-1).cpu().numpy()
                g = Y[:, :, 1].numpy()
                for k in range(len(p)):
                    l = int(lens[k])
                    preds_model.extend(p[k, :l])
                    golds_model.extend(g[k, :l])
        
        all_preds_matrix.append(preds_model)
        if rank == 0: gold_truth = golds_model
        
        torch.save(state, f"{args.outdir}/model_rank{rank+1}_seed{seed}.pt")

    ensemble_preds = np.mean(all_preds_matrix, axis=0)
    ensemble_r = pearson_safe(gold_truth, ensemble_preds)
    
    print(f"\n ENSEMBLE SCORE FINAL: {ensemble_r:.4f}")
    
    best_single = top_models[0][0]
    if ensemble_r > best_single:
        print(f" BOOST: Ensemble a imbunatatit scorul cu +{ensemble_r - best_single:.4f}")
    else:
        print(f" Ensemble nu a depasit cel mai bun model individual ({best_single:.4f}).")

    final_rows = []
    with torch.no_grad():
        for X, Y, M, lens, metas in val_dl:
            for i, meta in enumerate(metas):
                l = int(lens[i])
                row = meta.iloc[:l].copy()
                row["pred_delta_arousal"] = ensemble_preds[idx_counter : idx_counter+l]
                row["gold_delta_arousal"] = gold_truth[idx_counter : idx_counter+l]
                final_rows.append(row)
                idx_counter += l

    pd.concat(final_rows).to_csv(f"{args.outdir}/val_st2a_predictions.csv", index=False)
    print(f" Predictii Ensemble salvate in: {args.outdir}/val_st2a_predictions.csv")

if __name__ == "__main__":
    main()