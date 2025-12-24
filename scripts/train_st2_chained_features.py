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

# --- 0. UTILS & DETERMINISM ---
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

# --- 1. DATA PREP (7 FEATURES - FE AVANSAT) ---
def build_sequences(df):
    seqs = []
    df = df.sort_values(["user_id", "timestamp"]).copy()
    
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        if len(g) < 2: continue

        # 1. Delta Time (Log-scaled)
        ts = g["timestamp"].astype("int64") // 10**9
        dt = ts.diff().fillna(0).astype(np.float32).values
        dt_feat = np.log1p(np.clip(dt / (3600.0 * 24.0), 0, 3650)).astype(np.float32)

        # 2. Volatilitatea Valenței
        v_diff = g["pred_valence"].diff().fillna(0).astype(np.float32).values
        
        # 3. Ora Ciclică
        hour = g["timestamp"].dt.hour.astype(np.float32).values
        hour_sin = np.sin(2 * np.pi * hour / 24.0)

        # X Stack: 7 Dimensiuni
        x = np.stack([
            g["pred_valence"].astype(np.float32).values, # 0
            g["pred_arousal"].astype(np.float32).values, # 1
            dt_feat,                                     # 2
            g["collection_phase"].astype(np.float32).values / 7.0, # 3
            g["is_words"].astype(np.float32).values,      # 4
            v_diff,                                      # 5 (Volatilitate)
            hour_sin                                     # 6 (Ora)
        ], axis=1)

        y = np.stack([
            g["valence"].shift(-1).values - g["valence"].values,
            g["arousal"].shift(-1).values - g["arousal"].values,
        ], axis=1).astype(np.float32)

        x = x[:-1, :]
        y = y[:-1, :]
        
        meta = g.iloc[:-1][["user_id", "text_id", "timestamp"]].copy()
        seqs.append((x, y, meta))
    return seqs

class SeqDataset(Dataset):
    def __init__(self, seqs): self.seqs = seqs
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i): return torch.from_numpy(self.seqs[i][0]), torch.from_numpy(self.seqs[i][1]), self.seqs[i][2]

def collate_pad(batch):
    xs, ys, metas = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    max_l = int(lens.max().item())
    feat_dim = xs[0].shape[1]

    X = torch.zeros((len(xs), max_l, feat_dim), dtype=torch.float32)
    Y = torch.zeros((len(ys), max_l, 2), dtype=torch.float32)
    M = torch.zeros((len(xs), max_l), dtype=torch.bool) # False = Padding (logic inversat pt GRU maskare manuală)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = True # True = Data reală

    return X, Y, M, lens, metas

# --- 2. MODEL: CHAINED BI-DIRECTIONAL GRU ---
class ChainedBiGRU(nn.Module):
    def __init__(self, in_dim=7, hid=64, layers=1, dropout=0.25):
        super().__init__()
        
        # GRU Bidirecțional
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hid,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=True, 
        )
        
        gru_out_dim = hid * 2
        self.drop = nn.Dropout(dropout)
        
        # Head Valență
        self.head_valence = nn.Sequential(
            nn.Linear(gru_out_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )
        
        # Head Arousal (Primește Output GRU + Predicția Valență)
        self.head_arousal = nn.Sequential(
            nn.Linear(gru_out_dim + 1, hid), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

    def forward(self, x, lens):
        # Pack sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = self.gru(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        
        out = self.drop(out)
        
        # 1. Prezicem Valența
        pred_v = self.head_valence(out)
        
        # 2. Înlănțuim (Chain): Concatenăm Valența la output-ul GRU
        aro_input = torch.cat([out, pred_v], dim=2) 
        
        # 3. Prezicem Arousal folosind contextul + valența
        pred_a = self.head_arousal(aro_input)
        
        return torch.cat([pred_v, pred_a], dim=2)

# --- 3. MAIN (ENSEMBLE LOGIC) ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="models/st2_complete_ensemble")
    ap.add_argument("--eval_repo", required=False)
    
    # Hiperparametri
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hid", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--top_k", type=int, default=3) # Câte modele păstrăm pentru medie
    args = ap.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    device = get_device()
    
    # Load Data
    print(f"🔄 Loading data: {args.data}")
    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "collection_phase" not in df.columns: df["collection_phase"] = 1
    if "is_words" not in df.columns: df["is_words"] = False

    # Split
    if os.path.exists("fixed_split.json"):
        with open("fixed_split.json", "r") as f: split_data = json.load(f)
        train_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(split_data["train"])])), batch_size=8, shuffle=True, collate_fn=collate_pad)
        val_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(split_data["val"])])), batch_size=16, shuffle=False, collate_fn=collate_pad)
    else:
        print("❌ fixed_split.json missing!"); return

    # --- ENSEMBLE START ---
    seeds = [42, 123, 7, 2024, 99, 555, 777, 888, 10, 11, 22, 33, 101, 202, 303]
    results = [] # (combined_score, rv, ra, seed, state_dict)

    print(f"\n🌪️ Start Ensemble Chained-BiGRU ({len(seeds)} rulări)...")

    for idx, seed in enumerate(seeds):
        set_seed(seed)
        model = ChainedBiGRU(in_dim=7, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        
        best_run_score = -1.0
        best_run_metrics = (0, 0) # rv, ra
        best_run_state = None

        for ep in range(1, args.epochs + 1):
            model.train()
            for X, Y, M, lens, _ in train_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                opt.zero_grad()
                P = model(X, lens)
                
                # Loss pe datele reale (M=True)
                pred_v, pred_a = P[M][:,0], P[M][:,1]
                gold_v, gold_a = Y[M][:,0], Y[M][:,1]
                
                loss_v = nn.functional.smooth_l1_loss(pred_v, gold_v)
                loss_a = nn.functional.smooth_l1_loss(pred_a, gold_a)
                
                # Ponderare egală sau ușor spre Arousal (care e mai greu)
                loss = 1.0 * loss_v + 1.2 * loss_a
                loss.backward()
                opt.step()
            
            # Eval
            model.eval()
            all_pv, all_gv, all_pa, all_ga = [], [], [], []
            with torch.no_grad():
                for X, Y, M, lens, _ in val_dl:
                    X, Y, M = X.to(device), Y.to(device), M.to(device)
                    P = model(X, lens)
                    all_pv.append(P[M][:,0].cpu().numpy())
                    all_gv.append(Y[M][:,0].cpu().numpy())
                    all_pa.append(P[M][:,1].cpu().numpy())
                    all_ga.append(Y[M][:,1].cpu().numpy())
            
            rv = pearson_safe(np.concatenate(all_gv), np.concatenate(all_pv))
            ra = pearson_safe(np.concatenate(all_ga), np.concatenate(all_pa))
            
            # Scorul combinat (Media geometrică sau aritmetică)
            combined_score = (rv + ra) / 2.0 

            if combined_score > best_run_score:
                best_run_score = combined_score
                best_run_metrics = (rv, ra)
                # Salvăm în RAM
                best_run_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        print(f"[{idx+1}/{len(seeds)}] Seed {seed}: Best Mean r={best_run_score:.4f} (Valence={best_run_metrics[0]:.3f}, Arousal={best_run_metrics[1]:.3f})")
        
        if best_run_score > 0.30: # Prag minim de decență
            results.append((best_run_score, best_run_metrics[0], best_run_metrics[1], seed, best_run_state))

    # --- AGREGATION ---
    print(f"\n📊 Calculare Medie Ensemble (Top {args.top_k})...")
    # Sortăm după scorul combinat
    results.sort(key=lambda x: x[0], reverse=True)
    top_models = results[:args.top_k]
    
    if not top_models:
        print("❌ Nu am găsit modele bune.")
        return

    print("Campioni selectați:")
    for sc, rv, ra, sd, _ in top_models:
        print(f"  -> Seed {sd} (Mean={sc:.4f} | V={rv:.4f}, A={ra:.4f})")

    # Facem predicții finale (V + A)
    final_pv_sum = None
    final_pa_sum = None
    
    # Colectăm gold truth (deltas) pentru evaluarea finală a ansamblului
    # Re-citim din DataLoader
    flat_gold_v = []
    flat_gold_a = []
    
    with torch.no_grad():
         for _, Y, M, lens, _ in val_dl:
             Ynp = Y.numpy()
             for k in range(len(Ynp)):
                 l = int(lens[k])
                 flat_gold_v.append(Ynp[k, :l, 0])
                 flat_gold_a.append(Ynp[k, :l, 1])
    
    flat_gold_v = np.concatenate(flat_gold_v)
    flat_gold_a = np.concatenate(flat_gold_a)

    # Iterăm prin campioni pentru predicții
    for idx, (_, _, _, _, state) in enumerate(top_models):
        model = ChainedBiGRU(in_dim=7, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
        model.load_state_dict(state)
        model.eval()
        
        pv_list, pa_list = [], []
        
        with torch.no_grad():
            for X, Y, M, lens, _ in val_dl:
                X = X.to(device)
                P = model(X, lens).cpu().numpy()
                for bi in range(len(P)):
                    L = int(lens[bi])
                    pv_list.append(P[bi, :L, 0])
                    pa_list.append(P[bi, :L, 1])
        
        flat_pv = np.concatenate(pv_list)
        flat_pa = np.concatenate(pa_list)
        
        if idx == 0:
            final_pv_sum = flat_pv
            final_pa_sum = flat_pa
        else:
            final_pv_sum += flat_pv
            final_pa_sum += flat_pa

    # Media
    final_pv = final_pv_sum / len(top_models)
    final_pa = final_pa_sum / len(top_models)

    # Evaluăm Ensemble-ul
    ens_rv = pearson_safe(flat_gold_v, final_pv)
    ens_ra = pearson_safe(flat_gold_a, final_pa)
    
    print(f"\n🏆 ENSEMBLE SCORE FINAL:")
    print(f"   Valence r: {ens_rv:.4f}")
    print(f"   Arousal r: {ens_ra:.4f}")
    print(f"   Mean    r: {(ens_rv + ens_ra)/2:.4f}")

    # Reconstruim DataFrame-ul final
    out_rows = []
    cursor = 0
    
    # Re-iterăm prin validation loader pentru metadate
    with torch.no_grad():
        for _, _, _, _, metas in val_dl:
            for meta in metas:
                L = len(meta)
                chunk_v = final_pv[cursor : cursor+L]
                chunk_a = final_pa[cursor : cursor+L]
                
                meta_out = meta.copy()
                meta_out["pred_delta_valence"] = chunk_v
                meta_out["pred_delta_arousal"] = chunk_a
                
                # Punem și gold-ul pentru referință
                # (Atenție: aici gold e recalculat din Y-ul original în loop-ul anterior, 
                # dar meta conține valorile absolute. Nu suprascriem gold-ul absolut)
                # Dacă vrem gold delta, le putem lua din flat_gold_v/a
                meta_out["gold_delta_valence"] = flat_gold_v[cursor : cursor+L]
                meta_out["gold_delta_arousal"] = flat_gold_a[cursor : cursor+L]
                
                out_rows.append(meta_out)
                cursor += L

    out_csv = f"{args.outdir}/val_st2_ensemble_predictions.csv"
    pd.concat(out_rows).to_csv(out_csv, index=False)
    print(f"📂 Predicții Full (V+A) salvate în: {out_csv}")

if __name__ == "__main__":
    main()