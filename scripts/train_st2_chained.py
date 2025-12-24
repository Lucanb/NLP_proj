import os
import sys
import argparse
import math
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

# --- 1. UTILS ---
def pearson_safe(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    return float(pearsonr(a[m], b[m])[0])

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# --- 2. DATA PREP ---
def build_sequences(df):
    seqs = []
    df = df.sort_values(["user_id", "timestamp"]).copy()
    
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        if len(g) < 2:
            continue

        ts = g["timestamp"].astype("int64") // 10**9
        dt = ts.diff().fillna(0).astype(np.float32).values
        dt_days = dt / (3600.0 * 24.0)
        dt_feat = np.log1p(np.clip(dt_days, 0, 3650)).astype(np.float32)

        phase = g["collection_phase"].astype(np.float32).values
        is_words = g["is_words"].astype(np.float32).values

        x = np.stack(
            [
                g["pred_valence"].astype(np.float32).values,
                g["pred_arousal"].astype(np.float32).values,
                dt_feat,
                phase / 7.0,
                is_words,
            ],
            axis=1,
        )

        y = np.stack(
            [
                g["valence"].shift(-1).values - g["valence"].values,
                g["arousal"].shift(-1).values - g["arousal"].values,
            ],
            axis=1,
        ).astype(np.float32)

        x = x[:-1, :]
        y = y[:-1, :]
        meta = g.iloc[:-1][["user_id", "text_id", "timestamp"]].copy()
        seqs.append((x, y, meta))
    return seqs

class SeqDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        x, y, meta = self.seqs[i]
        return torch.from_numpy(x), torch.from_numpy(y), meta

def collate_pad(batch):
    xs, ys, metas = zip(*batch)
    lens = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    maxlen = int(lens.max().item())
    feat_dim = xs[0].shape[1]

    X = torch.zeros((len(xs), maxlen, feat_dim), dtype=torch.float32)
    Y = torch.zeros((len(ys), maxlen, 2), dtype=torch.float32)
    M = torch.zeros((len(xs), maxlen), dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = True

    return X, Y, M, lens, metas

# --- 3. MODEL SOTA: CHAINED BI-DIRECTIONAL GRU ---
class ChainedBiGRU(nn.Module):
    def __init__(self, in_dim=5, hid=64, layers=1, dropout=0.2):
        super().__init__()
        
        # 1. BI-DIRECTIONAL GRU
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
        
        # 2. VALENCE HEAD
        self.head_valence = nn.Sequential(
            nn.Linear(gru_out_dim, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )
        
        # 3. AROUSAL HEAD (CHAINED)
        self.head_arousal = nn.Sequential(
            nn.Linear(gru_out_dim + 1, hid), # +1 vine de la Valence
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 1)
        )

    def forward(self, x, lens):
        out, _ = self.gru(x) 
        out = self.drop(out)
        
        pred_v = self.head_valence(out)
        aro_input = torch.cat([out, pred_v], dim=2) 
        pred_a = self.head_arousal(aro_input)
        
        return torch.cat([pred_v, pred_a], dim=2)

# --- 4. MAIN ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV output from Task 1")
    ap.add_argument("--outdir", default="models/st2a_chained_sota")
    ap.add_argument("--eval_repo", required=True, help="Path to semeval2026-task2-eval")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_users", type=int, default=8)
    ap.add_argument("--hid", type=int, default=64) 
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--val_users_ratio", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--disable_mps_watermark", action="store_true")
    
    args = ap.parse_args()

    # --- SETUP ---
    if not os.path.exists(args.eval_repo):
         raise FileNotFoundError(f"Nu găsesc repo-ul: {args.eval_repo}")
    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print(f"✅ Evaluator oficial integrat.")
    except ImportError:
        sys.exit(1)

    if args.disable_mps_watermark:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- DATA ---
    print(f"Încărcare date din: {args.data}")
    df = pd.read_csv(args.data)
    need = {"user_id", "timestamp", "valence", "arousal", "pred_valence", "pred_arousal"}
    df = df.dropna(subset=list(need)).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    if "collection_phase" not in df.columns: df["collection_phase"] = 1
    if "is_words" not in df.columns: df["is_words"] = False


    if os.path.exists("fixed_split.json"):
        print("🔒 Se folosește fixed_split.json pentru a îngheța userii de antrenare/validare.")
        with open("fixed_split.json", "r") as f:
            split_data = json.load(f)
        train_users = set(split_data["train"])
        val_users = set(split_data["val"])
    else:
        print("⚠️ ATENȚIE: Nu s-a găsit fixed_split.json! Se folosește shuffle random (nerecomandat pt ensemble).")
        users = df["user_id"].unique().tolist()
        users.sort()
        np.random.shuffle(users)
        cut = int((1.0 - args.val_users_ratio) * len(users))
        train_users = set(users[:cut])
        val_users = set(users[cut:])

    train_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(train_users)].copy())), 
                          batch_size=args.batch_users, shuffle=True, collate_fn=collate_pad)
    val_dl = DataLoader(SeqDataset(build_sequences(df[df["user_id"].isin(val_users)].copy())), 
                        batch_size=args.batch_users*2, shuffle=False, collate_fn=collate_pad)

    # --- MODEL & OPTIM ---
    device = get_device()
    model = ChainedBiGRU(in_dim=5, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mean = -1e9
    log_rows = []

    def eval_epoch():
        model.eval()
        all_pred_v, all_gold_v, all_pred_a, all_gold_a = [], [], [], []
        with torch.no_grad():
            for X, Y, M, lens, metas in val_dl:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                P = model(X, lens)
                P_flat = P[M]
                G_flat = Y[M]
                all_pred_v.append(P_flat[:,0].cpu().numpy())
                all_gold_v.append(G_flat[:,0].cpu().numpy())
                all_pred_a.append(P_flat[:,1].cpu().numpy())
                all_gold_a.append(G_flat[:,1].cpu().numpy())

        pv = np.concatenate(all_pred_v) if all_pred_v else np.array([])
        gv = np.concatenate(all_gold_v) if all_gold_v else np.array([])
        pa = np.concatenate(all_pred_a) if all_pred_a else np.array([])
        ga = np.concatenate(all_gold_a) if all_gold_a else np.array([])
        rv = pearson_safe(gv, pv)
        ra = pearson_safe(ga, pa)
        return rv, ra, float(np.nanmean([rv, ra]))

    # --- TRAINING LOOP ---
    print("\n🚀 Start Training (Chained Bi-GRU)...")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []

        for X, Y, M, lens, metas in train_dl:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            opt.zero_grad(set_to_none=True)
            P = model(X, lens)

            pred_v, pred_a = P[:, :, 0], P[:, :, 1]
            gold_v, gold_a = Y[:, :, 0], Y[:, :, 1]

            lv = nn.functional.smooth_l1_loss(pred_v, gold_v, reduction='none')[M].mean()
            la = nn.functional.smooth_l1_loss(pred_a, gold_a, reduction='none')[M].mean()

            # Păstrăm ponderarea
            loss = 1.0 * lv + 1.5 * la 
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        rv, ra, rmean = eval_epoch()
        row = {"epoch": ep, "loss": f"{np.mean(tr_losses):.4f}", "val_r_val": f"{rv:.4f}", "val_r_aro": f"{ra:.4f}", "mean": f"{rmean:.4f}"}
        print(row)
        log_rows.append(row)

        if rmean > best_mean:
            best_mean = rmean
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.outdir, "best.pt"))
            
    # --- FINAL ---
    ckpt = torch.load(os.path.join(args.outdir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    pred_rows, f_ids, f_pv, f_gv, f_pa, f_ga = [], [], [], [], [], []
    with torch.no_grad():
        for X, Y, M, lens, metas in val_dl:
            X = X.to(device)
            P = model(X, lens).cpu().numpy(); G = Y.numpy(); Mnp = M.numpy()
            for bi, meta in enumerate(metas):
                L = int(Mnp[bi].sum())
                meta = meta.iloc[:L].copy()
                meta["pred_delta_valence"] = P[bi, :L, 0]
                meta["pred_delta_arousal"] = P[bi, :L, 1]
                meta["gold_delta_valence"] = G[bi, :L, 0]
                meta["gold_delta_arousal"] = G[bi, :L, 1]
                pred_rows.append(meta)
                f_ids.extend(meta["text_id"].astype(str).tolist())
                f_pv.extend(P[bi, :L, 0]); f_gv.extend(G[bi, :L, 0])
                f_pa.extend(P[bi, :L, 1]); f_ga.extend(G[bi, :L, 1])

    out_csv = os.path.join(args.outdir, "val_st2a_predictions.csv")
    pd.concat(pred_rows).to_csv(out_csv, index=False)
    
    print("\n🏆 REZULTATE OFICIALE (CHAINED BI-GRU)")
    try:
        print("Valence:", official_eval.task2_correlation(f_ids, f_pv, f_gv))
        print("Arousal:", official_eval.task2_correlation(f_ids, f_pa, f_ga))
    except: pass

if __name__ == "__main__":
    main()