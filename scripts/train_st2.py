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
    # Sortăm după user și timp pentru a crea secvențe corecte
    df = df.sort_values(["user_id", "timestamp"]).copy()
    
    for uid, g in df.groupby("user_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)

        if len(g) < 2:
            continue

        # Feature Engineering: Timp relativ (delta t)
        ts = g["timestamp"].astype("int64") // 10**9
        dt = ts.diff().fillna(0).astype(np.float32).values
        dt_days = dt / (3600.0 * 24.0)
        # Transformare logaritmică pentru a stabiliza variațiile mari de timp
        dt_feat = np.log1p(np.clip(dt_days, 0, 3650)).astype(np.float32)

        phase = g["collection_phase"].astype(np.float32).values
        is_words = g["is_words"].astype(np.float32).values

        # INPUT (X): [Pred_V, Pred_A, TimeDelta, Phase, IsWords]
        # Acestea sunt output-urile de la Task 1 + Metadata temporală
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

        # TARGET (Y): State Change (Diferența dintre t+1 și t)
        y = np.stack(
            [
                g["valence"].shift(-1).values - g["valence"].values,
                g["arousal"].shift(-1).values - g["arousal"].values,
            ],
            axis=1,
        ).astype(np.float32)

        # Eliminăm ultima linie (nu are "next" pentru target)
        x = x[:-1, :]
        y = y[:-1, :]

        # Păstrăm metadatele pentru identificare la evaluare
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

    # Padding cu zero pentru secvențe de lungimi diferite
    X = torch.zeros((len(xs), maxlen, feat_dim), dtype=torch.float32)
    Y = torch.zeros((len(ys), maxlen, 2), dtype=torch.float32)
    M = torch.zeros((len(xs), maxlen), dtype=torch.bool) # Mask

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = x.shape[0]
        X[i, :L] = x
        Y[i, :L] = y
        M[i, :L] = True

    return X, Y, M, lens, metas

# --- 3. SOTA MODEL (Multi-Head GRU) ---
class GRURegressor(nn.Module):
    def __init__(self, in_dim=5, hid=64, layers=1, dropout=0.2):
        super().__init__()
        
        # CORPUL COMUN (Feature Extractor)
        # Învață contextul temporal indiferent de emoție
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hid,
            num_layers=layers,
            batch_first=True,
            dropout=(dropout if layers > 1 else 0.0),
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        
        # HEAD 1: Specialist în VALENCE
        # Rețea densă separată care decide doar dacă utilizatorul e fericit/trist
        self.head_valence = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.GELU(),  # Activare modernă (State of the Art)
            nn.Dropout(dropout),
            nn.Linear(hid // 2, 1)
        )
        
        # HEAD 2: Specialist în AROUSAL
        # Rețea densă separată care decide doar intensitatea
        self.head_arousal = nn.Sequential(
            nn.Linear(hid, hid // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid // 2, 1)
        )

    def forward(self, x, lens):
        # x: [Batch, Time, Feats]
        out, _ = self.gru(x)
        out = self.drop(out)
        
        # Ramificăm execuția în două direcții
        val_pred = self.head_valence(out)
        aro_pred = self.head_arousal(out)
        
        # Re-combinăm rezultatele: [Batch, Time, 2]
        return torch.cat([val_pred, aro_pred], dim=2)

# --- 4. MAIN ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV output from Task 1 (val_predictions_FULL.csv)")
    ap.add_argument("--outdir", default="models/st2a_gru_sota")
    ap.add_argument("--eval_repo", required=True, help="Path to semeval2026-task2-eval folder")

    # Hiperparametri optimizați
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3) # Learning rate stabil
    ap.add_argument("--batch_users", type=int, default=8)
    ap.add_argument("--hid", type=int, default=64) # Dimensiune medie (balansată)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-4) # Regularizare L2

    ap.add_argument("--val_users_ratio", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--disable_mps_watermark", action="store_true")
    
    args = ap.parse_args()

    # --- SETUP EVALUATOR ---
    if not os.path.exists(args.eval_repo):
         raise FileNotFoundError(f"Nu găsesc repo-ul de evaluare: {args.eval_repo}")
    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print(f"✅ Evaluator oficial integrat.")
    except ImportError:
        print("❌ Eroare la import eval.py")
        sys.exit(1)

    if args.disable_mps_watermark:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --- ÎNCĂRCARE DATE ---
    print(f"Încărcare date din: {args.data}")
    df = pd.read_csv(args.data)
    
    need = {"user_id", "timestamp", "valence", "arousal", "pred_valence", "pred_arousal"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"Lipsesc coloane din CSV: {missing}. Rulează întâi prepare_st2_data.py!")

    df = df.dropna(subset=list(need)).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    if "collection_phase" not in df.columns:
        df["collection_phase"] = 1
    if "is_words" not in df.columns:
        df["is_words"] = False

    users = df["user_id"].unique().tolist()
    users.sort() 
    np.random.seed(args.seed)
    np.random.shuffle(users)
    
    cut = int((1.0 - args.val_users_ratio) * len(users))
    train_users = set(users[:cut])
    val_users = set(users[cut:])

    train_df = df[df["user_id"].isin(train_users)].copy()
    val_df = df[df["user_id"].isin(val_users)].copy()

    print(f"Dataset split: {len(train_users)} Useri Train / {len(val_users)} Useri Val")

    train_seqs = build_sequences(train_df)
    val_seqs = build_sequences(val_df)

    train_dl = DataLoader(
        SeqDataset(train_seqs),
        batch_size=args.batch_users,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
        pin_memory=False,
    )
    val_dl = DataLoader(
        SeqDataset(val_seqs),
        batch_size=args.batch_users * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_pad,
        pin_memory=False,
    )

    device = get_device()
    model = GRURegressor(in_dim=5, hid=args.hid, layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Notă: Nu mai definim loss_fn global, îl calculăm custom în buclă

    best_mean = -1e9
    log_rows = []

    # --- EVALUARE INTERNĂ ---
    def eval_epoch():
        model.eval()
        all_pred_v, all_gold_v = [], []
        all_pred_a, all_gold_a = [], []

        with torch.no_grad():
            for X, Y, M, lens, metas in val_dl:
                X = X.to(device)
                Y = Y.to(device)
                M = M.to(device)

                P = model(X, lens)
                
                # Extragem doar valorile valide
                P_flat = P[M]
                G_flat = Y[M]

                pv = P_flat[:, 0].detach().cpu().numpy()
                pa = P_flat[:, 1].detach().cpu().numpy()
                gv = G_flat[:, 0].detach().cpu().numpy()
                ga = G_flat[:, 1].detach().cpu().numpy()

                all_pred_v.append(pv); all_gold_v.append(gv)
                all_pred_a.append(pa); all_gold_a.append(ga)

        pv = np.concatenate(all_pred_v) if all_pred_v else np.array([])
        gv = np.concatenate(all_gold_v) if all_gold_v else np.array([])
        pa = np.concatenate(all_pred_a) if all_pred_a else np.array([])
        ga = np.concatenate(all_gold_a) if all_gold_a else np.array([])

        rv = pearson_safe(gv, pv)
        ra = pearson_safe(ga, pa)
        return rv, ra, float(np.nanmean([rv, ra]))

    # --- ANTRENARE CU WEIGHTED LOSS ---
    print("\n🚀 Start Training (SOTA Multi-Head)...")
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_losses = []

        for X, Y, M, lens, metas in train_dl:
            X = X.to(device)
            Y = Y.to(device)
            M = M.to(device)

            opt.zero_grad(set_to_none=True)
            P = model(X, lens) # Output: (Batch, Time, 2)

            # --- CUSTOM LOSS LOGIC ---
            pred_v = P[:, :, 0]
            pred_a = P[:, :, 1]
            gold_v = Y[:, :, 0]
            gold_a = Y[:, :, 1]

            # Calculăm loss per element
            lv = nn.functional.smooth_l1_loss(pred_v, gold_v, reduction='none')
            la = nn.functional.smooth_l1_loss(pred_a, gold_a, reduction='none')

            # Aplicăm masca (excludem padding-ul)
            lv = lv[M].mean()
            la = la[M].mean()

            # PONDERARE: Dăm prioritate Arousal-ului (x2.0) pentru că e mai greu
            total_loss = 1.0 * lv + 2.0 * la
            
            total_loss.backward()
            opt.step()

            tr_losses.append(float(total_loss.detach().cpu().item()))

        # Validare
        rv, ra, rmean = eval_epoch()
        
        row = {
            "epoch": ep,
            "loss": f"{np.mean(tr_losses):.4f}",
            "val_r_val": f"{rv:.4f}",
            "val_r_aro": f"{ra:.4f}",
            "mean": f"{rmean:.4f}",
        }
        print(row)
        log_rows.append(row)

        if rmean > best_mean:
            best_mean = rmean
            torch.save({"model": model.state_dict(), "args": vars(args)}, os.path.join(args.outdir, "best.pt"))

    pd.DataFrame(log_rows).to_csv(os.path.join(args.outdir, "train_log.csv"), index=False)

    # --- FINAL PREDICTIONS & OFFICIAL EVAL ---
    print("\n📊 Generare predicții finale...")
    ckpt = torch.load(os.path.join(args.outdir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pred_rows = []
    final_ids, final_pred_v, final_gold_v, final_pred_a, final_gold_a = [], [], [], [], []

    with torch.no_grad():
        for X, Y, M, lens, metas in val_dl:
            X = X.to(device)
            P = model(X, lens).detach().cpu().numpy()
            G = Y.numpy()
            Mnp = M.numpy()

            for bi, meta in enumerate(metas):
                L = int(Mnp[bi].sum())
                p_seq = P[bi, :L, :]
                g_seq = G[bi, :L, :]
                meta = meta.iloc[:L].copy()
                
                meta["pred_delta_valence"] = p_seq[:, 0]
                meta["pred_delta_arousal"] = p_seq[:, 1]
                meta["gold_delta_valence"] = g_seq[:, 0]
                meta["gold_delta_arousal"] = g_seq[:, 1]
                pred_rows.append(meta)
                
                final_ids.extend(meta["text_id"].astype(str).tolist())
                final_pred_v.extend(p_seq[:, 0].tolist())
                final_gold_v.extend(g_seq[:, 0].tolist())
                final_pred_a.extend(p_seq[:, 1].tolist())
                final_gold_a.extend(g_seq[:, 1].tolist())

    pred_df = pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()
    out_csv = os.path.join(args.outdir, "val_st2a_predictions.csv")
    pred_df.to_csv(out_csv, index=False)
    print(f"✅ CSV final salvat: {out_csv}")

    print("\n" + "="*40)
    print("🏆 REZULTATE OFICIALE (SOTA ARCHITECTURE)")
    print("="*40)
    
    try:
        print("\n--- Valence Delta Results ---")
        res_v = official_eval.task2_correlation(final_ids, final_pred_v, final_gold_v)
        print(res_v)

        print("\n--- Arousal Delta Results ---")
        res_a = official_eval.task2_correlation(final_ids, final_pred_a, final_gold_a)
        print(res_a)
        
        with open(os.path.join(args.outdir, "official_metrics.json"), "w") as f:
            json.dump({"valence": res_v, "arousal": res_a}, f, indent=2)
            
    except Exception as e:
        print(f"⚠️ Eroare evaluare: {e}")

if __name__ == "__main__":
    main()