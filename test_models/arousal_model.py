import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback,
)
from safetensors.torch import save_file


def pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def extract_feats(texts):
    feats = []
    for t in texts:
        if t is None:
            t = ""
        exclam = t.count("!")
        quest = t.count("?")
        caps = sum(1 for c in t if c.isalpha() and c.isupper())
        letters = sum(1 for c in t if c.isalpha())
        caps_ratio = (caps / letters) if letters > 0 else 0.0
        length = len(t)
        feats.append([exclam, quest, caps_ratio, length])
    return np.asarray(feats, dtype=np.float32)


def freeze_bottom_layers(encoder, freeze_n: int):
    # DeBERTa-v3 (HF) are de obicei encoder.encoder.layer / encoder.deberta.encoder.layer
    layer_paths = [
        ("deberta", "encoder", "layer"),
        ("encoder", "layer"),
    ]
    layers = None
    for path in layer_paths:
        obj = encoder
        ok = True
        for p in path:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok:
            layers = obj
            break

    if layers is None:
        return

    freeze_n = max(0, int(freeze_n))
    freeze_n = min(freeze_n, len(layers))
    for i in range(freeze_n):
        for p in layers[i].parameters():
            p.requires_grad = False


class DebertaArousalSOTA(nn.Module):
    def __init__(
        self,
        base_model: str,
        vocab_size: int,
        hidden_dropout: float = 0.2,
        gru_hidden: int = 64,
        feat_dim: int = 4,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.encoder.resize_token_embeddings(int(vocab_size))

        # reduce memory
        try:
            self.encoder.gradient_checkpointing_enable()
        except Exception:
            pass

        if freeze_layers > 0:
            freeze_bottom_layers(self.encoder, freeze_layers)

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(32, 16),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(2 * gru_hidden + 16, 128),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(128, 1),
        )

        self.loss_fn = nn.SmoothL1Loss(beta=0.1)

    def forward(self, input_ids=None, attention_mask=None, feats=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state

        h = self.dropout(h)
        gru_out, _ = self.gru(h)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (gru_out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))

        if feats is None:
            feats = torch.zeros((pooled.size(0), 4), device=pooled.device, dtype=pooled.dtype)
        feat_emb = self.feat_mlp(feats)

        x = torch.cat([pooled, feat_emb], dim=1)
        pred = self.head(x).squeeze(-1)

        if labels is not None:
            labels = labels.float().view(-1)
            loss = self.loss_fn(pred, labels)
            return {"loss": loss, "logits": pred}

        return {"logits": pred}


def make_kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


class FeatCollator:
    def __init__(self, pad_collator):
        self.pad = pad_collator

    def __call__(self, features):
        raw = [f["feats"] for f in features]
        feats = torch.as_tensor(
            np.stack([np.asarray(x, dtype=np.float32).reshape(-1) for x in raw], axis=0),
            dtype=torch.float32
        )
        for f in features:
            f.pop("feats", None)
        batch = self.pad(features)
        batch["feats"] = feats
        return batch


def mps_cleanup():
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="arousal")
    ap.add_argument("--base_model", default="microsoft/deberta-v3-base")
    ap.add_argument("--out_root", default="models/arousal_sota_len128_cv")
    ap.add_argument("--maxlen", type=int, default=128)

    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.01)

    # MPS-safe defaults
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)  # effective batch = 16
    ap.add_argument("--eval_batch", type=int, default=2)

    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", action="store_true")

    ap.add_argument("--save_best", action="store_true")
    ap.add_argument("--no_pearson", action="store_true")

    ap.add_argument("--gru_hidden", type=int, default=64)
    ap.add_argument("--freeze_layers", type=int, default=6)

    args = ap.parse_args()

    set_seed(args.seed)

    device = "cpu" if args.force_cpu else (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    os.makedirs(args.out_root, exist_ok=True)

    df = pd.read_csv(args.train_csv).copy()
    df = df.dropna(subset=[args.text_col, args.label_col]).reset_index(drop=True)
    df[args.label_col] = df[args.label_col].astype(float)
    df["__feats__"] = list(extract_feats(df[args.text_col].tolist()))

    folds = make_kfold_indices(len(df), args.folds, args.seed)
    cv_rows = []

    for fi in range(args.folds):
        mps_cleanup()

        val_idx = folds[fi]
        tr_idx = np.concatenate([folds[j] for j in range(args.folds) if j != fi])

        df_tr = df.iloc[tr_idx].reset_index(drop=True)
        df_va = df.iloc[val_idx].reset_index(drop=True)

        print(f"\n===== FOLD {fi+1}/{args.folds}  train={len(df_tr)}  val={len(df_va)} =====")

        fold_dir = os.path.join(args.out_root, f"fold_{fi+1}")
        os.makedirs(fold_dir, exist_ok=True)

        tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

        tr_ds = Dataset.from_pandas(df_tr[[args.text_col, args.label_col, "__feats__"]])
        va_ds = Dataset.from_pandas(df_va[[args.text_col, args.label_col, "__feats__"]])

        def tok_fn(batch):
            enc = tok(batch[args.text_col], truncation=True, max_length=args.maxlen)
            enc["labels"] = batch[args.label_col]
            enc["feats"] = batch["__feats__"]
            return enc

        tr_ds = tr_ds.map(tok_fn, batched=True)
        va_ds = va_ds.map(tok_fn, batched=True)

        keep = ["input_ids", "attention_mask", "labels", "feats"]
        tr_ds = tr_ds.remove_columns([c for c in tr_ds.column_names if c not in keep])
        va_ds = va_ds.remove_columns([c for c in va_ds.column_names if c not in keep])
        tr_ds.set_format("torch")
        va_ds.set_format("torch")

        model = DebertaArousalSOTA(
            base_model=args.base_model,
            vocab_size=len(tok),
            hidden_dropout=args.dropout,
            gru_hidden=args.gru_hidden,
            feat_dim=4,
            freeze_layers=args.freeze_layers,
        ).to(device)

        pad = DataCollatorWithPadding(tokenizer=tok)
        collator = FeatCollator(pad)

        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            preds = np.asarray(preds).reshape(-1)
            labels = np.asarray(labels).reshape(-1)
            mse = float(np.mean((preds - labels) ** 2))
            if args.no_pearson:
                return {"mse": mse}
            return {"mse": mse, "pearson": pearsonr_np(preds, labels)}

        targs = TrainingArguments(
            output_dir=os.path.join(fold_dir, "_hf"),
            learning_rate=args.lr,
            weight_decay=args.wd,
            num_train_epochs=args.epochs,

            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            per_device_eval_batch_size=args.eval_batch,

            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            save_total_limit=2,
            logging_steps=50,
            report_to="none",

            dataloader_drop_last=False,
            dataloader_num_workers=0,

            fp16=False,
            bf16=False,
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=tr_ds,
            eval_dataset=va_ds,
            data_collator=collator,
            tokenizer=tok,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )

        trainer.train()
        best_metrics = trainer.evaluate()
        print("Best eval:", best_metrics)

        cv_rows.append({
            "fold": fi + 1,
            "train_size": len(df_tr),
            "val_size": len(df_va),
            "best_eval_loss": float(best_metrics.get("eval_loss", np.nan)),
            "best_eval_mse": float(best_metrics.get("eval_mse", np.nan)),
            "best_eval_pearson": float(best_metrics.get("eval_pearson", np.nan)) if not args.no_pearson else np.nan,
        })

        if args.save_best:
            tok.save_pretrained(fold_dir)
            save_file(trainer.model.state_dict(), os.path.join(fold_dir, "model.safetensors"))

        del trainer, model, tr_ds, va_ds, tok
        mps_cleanup()

    cv_df = pd.DataFrame(cv_rows)
    cv_path = os.path.join(args.out_root, "cv_summary.csv")
    cv_df.to_csv(cv_path, index=False)

    print("\n===== CV SUMMARY =====")
    print(cv_df)
    if not args.no_pearson:
        print("\nMean pearson:", float(cv_df["best_eval_pearson"].mean()))
        print("Std pearson:", float(cv_df["best_eval_pearson"].std(ddof=0)))
    print("Saved:", cv_path)


if __name__ == "__main__":
    main()
