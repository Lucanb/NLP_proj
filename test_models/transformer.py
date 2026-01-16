import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import pearsonr

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed
)

from sklearn.model_selection import GroupKFold, KFold


class DebertaRegressor(nn.Module):
    def __init__(self, model_name: str, hidden_dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        self.head_v = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.head_a = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden // 2, 1),
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.encoder.gradient_checkpointing_enable()
            else:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)

        v = self.head_v(x).squeeze(-1)
        a = self.head_a(x).squeeze(-1)
        preds = torch.stack([v, a], dim=1)

        loss = None
        if labels is not None:
            lv = nn.functional.smooth_l1_loss(preds[:, 0], labels[:, 0])
            la = nn.functional.smooth_l1_loss(preds[:, 1], labels[:, 1])
            loss = 1.0 * lv + 1.25 * la

        return {"loss": loss, "logits": preds}


def _safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return 0.0
    r = pearsonr(y_true, y_pred)[0]
    return 0.0 if np.isnan(r) else float(r)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    rv = _safe_pearson(labels[:, 0], preds[:, 0])
    ra = _safe_pearson(labels[:, 1], preds[:, 1])
    return {"pearson_valence": rv, "pearson_arousal": ra, "pearson_mean": (rv + ra) / 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to csv data")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--outdir", default="models/st1_deberta_cv")

    ap.add_argument("--eval_repo", required=True, help="Absolute path to 'semeval2026-task2-eval' folder")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=128)

    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--disable_mps_watermark", action="store_true")

    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--group_col", default="user_id")
    ap.add_argument("--min_group_size", type=int, default=1)

    args = ap.parse_args()

    if not os.path.exists(args.eval_repo):
        raise FileNotFoundError(f"Nu găsesc folderul de evaluare la: {args.eval_repo}")

    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print(f"Evaluator oficial încărcat din: {args.eval_repo}")
    except ImportError as e:
        print(f"Eroare critică: Nu pot importa 'eval.py'. Verifică calea. ({e})")
        sys.exit(1)

    if args.disable_mps_watermark:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    set_seed(args.seed)

    df = pd.read_csv(args.data)
    required_cols = ["text", "valence", "arousal"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if "text_id" not in df.columns:
        if "id" in df.columns:
            df["text_id"] = df["id"]
        else:
            df["text_id"] = [f"item_{i}" for i in range(len(df))]

    use_groups = args.group_col in df.columns
    if use_groups:
        df[args.group_col] = df[args.group_col].astype(str)
        if args.min_group_size > 1:
            vc = df[args.group_col].value_counts()
            keep_groups = set(vc[vc >= args.min_group_size].index)
            df = df[df[args.group_col].isin(keep_groups)].reset_index(drop=True)

    n = len(df)
    if n < 10:
        print(f"Ai doar {n} exemple dupa filtrare. CV poate fi instabil.")

    if use_groups:
        groups = df[args.group_col].values
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            raise ValueError(f"Prea putine grupuri în '{args.group_col}' (ai {len(unique_groups)}).")
        if len(unique_groups) < args.cv_folds:
            args.cv_folds = max(2, len(unique_groups))
            print(f" Prea putini useri pentru folds; setez cv_folds={args.cv_folds}")
        splitter = GroupKFold(n_splits=args.cv_folds)
        folds = list(splitter.split(np.zeros(n), groups=groups))
        print(f" GroupKFold cu {args.cv_folds} folds pe '{args.group_col}'")
    else:
        splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        folds = list(splitter.split(np.zeros(n)))
        print(f" Nu am '{args.group_col}' în CSV. Folosesc KFold({args.cv_folds}) random.")

    os.makedirs(args.outdir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.maxlen)

    def add_labels(batch):
        batch["labels"] = np.stack([batch["valence"], batch["arousal"]], axis=1).astype(np.float32)
        return batch

    keep_cols = ["input_ids", "attention_mask", "labels"]

    all_fold_metrics = []
    oof_pred_v = np.zeros(len(df), dtype=np.float32)
    oof_pred_a = np.zeros(len(df), dtype=np.float32)

    for fold_i, (train_idx, val_idx) in enumerate(folds, start=1):
        print("\n" + "=" * 60)
        print(f"🧪 FOLD {fold_i}/{len(folds)} | train={len(train_idx)} | val={len(val_idx)}")
        print("=" * 60)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_ds = Dataset.from_pandas(train_df).map(tok_fn, batched=True).map(add_labels, batched=True)
        val_ds = Dataset.from_pandas(val_df).map(tok_fn, batched=True).map(add_labels, batched=True)

        train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
        val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

        train_ds.set_format("torch")
        val_ds.set_format("torch")

        model = DebertaRegressor(args.model, hidden_dropout=0.2)

        fold_out = os.path.join(args.outdir, f"fold_{fold_i}")
        os.makedirs(fold_out, exist_ok=True)

        targs = TrainingArguments(
            output_dir=fold_out,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch * 2,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,

            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="pearson_mean",
            greater_is_better=True,

            logging_steps=50,
            report_to="none",
            dataloader_num_workers=args.num_workers,
            fp16=torch.cuda.is_available(),

            optim="adamw_torch",
            save_total_limit=1
        )

        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tok,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

        print("\n Incepe antrenarea...")
        trainer.train()

        eval_metrics = trainer.evaluate()
        all_fold_metrics.append({"fold": fold_i, **eval_metrics})

        print("\n Generare predictii pe setul de validare (fold)...")
        pred_output = trainer.predict(val_ds)
        preds = np.asarray(pred_output.predictions)

        oof_pred_v[val_idx] = preds[:, 0].astype(np.float32)
        oof_pred_a[val_idx] = preds[:, 1].astype(np.float32)

        val_ids = df.iloc[val_idx]["text_id"].astype(str).tolist()
        gold_valence = df.iloc[val_idx]["valence"].tolist()
        gold_arousal = df.iloc[val_idx]["arousal"].tolist()

        pred_valence = preds[:, 0].tolist()
        pred_arousal = preds[:, 1].tolist()

        print("\n" + "=" * 40)
        print(f"🏆 REZULTATE OFICIALE (fold {fold_i})")
        print("=" * 40)
        try:
            print("\n--- Valence Results ---")
            res_v = official_eval.task2_correlation(val_ids, pred_valence, gold_valence)
            print(res_v)

            print("\n--- Arousal Results ---")
            res_a = official_eval.task2_correlation(val_ids, pred_arousal, gold_arousal)
            print(res_a)

            with open(os.path.join(fold_out, "official_metrics.json"), "w") as f:
                json.dump({"official_valence": res_v, "official_arousal": res_a}, f, indent=2)
        except Exception as e:
            print(f"Eroare la rularea scriptului oficial pe fold {fold_i}: {e}")

        fold_csv = os.path.join(fold_out, "oof_val_predictions.csv")
        pd.DataFrame({
            "text_id": val_ids,
            "true_valence": gold_valence,
            "pred_valence": pred_valence,
            "true_arousal": gold_arousal,
            "pred_arousal": pred_arousal,
            "text": df.iloc[val_idx]["text"].tolist(),
        }).to_csv(fold_csv, index=False)

        trainer.save_model(fold_out)

    rv = _safe_pearson(df["valence"].values, oof_pred_v)
    ra = _safe_pearson(df["arousal"].values, oof_pred_a)
    mean_score = (rv + ra) / 2.0

    print("\n" + "=" * 60)
    print("🏁 OOF RESULTS (CV estimate)")
    print("=" * 60)
    print(f"Pearson Valence:  {rv:.4f}")
    print(f"Pearson Arousal:  {ra:.4f}")
    print(f"Pearson Mean:     {mean_score:.4f}")

    official_oof = {}
    try:
        ids_all = df["text_id"].astype(str).tolist()
        res_v = official_eval.task2_correlation(ids_all, oof_pred_v.tolist(), df["valence"].tolist())
        res_a = official_eval.task2_correlation(ids_all, oof_pred_a.tolist(), df["arousal"].tolist())
        official_oof = {"official_oof_valence": res_v, "official_oof_arousal": res_a}
        print("\n🏆 OFFICIAL on OOF")
        print("Valence:", res_v)
        print("Arousal:", res_a)
    except Exception as e:
        print("\n Official eval on OOF failed:", e)

    summary_path = os.path.join(args.outdir, "cv_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "oof_pearson_valence": rv,
            "oof_pearson_arousal": ra,
            "oof_pearson_mean": mean_score,
            "fold_metrics": all_fold_metrics,
            "cv_folds": len(folds),
            "group_col_used": args.group_col if use_groups else None,
            **official_oof
        }, f, indent=2)

    oof_path = os.path.join(args.outdir, "oof_predictions.csv")
    pd.DataFrame({
        "text_id": df["text_id"].astype(str).values,
        "true_valence": df["valence"].values,
        "pred_valence": oof_pred_v,
        "true_arousal": df["arousal"].values,
        "pred_arousal": oof_pred_a,
        "text": df["text"].values
    }).to_csv(oof_path, index=False)

    print(f"\n CV summary: {summary_path}")
    print(f" OOF preds:  {oof_path}")

    tok.save_pretrained(args.outdir)


if __name__ == "__main__":
    main()
