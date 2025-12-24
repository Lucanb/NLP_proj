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

# --- CONFIGURARE MODEL ---
class DebertaRegressor(nn.Module):
    def __init__(self, model_name: str, hidden_dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        # Separate heads (V and A)
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
            # Weighted SmoothL1: emphasize arousal
            lv = nn.functional.smooth_l1_loss(preds[:, 0], labels[:, 0])
            la = nn.functional.smooth_l1_loss(preds[:, 1], labels[:, 1])
            loss = 1.0 * lv + 1.25 * la

        return {"loss": loss, "logits": preds}

# Metrica simplă pentru afișare în timpul antrenării (progressbar)
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    # Putem păstra scipy aici pentru rapiditate în timpul epocilor
    rv = pearsonr(labels[:, 0], preds[:, 0])[0]
    ra = pearsonr(labels[:, 1], preds[:, 1])[0]
    return {"pearson_valence": rv, "pearson_arousal": ra, "pearson_mean": (rv + ra) / 2}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to csv data")
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--outdir", default="models/st1_deberta")
    
    # ARGUMENT NOU: Calea către repo-ul de evaluare
    ap.add_argument("--eval_repo", required=True, help="Absolute path to 'semeval2026-task2-eval' folder")
    
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=8) # Am marit putin batch-ul default
    ap.add_argument("--maxlen", type=int, default=192)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--disable_mps_watermark", action="store_true")

    args = ap.parse_args()

    # --- IMPORTARE DINAMICĂ EVALUATOR OFICIAL ---
    if not os.path.exists(args.eval_repo):
        raise FileNotFoundError(f"Nu găsesc folderul de evaluare la: {args.eval_repo}")
    
    sys.path.append(args.eval_repo)
    try:
        import eval as official_eval
        print(f"✅ Evaluator oficial încărcat din: {args.eval_repo}")
    except ImportError as e:
        print(f"❌ Eroare critică: Nu pot importa 'eval.py'. Verifică calea. ({e})")
        sys.exit(1)

    if args.disable_mps_watermark:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    set_seed(args.seed)

    # --- ÎNCĂRCARE DATE ---
    df = pd.read_csv(args.data)
    # Ne asigurăm că avem coloanele necesare
    required_cols = ["text", "valence", "arousal"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Verificăm dacă există o coloană de ID, dacă nu, folosim indexul
    if "text_id" not in df.columns:
        if "id" in df.columns:
            df["text_id"] = df["id"]
        else:
            df["text_id"] = [f"item_{i}" for i in range(len(df))]

    n = len(df)
    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int((1.0 - args.val_ratio) * n)
    train_idx, val_idx = idx[:cut], idx[cut:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.maxlen)

    train_ds = train_ds.map(tok_fn, batched=True)
    val_ds = val_ds.map(tok_fn, batched=True)

    def add_labels(batch):
        batch["labels"] = np.stack([batch["valence"], batch["arousal"]], axis=1).astype(np.float32)
        return batch

    train_ds = train_ds.map(add_labels, batched=True)
    val_ds = val_ds.map(add_labels, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    # Curățăm coloanele care nu intră în model
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    model = DebertaRegressor(args.model, hidden_dropout=0.2)

    collator = DataCollatorWithPadding(tokenizer=tok)

    os.makedirs(args.outdir, exist_ok=True)

    targs = TrainingArguments(
        output_dir=args.outdir,
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
        fp16=torch.cuda.is_available(), # Activează FP16 dacă ai GPU NVIDIA
        
        optim="adamw_torch",
        save_total_limit=2 # Nu salva tone de checkpoint-uri
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

    # --- RULARE ANTRENARE ---
    print("\n🚀 Începe antrenarea...")
    trainer.train()
    
    # --- PREPARARE DATE PENTRU EVALUAREA OFICIALĂ ---
    print("\n📊 Generare predicții pe setul de validare...")
    pred_output = trainer.predict(val_ds)
    preds = pred_output.predictions # Shape: (N, 2) -> [Valence, Arousal]
    
    # Extragem metadatele originale pentru setul de validare
    # Ne asigurăm că ordinea e păstrată (trainer.predict păstrează ordinea din dataset)
    
    val_ids = val_df["text_id"].astype(str).tolist()
    
    # VALENCE
    gold_valence = val_df["valence"].tolist()
    pred_valence = preds[:, 0].tolist()
    
    # AROUSAL
    gold_arousal = val_df["arousal"].tolist()
    pred_arousal = preds[:, 1].tolist()

    # --- APELARE SCRIPT OFICIAL (TASK 2) ---
    print("\n" + "="*40)
    print("🏆 REZULTATE OFICIALE (semeval script)")
    print("="*40)

    try:
        # Evaluare VALENCE
        print("\n--- Valence Results ---")
        # Funcția task2_correlation cere: (ids, predictions, labels)
        res_v = official_eval.task2_correlation(val_ids, pred_valence, gold_valence)
        print(res_v)

        # Evaluare AROUSAL
        print("\n--- Arousal Results ---")
        res_a = official_eval.task2_correlation(val_ids, pred_arousal, gold_arousal)
        print(res_a)

        # Salvare rezultate oficiale într-un JSON
        final_metrics = {
            "official_valence": res_v,
            "official_arousal": res_a
        }
        with open(os.path.join(args.outdir, "official_metrics.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)

    except Exception as e:
        print(f"⚠️ Eroare la rularea scriptului oficial: {e}")
        print("Verifică dacă input-urile conțin valori NaN sau liste goale.")

    # --- SALVARE CSV PENTRU ANALIZĂ ---
    out_csv = os.path.join(args.outdir, "val_predictions_official_format.csv")
    
    # Construim DataFrame-ul final
    res_df = pd.DataFrame({
        "id": val_ids,
        "true_valence": gold_valence,
        "pred_valence": pred_valence,
        "true_arousal": gold_arousal,
        "pred_arousal": pred_arousal
    })
    
    # Adăugăm și textul ca să poți vedea unde greșește modelul
    if "text" in val_df.columns:
        res_df["text"] = val_df["text"].tolist()

    res_df.to_csv(out_csv, index=False)
    print(f"\n✅ Predicții salvate în: {out_csv}")

    # Salvare model
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)

if __name__ == "__main__":
    main()
