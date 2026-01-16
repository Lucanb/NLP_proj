import argparse
import os
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
)
from safetensors.torch import load_file


class DebertaRegressor(nn.Module):
    def __init__(self, base_model: str, vocab_size: int, hidden_dropout: float = 0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.encoder.resize_token_embeddings(int(vocab_size))

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

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = self.dropout(cls)
        v = self.head_v(x).squeeze(-1)
        a = self.head_a(x).squeeze(-1)
        preds = torch.stack([v, a], dim=1)
        return {"logits": preds}


def pick_embedding_key(sd: dict) -> str:
    preferred = [
        "encoder.embeddings.word_embeddings.weight",
        "encoder.deberta.embeddings.word_embeddings.weight",
    ]
    for k in preferred:
        if k in sd:
            return k

    cands = [k for k in sd.keys() if k.endswith("embeddings.word_embeddings.weight")]
    if cands:
        return cands[0]

    cands = [k for k in sd.keys() if "word_embeddings.weight" in k]
    if cands:
        return cands[0]

    raise KeyError(
        "Nu gasesc cheia embedding-ului în model.safetensors. "
        "Tipareste primele 50 keys din sd.keys() ca să ajustam automat."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Folderul cu tokenizer + model.safetensors (outdir-ul de la train)")
    ap.add_argument("--base_model", default="microsoft/deberta-v3-base", help="Modelul de bază folosit la train")
    ap.add_argument("--test_csv", default="data/test_subtask1.csv")
    ap.add_argument("--out_csv", default="pred_subtask1.csv")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cpu" if args.force_cpu else (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    safe_path = os.path.join(args.model_dir, "model.safetensors")
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"Missing {safe_path}")

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tok_vocab = len(tok)

    sd = load_file(safe_path, device="cpu")
    emb_key = pick_embedding_key(sd)
    ckpt_vocab = int(sd[emb_key].shape[0])

    model = DebertaRegressor(args.base_model, vocab_size=tok_vocab, hidden_dropout=0.2)

    if ckpt_vocab != tok_vocab:
        print(f"[WARN] Vocab mismatch: ckpt={ckpt_vocab}, tok={tok_vocab}. "
              f"Resizing to ckpt for load, then back to tokenizer vocab.")
        model.encoder.resize_token_embeddings(ckpt_vocab)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing[:20], "..." if len(missing) > 20 else "")
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")

    if ckpt_vocab != tok_vocab:
        model.encoder.resize_token_embeddings(tok_vocab)

    model.to(device)
    model.eval()

    collator = DataCollatorWithPadding(tokenizer=tok)

    df = pd.read_csv(args.test_csv).copy()
    if "text" not in df.columns:
        raise ValueError(f"test_csv must have column 'text'. Found: {list(df.columns)}")

    if "text_id" not in df.columns:
        if "id" in df.columns:
            df["text_id"] = df["id"]
        else:
            df["text_id"] = [f"item_{i}" for i in range(len(df))]
    df["text_id"] = df["text_id"].astype(str)

    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)

    test_ds = Dataset.from_pandas(df)

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.maxlen)

    test_ds = test_ds.map(tok_fn, batched=True)
    keep_cols = ["input_ids", "attention_mask"]
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])
    test_ds.set_format("torch")

    targs = TrainingArguments(
        output_dir="._tmp_predict",
        per_device_eval_batch_size=args.batch,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collator,
    )

    pred_out = trainer.predict(test_ds)
    preds = np.asarray(pred_out.predictions)

    if preds.ndim != 2 or preds.shape[1] != 2:
        raise RuntimeError(f"Unexpected predictions shape: {preds.shape}")

    out = pd.DataFrame({
        "text_id": df["text_id"].tolist(),
        "pred_valence": preds[:, 0].astype(float),
        "pred_arousal": preds[:, 1].astype(float),
    })

    if out["pred_valence"].isna().any() or out["pred_arousal"].isna().any():
        raise RuntimeError("NaNs found in predictions. Not writing submission.")

    if "user_id" in df.columns:
        out.insert(0, "user_id", df["user_id"].tolist())

    out.to_csv(args.out_csv, index=False)
    print(" Wrote:", args.out_csv, "rows:", len(out))


if __name__ == "__main__":
    main()
