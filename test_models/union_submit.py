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


class DebertaValenceSOTA(nn.Module):
    def __init__(self, base_model: str, vocab_size: int, hidden_dropout: float = 0.2, feat_dim: int = 4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        self.encoder.resize_token_embeddings(int(vocab_size))

        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout)

        self.feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(32, 16),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden + hidden + 16, 256),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(256, 1),
        )

    def forward(self, input_ids=None, attention_mask=None, feats=None, labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls = h[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        mean = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        x_txt = torch.cat([cls, mean], dim=1)
        x_txt = self.dropout(x_txt)

        if feats is None:
            feats = torch.zeros((x_txt.size(0), 4), device=x_txt.device, dtype=x_txt.dtype)
        feat_emb = self.feat_mlp(feats)

        x = torch.cat([x_txt, feat_emb], dim=1)
        pred = self.head(x).squeeze(-1)
        return {"logits": pred}


def pick_embedding_key(sd: dict) -> str:
    cands = [k for k in sd.keys() if "word_embeddings.weight" in k]
    if not cands:
        raise KeyError("No word_embeddings.weight key found in safetensors.")
    return cands[0]


def resolve_model_dir(model_dir: str) -> str:
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return model_dir

    hf_dir = os.path.join(model_dir, "_hf")
    if os.path.isdir(hf_dir):
        cands = []
        for name in os.listdir(hf_dir):
            if name.startswith("checkpoint-"):
                ckpt_dir = os.path.join(hf_dir, name)
                if os.path.exists(os.path.join(ckpt_dir, "model.safetensors")):
                    step = int(name.split("-")[-1])
                    cands.append((step, ckpt_dir))
        if cands:
            cands.sort(key=lambda x: x[0])
            best = cands[-1][1]
            print(f"[INFO] Using checkpoint: {best}")
            return best

    raise FileNotFoundError(f"Could not find model.safetensors in {model_dir} or {model_dir}/_hf/checkpoint-*/")


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


def build_dataset(df: pd.DataFrame, tok, text_col: str, maxlen: int):
    ds = Dataset.from_pandas(df)

    def tok_fn(batch):
        enc = tok(batch[text_col], truncation=True, max_length=maxlen)
        enc["feats"] = batch["__feats__"]
        return enc

    ds = ds.map(tok_fn, batched=True)
    keep = ["input_ids", "attention_mask", "feats"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds.set_format("torch")
    return ds


def build_arousal_from_sd(base_model: str, tok_vocab: int, sd: dict, hidden_dropout: float = 0.2, feat_dim: int = 4):
    if "gru.weight_hh_l0" not in sd:
        raise KeyError("Checkpoint does not contain gru.weight_hh_l0; arousal model is not GRU-based.")

    hh = sd["gru.weight_hh_l0"]
    gru_hidden = int(hh.shape[1])

    feat_keys = [k for k in sd.keys() if k.startswith("feat_mlp.") and k.endswith(".weight")]
    if not feat_keys:
        raise KeyError("Checkpoint does not contain feat_mlp.*.weight")

    idxs = []
    for k in feat_keys:
        parts = k.split(".")
        if len(parts) >= 3:
            try:
                idxs.append((int(parts[1]), k))
            except:
                pass
    idxs.sort(key=lambda x: x[0])
    last_feat_w = sd[idxs[-1][1]]
    feat_out = int(last_feat_w.shape[0])

    if "head.0.weight" not in sd:
        raise KeyError("Checkpoint does not contain head.0.weight")
    head0 = sd["head.0.weight"]
    head_hidden = int(head0.shape[0])
    head_in = int(head0.shape[1])

    print(f"[INFO] Arousal arch from ckpt: gru_hidden={gru_hidden}, feat_out={feat_out}, head_in={head_in}, head_hidden={head_hidden}")

    class DebertaArousalAuto(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_model)
            self.encoder.resize_token_embeddings(int(tok_vocab))

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
                nn.Linear(32, feat_out),
                nn.GELU(),
            )

            self.head = nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(head_hidden, 1),
            )

        def forward(self, input_ids=None, attention_mask=None, feats=None, labels=None, **kwargs):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            h = out.last_hidden_state

            h = self.dropout(h)
            gru_out, _ = self.gru(h)

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (gru_out * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1.0))

            if feats is None:
                feats = torch.zeros((pooled.size(0), feat_dim), device=pooled.device, dtype=pooled.dtype)
            feat_emb = self.feat_mlp(feats)

            x = torch.cat([pooled, feat_emb], dim=1)
            pred = self.head(x).squeeze(-1)
            return {"logits": pred}

    return DebertaArousalAuto()


def load_valence(model_dir: str, base_model: str, device: str):
    model_dir = resolve_model_dir(model_dir)
    safe_path = os.path.join(model_dir, "model.safetensors")

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tok_vocab = len(tok)

    sd = load_file(safe_path, device="cpu")
    emb_key = pick_embedding_key(sd)
    ckpt_vocab = int(sd[emb_key].shape[0])

    model = DebertaValenceSOTA(base_model, vocab_size=tok_vocab, hidden_dropout=0.2, feat_dim=4)

    if ckpt_vocab != tok_vocab:
        print(f"[WARN] Vocab mismatch: ckpt={ckpt_vocab}, tok={tok_vocab}. Resize to ckpt for load, then back.")
        model.encoder.resize_token_embeddings(ckpt_vocab)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] Missing keys (valence):", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected keys (valence):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    if ckpt_vocab != tok_vocab:
        model.encoder.resize_token_embeddings(tok_vocab)

    model.to(device)
    model.eval()
    return tok, model


def load_arousal(model_dir: str, base_model: str, device: str):
    model_dir = resolve_model_dir(model_dir)
    safe_path = os.path.join(model_dir, "model.safetensors")

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    tok_vocab = len(tok)

    sd = load_file(safe_path, device="cpu")
    emb_key = pick_embedding_key(sd)
    ckpt_vocab = int(sd[emb_key].shape[0])

    model = build_arousal_from_sd(base_model, tok_vocab, sd, hidden_dropout=0.2, feat_dim=4)

    if ckpt_vocab != tok_vocab:
        print(f"[WARN] Vocab mismatch: ckpt={ckpt_vocab}, tok={tok_vocab}. Resize to ckpt for load, then back.")
        model.encoder.resize_token_embeddings(ckpt_vocab)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[WARN] Missing keys (arousal):", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] Unexpected keys (arousal):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    if ckpt_vocab != tok_vocab:
        model.encoder.resize_token_embeddings(tok_vocab)

    model.to(device)
    model.eval()
    return tok, model


def predict(tok, model, ds, batch: int):
    pad = DataCollatorWithPadding(tokenizer=tok)
    collator = FeatCollator(pad)

    targs = TrainingArguments(
        output_dir="._tmp_predict",
        per_device_eval_batch_size=batch,
        dataloader_drop_last=False,
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, data_collator=collator)
    pred_out = trainer.predict(ds)
    preds = np.asarray(pred_out.predictions).reshape(-1)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_model_dir", required=True)
    ap.add_argument("--aro_model_dir", required=True)
    ap.add_argument("--base_model", default="microsoft/deberta-v3-base")

    ap.add_argument("--test_csv", default="data/test_subtask1.csv")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force_cpu", action="store_true")
    ap.add_argument("--out_csv", default="submission.csv")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cpu" if args.force_cpu else (
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print("Device:", device)

    df = pd.read_csv(args.test_csv).copy()
    if args.text_col not in df.columns:
        raise ValueError(f"test_csv must have column '{args.text_col}'. Found: {list(df.columns)}")

    if "text_id" not in df.columns:
        if "id" in df.columns:
            df["text_id"] = df["id"]
        else:
            df["text_id"] = [f"item_{i}" for i in range(len(df))]
    df["text_id"] = df["text_id"].astype(str)
    if "user_id" in df.columns:
        df["user_id"] = df["user_id"].astype(str)

    df["__feats__"] = list(extract_feats(df[args.text_col].tolist()))

    tok_v, model_v = load_valence(args.val_model_dir, args.base_model, device)
    ds_v = build_dataset(df[[args.text_col, "__feats__"]].copy(), tok_v, args.text_col, args.maxlen)
    pred_v = predict(tok_v, model_v, ds_v, args.batch)

    tok_a, model_a = load_arousal(args.aro_model_dir, args.base_model, device)
    ds_a = build_dataset(df[[args.text_col, "__feats__"]].copy(), tok_a, args.text_col, args.maxlen)
    pred_a = predict(tok_a, model_a, ds_a, args.batch)

    if np.isnan(pred_v).any() or np.isnan(pred_a).any():
        raise RuntimeError("NaNs found in predictions. Not writing submission.")

    out = pd.DataFrame({
        "text_id": df["text_id"].tolist(),
        "pred_valence": pred_v.astype(float),
        "pred_arousal": pred_a.astype(float),
    })
    if "user_id" in df.columns:
        out.insert(0, "user_id", df["user_id"].tolist())
        out = out[["user_id", "text_id", "pred_valence", "pred_arousal"]]
    else:
        out = out[["text_id", "pred_valence", "pred_arousal"]]

    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv, "rows:", len(out))


if __name__ == "__main__":
    main()
