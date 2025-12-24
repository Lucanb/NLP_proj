import argparse
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--maxlen", type=int, default=256)
    args = ap.parse_args()

    df = pd.read_csv(args.data).dropna(subset=["text"]).reset_index(drop=True)
    texts = df["text"].astype(str).tolist()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    lens = []
    trunc_hits = 0
    bs = 64

    for i in range(0, len(texts), bs):
        batch = texts[i:i+bs]
        enc = tok(batch, truncation=False, add_special_tokens=True)
        l = [len(x) for x in enc["input_ids"]]
        lens.extend(l)

        enc_tr = tok(batch, truncation=True, max_length=args.maxlen, add_special_tokens=True)
        ltr = [len(x) for x in enc_tr["input_ids"]]
        trunc_hits += sum(1 for a, b in zip(l, ltr) if a > b)

    lens = np.asarray(lens, dtype=np.int32)
    p = lambda q: int(np.percentile(lens, q))

    print("N =", len(lens))
    print("min/mean/max =", int(lens.min()), float(lens.mean()), int(lens.max()))
    print("p50/p75/p90/p95/p99 =", p(50), p(75), p(90), p(95), p(99))
    print(f"maxlen={args.maxlen} truncation hits =", int(trunc_hits),
          f"({100.0*trunc_hits/len(lens):.2f}%)")

    # quick recommendation
    rec = p(95)
    if rec < 96: rec = 96
    if rec > 256: rec = 256
    print("recommended maxlen (≈p95, clipped to [96,256]) =", rec)

if __name__ == "__main__":
    main()

