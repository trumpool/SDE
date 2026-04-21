"""Encode every post in a Weibo-COV monthly CSV into [CLS] embeddings.

Saves a single cache file whose structure is
    {"ids": List[str],
     "embeddings": torch.HalfTensor of shape (N, 768),
     "model_name": str, "max_length": int}
The ``ids`` list aligns to the row order of the CSV; downstream code does
``id → index`` lookup to attach embeddings to events.

Usage:
    python scripts/encode_weibo.py                          # default: 2019-12
    python scripts/encode_weibo.py --csv data/raw/2020-01.csv
    python scripts/encode_weibo.py --max-rows 500           # sanity subset
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pandas as pd
import torch

from src.text_encoder import EncoderConfig, WeiboTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/2019-12.csv")
    parser.add_argument("--out", type=str, default=None,
                        help="default: data/encoded/<stem>_cls.pt")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="truncate inputs for debug; default = all")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_ROOT, args.csv)
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    out_path = args.out or os.path.join(_ROOT, "data", "encoded", f"{stem}_cls.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --- load ---
    df = pd.read_csv(csv_path, usecols=["_id", "content"], low_memory=False)
    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].reset_index(drop=True)
    ids = df["_id"].astype(str).tolist()
    texts = df["content"].fillna("").astype(str).tolist()
    print(f"[encode] {len(texts)} rows from {csv_path}")

    # --- encoder ---
    cfg = EncoderConfig(max_length=args.max_length, batch_size=args.batch_size,
                        device=args.device)
    enc = WeiboTextEncoder(cfg)

    # --- encode with progress ---
    t0 = time.time()
    embeds = enc.encode(texts, progress=True)
    dt = time.time() - t0
    print(f"[encode] device={enc.device}, {len(texts)} texts in {dt:.1f}s "
          f"({len(texts)/dt:.1f} texts/s)")

    # --- save ---
    payload = {
        "ids": ids,
        "embeddings": embeds,            # (N, 768) float16
        "model_name": cfg.model_name,
        "max_length": cfg.max_length,
    }
    torch.save(payload, out_path)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[encode] saved → {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
