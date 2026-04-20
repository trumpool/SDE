"""Generate and save a synthetic dataset to ``data/synth_small.pt``.

Quick inspection:

    python scripts/make_synth.py
    python -c "import torch; d=torch.load('data/synth_small.pt'); print(len(d['sequences']))"
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse

import torch

from src.synth import GroundTruthParams, make_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seq", type=int, default=16)
    parser.add_argument("--T", type=float, default=5.0)
    parser.add_argument("--dt-sim", type=float, default=0.01)
    parser.add_argument("--out", type=str, default="data/synth_small.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params = GroundTruthParams()
    ds = make_dataset(
        n_seq=args.n_seq, T=args.T, dt_sim=args.dt_sim, params=params, seed=args.seed
    )
    payload = {
        "sequences": [
            {
                "t0": s.t0, "T": s.T,
                "event_times": s.event_times, "event_marks": s.event_marks,
            }
            for s in ds.sequences
        ],
        "params": vars(params),
        "W_j": ds.W_j, "b_j": ds.b_j, "W_x": ds.W_x,
    }
    out_path = os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(payload, out_path)
    total_events = sum(len(s.event_times) for s in ds.sequences)
    print(f"[make_synth] saved {args.n_seq} sequences, {total_events} events → {args.out}")


if __name__ == "__main__":
    main()
