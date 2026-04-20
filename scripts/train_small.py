"""Tiny end-to-end sanity run on synthetic data.

Usage (from the project root, with the venv active):

    python scripts/train_small.py

Prints per-step loss components and verifies that the total NLL decreases.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import time

import torch

from src.loss import compute_loss
from src.model import NeuralSVMPP, ModelConfig
from src.synth import GroundTruthParams, make_dataset
from src.utils import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seq", type=int, default=8)
    parser.add_argument("--T", type=float, default=4.0)
    parser.add_argument("--dt-sim", type=float, default=0.02)
    parser.add_argument("--dt-train", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)

    gt = GroundTruthParams()
    ds = make_dataset(n_seq=args.n_seq, T=args.T, dt_sim=args.dt_sim, params=gt)
    total_events = sum(len(s.event_times) for s in ds.sequences)
    print(f"[data] {args.n_seq} sequences, {total_events} events, T={args.T}")

    cfg = ModelConfig(d_z=gt.d_z, d_v=gt.d_v, d_x=gt.d_x)
    model = NeuralSVMPP(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    t_start = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        per_seq_totals = []
        per_seq_nll_time = []
        per_seq_nll_mark = []
        per_seq_surv = []
        for seq in ds.sequences:
            res = model.forward_sequence(
                seq.event_times, seq.event_marks,
                t0=seq.t0, T=seq.T, dt=args.dt_train,
            )
            lc = compute_loss(model, res, seq.event_marks, beta=args.beta)
            per_seq_totals.append(lc.total)
            per_seq_nll_time.append(lc.nll_time.detach())
            per_seq_nll_mark.append(lc.nll_mark.detach())
            per_seq_surv.append(lc.survival.detach())
        loss = torch.stack(per_seq_totals).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"[step {step:3d}] loss={loss.item():+.3f}  "
                f"nll_time={torch.stack(per_seq_nll_time).mean().item():+.3f}  "
                f"nll_mark={torch.stack(per_seq_nll_mark).mean().item():+.3f}  "
                f"surv={torch.stack(per_seq_surv).mean().item():+.3f}  "
                f"κ̄={model.kappa.mean().item():.3f}  "
                f"v̄̄={model.v_bar.mean().item():.3f}  "
                f"ρ={model.rho.item():+.3f}"
            )

    dt_train = time.time() - t_start
    print(f"[train] {args.steps} steps in {dt_train:.1f}s")
    print(f"[train] first loss {losses[0]:.3f} → final {losses[-1]:.3f}")
    # Crude health check: loss should go down on synthetic data.
    improved = (losses[-1] < losses[0])
    print(f"[train] loss decreased: {improved}")


if __name__ == "__main__":
    main()
