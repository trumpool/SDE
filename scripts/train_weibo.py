"""End-to-end training on one month of Weibo-COV 2.0 data (default: 2019-12).

Usage:
    python scripts/train_weibo.py                       # defaults
    python scripts/train_weibo.py --max-seqs 30 --steps 100
    python scripts/train_weibo.py --csv data/raw/2019-12.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.loss import compute_loss
from src.model import NeuralSVMPP, ModelConfig
from src.utils import set_seed, select_device
from src.weibo_data import MARK_DIM, sequences_from_path, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/2019-12.csv")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-seqs", type=int, default=30,
                        help="limit number of user sequences for sanity runs")
    parser.add_argument("--dt-train", type=float, default=0.05,
                        help="SDE step size in DAYS (0.05 ≈ 72 min)")
    parser.add_argument("--d-z", type=int, default=8)
    parser.add_argument("--d-v", type=int, default=8)
    parser.add_argument("--d-x", type=int, default=32,
                        help="mark dim (BERT is projected from 768 down to this)")
    parser.add_argument("--bert-cache", type=str, default=None,
                        help="path to encoded [CLS] cache; omit to use placeholder features")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="cpu / mps / cuda; default = auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"[env] device={device}")

    # --- load ---
    csv_path = os.path.join(_ROOT, args.csv) if not os.path.isabs(args.csv) else args.csv
    bert_cache = (
        os.path.join(_ROOT, args.bert_cache)
        if args.bert_cache and not os.path.isabs(args.bert_cache)
        else args.bert_cache
    )
    sequences = sequences_from_path(
        csv_path, min_length=args.min_length, max_sequences=args.max_seqs,
        bert_cache_path=bert_cache, seed=args.seed,
    )
    print("[data]")
    print(summarize(sequences))
    mark_in_dim = sequences[0].event_marks.shape[-1] if sequences else MARK_DIM
    using_bert = bert_cache is not None
    print(f"[data] mark input dim: {mark_in_dim} ({'BERT [CLS]' if using_bert else 'placeholder features'})")

    for s in sequences:
        s.event_times = s.event_times.to(device)
        s.event_marks = s.event_marks.to(device)

    # --- model ---
    cfg = ModelConfig(
        d_z=args.d_z, d_v=args.d_v,
        d_x=args.d_x if using_bert else MARK_DIM,
        bert_dim=mark_in_dim if using_bert else None,
    )
    model = NeuralSVMPP(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- train ---
    losses = []
    t_start = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        totals = []
        nll_ts = []
        nll_ms = []
        survs = []
        for seq in sequences:
            res = model.forward_sequence(
                seq.event_times, seq.event_marks,
                t0=seq.t0, T=seq.T, dt=args.dt_train,
            )
            lc = compute_loss(model, res, seq.event_marks, beta=args.beta)
            # Normalize by sequence length so cross-sequence scales are comparable.
            totals.append(lc.total / max(lc.n_events, 1))
            nll_ts.append(lc.nll_time.detach() / max(lc.n_events, 1))
            nll_ms.append(lc.nll_mark.detach() / max(lc.n_events, 1))
            survs.append(lc.survival.detach() / max(lc.n_events, 1))
        loss = torch.stack(totals).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"[step {step:3d}] loss/ev={loss.item():+.3f}  "
                f"nll_time/ev={torch.stack(nll_ts).mean().item():+.3f}  "
                f"nll_mark/ev={torch.stack(nll_ms).mean().item():+.3f}  "
                f"surv/ev={torch.stack(survs).mean().item():+.3f}  "
                f"κ̄={model.kappa.mean().item():.3f}  "
                f"v̄̄={model.v_bar.mean().item():.3f}  "
                f"ρ={model.rho.item():+.3f}"
            )

    dt_train = time.time() - t_start
    print(f"[train] {args.steps} steps in {dt_train:.1f}s")
    print(f"[train] first loss/ev {losses[0]:.3f} → final {losses[-1]:.3f}")


if __name__ == "__main__":
    main()
