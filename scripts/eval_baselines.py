"""Compare the latent SVMPP model against simple baselines on Weibo 2019-12.

We fix a 80/20 train/test split (stratified by sequence length), fit each
model on train, evaluate per-event NLL on test, and print a comparison
table. Baselines:

    B1. Homogeneous Poisson + standalone GMM marks
    B2. Hawkes (exponential kernel) + standalone GMM marks
    B3. NeuralSVMPP with the volatility channel frozen (ablation)
    M.  NeuralSVMPP full model

All four use the **same BERT cache** so the mark features are identical
across rows. Survival-integral and timing-NLL are reported separately so
the comparison is interpretable.

Usage:
    python scripts/eval_baselines.py                 # all baselines + full model
    python scripts/eval_baselines.py --skip-bert     # skip the M / B3 rows
                                                     #   (faster; uses placeholder
                                                     #   marks if no cache)
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

from src.baselines import HawkesExponential, HomogeneousPoisson, NLLMetrics
from src.loss import compute_loss
from src.model import ModelConfig, NeuralSVMPP
from src.utils import set_seed
from src.weibo_data import MARK_DIM, sequences_from_path, summarize, train_test_split


# ---------------------------------------------------------------------------
# SVMPP wrappers exposing the same evaluate() surface
# ---------------------------------------------------------------------------


def train_svmpp(
    train_seqs, *, cfg: ModelConfig, steps: int, lr: float, beta: float,
    freeze_vol: bool, dt_train: float, device: torch.device,
) -> NeuralSVMPP:
    model = NeuralSVMPP(cfg).to(device)
    if freeze_vol:
        model.intensity_net.freeze_volatility_channel()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for step in range(steps):
        opt.zero_grad()
        per_seq_totals = []
        for seq in train_seqs:
            res = model.forward_sequence(
                seq.event_times, seq.event_marks,
                t0=seq.t0, T=seq.T, dt=dt_train,
            )
            lc = compute_loss(model, res, seq.event_marks, beta=beta)
            per_seq_totals.append(lc.total / max(lc.n_events, 1))
        loss = torch.stack(per_seq_totals).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model


@torch.no_grad()
def evaluate_svmpp(
    model: NeuralSVMPP, test_seqs, *, beta: float, dt_train: float,
) -> NLLMetrics:
    total_nll_time = 0.0
    total_nll_mark = 0.0
    total_survival = 0.0
    n_events = 0
    for seq in test_seqs:
        res = model.forward_sequence(
            seq.event_times, seq.event_marks,
            t0=seq.t0, T=seq.T, dt=dt_train,
        )
        lc = compute_loss(model, res, seq.event_marks, beta=beta)
        total_nll_time += float(lc.nll_time)
        total_nll_mark += float(lc.nll_mark)
        total_survival += float(lc.survival)
        n_events += int(lc.n_events)
    n_events = max(n_events, 1)
    return NLLMetrics(
        nll_time_per_event=total_nll_time / n_events,
        nll_mark_per_event=total_nll_mark / n_events,
        survival_per_event=total_survival / n_events,
        total_per_event=(total_nll_time + beta * total_nll_mark + total_survival) / n_events,
        n_events=n_events, n_sequences=len(test_seqs), beta=beta,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_table(rows):
    cols = ["model", "n_seq", "n_events", "nll_time/ev", "nll_mark/ev",
            "surv/ev", "total/ev"]
    widths = {c: max(len(c), max(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    hdr = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    print(hdr); print(sep)
    for r in rows:
        print(" | ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:+.3f}"
    return str(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/2019-12.csv")
    parser.add_argument("--bert-cache", type=str, default="data/encoded/2019-12_cls.pt")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--svmpp-steps", type=int, default=80)
    parser.add_argument("--svmpp-lr", type=float, default=3e-3)
    parser.add_argument("--svmpp-d-z", type=int, default=8)
    parser.add_argument("--svmpp-d-x", type=int, default=32)
    parser.add_argument("--svmpp-beta", type=float, default=0.1)
    parser.add_argument("--dt-train", type=float, default=0.05)
    parser.add_argument("--hawkes-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip-bert", action="store_true",
                        help="skip B3 + M (no SDE training); useful for a quick run")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(_ROOT, args.csv)
    bert_path = (args.bert_cache if os.path.isabs(args.bert_cache or "")
                 else os.path.join(_ROOT, args.bert_cache) if args.bert_cache else None)

    use_bert = bert_path and os.path.exists(bert_path) and not args.skip_bert
    print(f"[setup] csv={csv_path}")
    print(f"[setup] bert_cache={'<used>' if use_bert else '<skipped>'}")

    sequences = sequences_from_path(
        csv_path, min_length=args.min_length,
        bert_cache_path=bert_path if use_bert else None,
        seed=args.seed,
    )
    train_seqs, test_seqs = train_test_split(
        sequences, test_frac=args.test_frac, seed=args.seed,
    )
    print(f"[split] train: {len(train_seqs)} seqs / "
          f"{sum(len(s.event_times) for s in train_seqs)} events;  "
          f"test: {len(test_seqs)} seqs / "
          f"{sum(len(s.event_times) for s in test_seqs)} events")

    # ---- B1. Homogeneous Poisson ----
    print("\n[fit] B1 — Homogeneous Poisson + GMM marks ...")
    t0 = time.time()
    poisson = HomogeneousPoisson(d_x=train_seqs[0].event_marks.shape[-1])
    poisson.fit(train_seqs)
    poisson_metrics = poisson.evaluate(test_seqs)
    print(f"  done in {time.time()-t0:.1f}s.  μ̂ = {poisson.mu.item():.3f} events/day")

    # ---- B2. Hawkes (exponential kernel) ----
    print("\n[fit] B2 — Hawkes (exponential) + GMM marks ...")
    t0 = time.time()
    hawkes = HawkesExponential(d_x=train_seqs[0].event_marks.shape[-1])
    hawkes.fit(train_seqs, steps=args.hawkes_steps, verbose=True)
    hawkes_metrics = hawkes.evaluate(test_seqs)
    print(f"  done in {time.time()-t0:.1f}s.  μ={hawkes.mu.item():.3f}  "
          f"α={hawkes.alpha.item():.3f}  β={hawkes.beta.item():.3f}")

    rows = [
        poisson_metrics.as_row("B1 Poisson"),
        hawkes_metrics.as_row("B2 Hawkes(exp)"),
    ]

    # ---- B3 & M. SVMPP variants ----
    if use_bert:
        mark_in_dim = train_seqs[0].event_marks.shape[-1]
        for s in train_seqs + test_seqs:
            s.event_times = s.event_times.to(device)
            s.event_marks = s.event_marks.to(device)
        cfg = ModelConfig(
            d_z=args.svmpp_d_z, d_v=args.svmpp_d_z, d_x=args.svmpp_d_x,
            bert_dim=mark_in_dim,
        )

        for name, freeze in [("B3 SVMPP no-vol", True), ("M  SVMPP full", False)]:
            print(f"\n[fit] {name}  ({args.svmpp_steps} steps, β={args.svmpp_beta}) ...")
            t0 = time.time()
            model = train_svmpp(
                train_seqs, cfg=cfg, steps=args.svmpp_steps, lr=args.svmpp_lr,
                beta=args.svmpp_beta, freeze_vol=freeze, dt_train=args.dt_train,
                device=device,
            )
            print(f"  done in {time.time()-t0:.1f}s.")
            metrics = evaluate_svmpp(
                model, test_seqs, beta=args.svmpp_beta, dt_train=args.dt_train,
            )
            rows.append(metrics.as_row(name))
            # Print a couple of learned params for context.
            print(f"  κ̄={model.kappa.mean().item():.3f}  "
                  f"v̄̄={model.v_bar.mean().item():.3f}  "
                  f"ρ={model.rho.item():+.3f}  "
                  f"|a_vol|={model.intensity_net.a_vol.abs().mean().item():.3f}")

    # ---- Report ----
    print("\n========= TEST SET COMPARISON (per-event NLL) =========")
    print_table(rows)
    print("\n(timing-NLL = −Σlog λ(t_i)/n_events;  surv = ∫λ dt / n_events;  "
          "mark-NLL = −Σlog p(x_i)/n_events;  total = nll_time + β·nll_mark + surv)")


if __name__ == "__main__":
    main()
