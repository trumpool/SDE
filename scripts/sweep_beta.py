"""β sweep + long-training run for the SVMPP on Weibo 2019-12.

Trains the same model multiple times under different ``β`` (the mark-loss
weight) and the no-vol ablation. All configs share

* the same Weibo data load
* the same BERT cache
* the same 80/20 train/test split (seed-fixed)
* the same model initialization seed

so the only varying factors are (β, freeze_vol, n_steps). For each config
we log per-25-step val NLL components and final test metrics, dump them to
``data/encoded/sweep_results.json``, and print a final summary table.

Default plan (~2.5 h on CPU):
    M β=0.01    × 300 steps
    M β=0.1    × 500 steps   ← long-training poster child
    M β=0.5    × 300 steps
    M β=1.0    × 300 steps
    B3 β=0.1   × 300 steps   ← no-vol ablation
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.baselines import NLLMetrics
from src.loss import compute_loss
from src.model import ModelConfig, NeuralSVMPP
from src.utils import set_seed
from src.weibo_data import MARK_DIM, sequences_from_path, train_test_split


def make_default_configs():
    return [
        {"name": "M_beta_0p01",  "beta": 0.01, "freeze_vol": False, "steps": 300},
        {"name": "M_beta_0p1",   "beta": 0.10, "freeze_vol": False, "steps": 500},
        {"name": "M_beta_0p5",   "beta": 0.50, "freeze_vol": False, "steps": 300},
        {"name": "M_beta_1p0",   "beta": 1.00, "freeze_vol": False, "steps": 300},
        {"name": "B3_no_vol",    "beta": 0.10, "freeze_vol": True,  "steps": 300},
    ]


@torch.no_grad()
def evaluate(
    model: NeuralSVMPP, seqs, *, beta: float, dt_train: float,
) -> NLLMetrics:
    total_nll_time = total_nll_mark = total_surv = 0.0
    n_events = 0
    for seq in seqs:
        res = model.forward_sequence(
            seq.event_times, seq.event_marks, t0=seq.t0, T=seq.T, dt=dt_train,
        )
        lc = compute_loss(model, res, seq.event_marks, beta=beta)
        total_nll_time += float(lc.nll_time)
        total_nll_mark += float(lc.nll_mark)
        total_surv += float(lc.survival)
        n_events += int(lc.n_events)
    n_events = max(n_events, 1)
    return NLLMetrics(
        nll_time_per_event=total_nll_time / n_events,
        nll_mark_per_event=total_nll_mark / n_events,
        survival_per_event=total_surv / n_events,
        total_per_event=(total_nll_time + beta * total_nll_mark + total_surv) / n_events,
        n_events=n_events, n_sequences=len(seqs), beta=beta,
    )


def train_one(
    cfg_run: dict, train_seqs, test_seqs, *, cfg_model: ModelConfig,
    lr: float, dt_train: float, log_every: int, device: torch.device,
    init_seed: int,
) -> dict:
    set_seed(init_seed)
    model = NeuralSVMPP(cfg_model).to(device)
    if cfg_run["freeze_vol"]:
        model.intensity_net.freeze_volatility_channel()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    beta = cfg_run["beta"]
    steps = cfg_run["steps"]
    curves: list[dict] = []
    t_start = time.time()

    for step in range(steps + 1):
        # Periodic val eval (also at step 0 and the final step).
        if step % log_every == 0 or step == steps:
            model.eval()
            val = evaluate(model, test_seqs, beta=beta, dt_train=dt_train)
            train_val = evaluate(model, train_seqs, beta=beta, dt_train=dt_train)
            model.train()
            elapsed = time.time() - t_start
            with torch.no_grad():
                entry = {
                    "step": step,
                    "elapsed_sec": elapsed,
                    "val": dataclasses.asdict(val),
                    "train": dataclasses.asdict(train_val),
                    "kappa_mean": float(model.kappa.mean()),
                    "v_bar_mean": float(model.v_bar.mean()),
                    "xi_mean": float(model.xi.mean()),
                    "rho": float(model.rho),
                    "a_vol_abs_mean": float(model.intensity_net.a_vol.abs().mean()),
                }
            curves.append(entry)
            print(
                f"  [{cfg_run['name']:>12s} step {step:4d}/{steps}]  "
                f"val total/ev={val.total_per_event:+.3f}  "
                f"(nll_t={val.nll_time_per_event:+.3f} | "
                f"nll_m={val.nll_mark_per_event:+.3f} | "
                f"surv={val.survival_per_event:+.3f})  "
                f"κ̄={entry['kappa_mean']:.3f} ρ={entry['rho']:+.3f} "
                f"|a_vol|={entry['a_vol_abs_mean']:.3f}  "
                f"[{elapsed/60:.1f} min]",
                flush=True,
            )

        if step == steps:
            break

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

    final = curves[-1]
    return {
        "config": cfg_run,
        "curves": curves,
        "final": final,
        "duration_sec": time.time() - t_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/raw/2019-12.csv")
    parser.add_argument("--bert-cache", type=str, default="data/encoded/2019-12_cls.pt")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--d-z", type=int, default=8)
    parser.add_argument("--d-x", type=int, default=32)
    parser.add_argument("--dt-train", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str,
                        default="data/encoded/sweep_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    csv_path = os.path.join(_ROOT, args.csv) if not os.path.isabs(args.csv) else args.csv
    bert_path = os.path.join(_ROOT, args.bert_cache) if not os.path.isabs(args.bert_cache) else args.bert_cache
    out_path = os.path.join(_ROOT, args.out) if not os.path.isabs(args.out) else args.out

    print(f"[setup] device={device}  csv={csv_path}  bert={bert_path}")

    sequences = sequences_from_path(
        csv_path, min_length=args.min_length, bert_cache_path=bert_path,
        seed=args.data_seed,
    )
    train_seqs, test_seqs = train_test_split(
        sequences, test_frac=args.test_frac, seed=args.data_seed,
    )
    for s in train_seqs + test_seqs:
        s.event_times = s.event_times.to(device)
        s.event_marks = s.event_marks.to(device)

    n_train_ev = sum(len(s.event_times) for s in train_seqs)
    n_test_ev = sum(len(s.event_times) for s in test_seqs)
    print(f"[split] train={len(train_seqs)} seqs / {n_train_ev} events  |  "
          f"test={len(test_seqs)} seqs / {n_test_ev} events")

    mark_in_dim = train_seqs[0].event_marks.shape[-1]
    cfg_model = ModelConfig(
        d_z=args.d_z, d_v=args.d_z, d_x=args.d_x, bert_dim=mark_in_dim,
    )

    configs = make_default_configs()
    total_planned = sum(c["steps"] for c in configs)
    print(f"[plan]  {len(configs)} configs, {total_planned} total training steps")
    print(f"[plan]  rough ETA: {total_planned * 6 / 60:.0f} min on CPU")

    results: list[dict] = []
    grand_t0 = time.time()
    for i, cfg_run in enumerate(configs):
        print(f"\n=== [{i+1}/{len(configs)}] {cfg_run['name']} "
              f"(β={cfg_run['beta']}, freeze_vol={cfg_run['freeze_vol']}, "
              f"steps={cfg_run['steps']}) ===", flush=True)
        r = train_one(
            cfg_run, train_seqs, test_seqs, cfg_model=cfg_model,
            lr=args.lr, dt_train=args.dt_train, log_every=args.log_every,
            device=device, init_seed=args.init_seed,
        )
        results.append(r)
        # Incremental save in case the long run is interrupted.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump({"configs": configs, "results": results,
                       "split_train_events": n_train_ev,
                       "split_test_events": n_test_ev}, fh, indent=2)
        print(f"  [{cfg_run['name']}] saved partial results to {out_path}",
              flush=True)

    print(f"\n[done] all configs in {(time.time()-grand_t0)/60:.1f} min")

    # ---- final summary table ----
    print("\n============== FINAL TEST METRICS (per-event NLL) ==============")
    cols = ["name", "β", "steps", "nll_time/ev", "nll_mark/ev", "surv/ev", "total/ev"]
    widths = {c: max(len(c), 12) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in results:
        f = r["final"]["val"]
        row = {
            "name": r["config"]["name"],
            "β": str(r["config"]["beta"]),
            "steps": str(r["config"]["steps"]),
            "nll_time/ev": f"{f['nll_time_per_event']:+.3f}",
            "nll_mark/ev": f"{f['nll_mark_per_event']:+.3f}",
            "surv/ev": f"{f['survival_per_event']:+.3f}",
            "total/ev": f"{f['total_per_event']:+.3f}",
        }
        print(" | ".join(row[c].ljust(widths[c]) for c in cols))


if __name__ == "__main__":
    main()
