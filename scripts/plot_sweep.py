"""Plot val curves from ``sweep_results.json`` produced by ``sweep_beta.py``.

Outputs four panels to ``figures/sweep.png``:
  (1) total/ev vs step
  (2) nll_time/ev vs step
  (3) nll_mark/ev vs step
  (4) |a_vol|, ρ, κ̄ trajectories on a twin-axis plot

Usage:
    python scripts/plot_sweep.py
    python scripts/plot_sweep.py --in <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", type=str,
                        default="data/encoded/sweep_results.json")
    parser.add_argument("--out", type=str, default="figures/sweep.png")
    args = parser.parse_args()

    inp_path = args.inp if os.path.isabs(args.inp) else os.path.join(_ROOT, args.inp)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(inp_path) as fh:
        payload = json.load(fh)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for run in payload["results"]:
        name = run["config"]["name"]
        steps = [e["step"] for e in run["curves"]]
        total = [e["val"]["total_per_event"] for e in run["curves"]]
        nll_t = [e["val"]["nll_time_per_event"] for e in run["curves"]]
        nll_m = [e["val"]["nll_mark_per_event"] for e in run["curves"]]
        rho = [e["rho"] for e in run["curves"]]
        a_vol = [e["a_vol_abs_mean"] for e in run["curves"]]

        axes[0, 0].plot(steps, total, label=name, marker="o", markersize=3)
        axes[0, 1].plot(steps, nll_t, label=name, marker="o", markersize=3)
        axes[1, 0].plot(steps, nll_m, label=name, marker="o", markersize=3)
        axes[1, 1].plot(steps, a_vol, label=name, marker="o", markersize=3)

    axes[0, 0].set_title("val total/ev")
    axes[0, 0].set_xlabel("step"); axes[0, 0].set_ylabel("nll/event")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("val nll_time/ev")
    axes[0, 1].set_xlabel("step"); axes[0, 1].set_ylabel("nll_time/event")
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("val nll_mark/ev")
    axes[1, 0].set_xlabel("step"); axes[1, 0].set_ylabel("nll_mark/event")
    axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("|a_vol| (mean abs)")
    axes[1, 1].set_xlabel("step"); axes[1, 1].set_ylabel("|a_vol|")
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.3)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")


if __name__ == "__main__":
    main()
