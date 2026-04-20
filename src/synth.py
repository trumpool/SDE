"""Synthetic data generator for small-batch correctness checks.

We simulate a ground-truth process that matches the paper's functional form —
latent jump-diffusion with CIR volatility, a dual-channel softplus intensity,
and volatility-sensitive Gaussian marks — but whose drift / jump / intensity /
mark networks are *hand-specified*. That way we know exactly what the model
*should* recover, at least qualitatively, and can build tests around the
generating process.

Event thinning uses a simple Bernoulli approximation with a fine time grid:
for each step of length ``dt_sim`` we draw one Bernoulli with
``p = 1 - exp(-λ · dt_sim)``. Error is O(dt_sim²).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Ground-truth parameters
# ---------------------------------------------------------------------------


@dataclass
class GroundTruthParams:
    """Hand-crafted ground-truth process parameters.

    Keep these small and simple so the structure of generated data is
    interpretable at a glance.
    """

    d_z: int = 2
    d_v: int = 2
    d_x: int = 3

    # Drift: μ(z, t) = -theta · z (mean reversion to 0, no time dependence).
    theta: float = 0.5

    # Jump: z <- z + γ · tanh(W_j x + b_j), bounded jump in each direction.
    gamma: float = 0.5
    # W_j, b_j are randomly drawn once per instance for reproducibility.

    # CIR volatility.
    kappa: float = 0.8
    v_bar: float = 1.0
    xi: float = 0.4
    rho: float = 0.0

    # Intensity: λ(t) = softplus(a_tr · z̄ + a_vol · v̄ + b).
    a_tr: float = 1.0
    a_vol: float = 0.8
    b_lam: float = -0.5

    # Mark: x ~ N(W_x z, (1 + mean(v)) · sigma_x²).
    sigma_x: float = 0.3

    # Random seed for the hand-picked linear weights.
    weight_seed: int = 0


@dataclass
class SyntheticSequence:
    t0: float
    T: float
    event_times: torch.Tensor             # (N,)
    event_marks: torch.Tensor             # (N, d_x)
    # Diagnostics kept on the fine grid so tests can verify the generator.
    t_grid: torch.Tensor                  # (M+1,)
    z_grid: torch.Tensor                  # (M+1, d_z)
    v_grid: torch.Tensor                  # (M+1, d_v)
    lam_grid: torch.Tensor                # (M+1,)


@dataclass
class SyntheticDataset:
    sequences: List[SyntheticSequence]
    params: GroundTruthParams
    W_j: torch.Tensor                     # (d_z, d_x) — jump weights
    b_j: torch.Tensor                     # (d_z,)
    W_x: torch.Tensor                     # (d_x, d_z) — mark mean weights


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def _sample_linear_weights(params: GroundTruthParams) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(params.weight_seed)
    W_j = 0.5 * torch.randn(params.d_z, params.d_x, generator=g)
    b_j = 0.1 * torch.randn(params.d_z, generator=g)
    W_x = 0.5 * torch.randn(params.d_x, params.d_z, generator=g)
    return W_j, b_j, W_x


def _drift(z: torch.Tensor, params: GroundTruthParams) -> torch.Tensor:
    return -params.theta * z


def _jump(z: torch.Tensor, x: torch.Tensor, W_j: torch.Tensor, b_j: torch.Tensor,
          params: GroundTruthParams) -> torch.Tensor:
    # γ · tanh(W_j x + b_j) — bounded, smooth.
    return params.gamma * torch.tanh(x @ W_j.T + b_j)


def _intensity(z: torch.Tensor, v: torch.Tensor, params: GroundTruthParams) -> torch.Tensor:
    z_bar = z.mean(dim=-1)
    v_bar_cur = v.mean(dim=-1)
    return F.softplus(params.a_tr * z_bar + params.a_vol * v_bar_cur + params.b_lam)


def _sample_mark(z: torch.Tensor, v: torch.Tensor, W_x: torch.Tensor,
                 params: GroundTruthParams, generator: Optional[torch.Generator] = None
                 ) -> torch.Tensor:
    mu = W_x @ z
    scale = params.sigma_x * math.sqrt(1.0 + float(v.mean().item()))
    return mu + scale * torch.randn(mu.shape, generator=generator)


def simulate_one_sequence(
    params: GroundTruthParams,
    W_j: torch.Tensor,
    b_j: torch.Tensor,
    W_x: torch.Tensor,
    t0: float,
    T: float,
    dt_sim: float = 0.01,
    z0: Optional[torch.Tensor] = None,
    v0: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    max_events: int = 10_000,
) -> SyntheticSequence:
    """Simulate one (t_grid, z_grid, v_grid, lam_grid, events) sample path.

    A Bernoulli thinning approximation is used: in each ``dt_sim`` step, an
    event is emitted with probability ``1 - exp(-λ · dt_sim)``. At most one
    event per step (fine enough dt_sim makes this reasonable).
    """
    if z0 is None:
        z0 = torch.zeros(params.d_z)
    if v0 is None:
        v0 = torch.full((params.d_v,), params.v_bar)

    n_steps = int(math.ceil((T - t0) / dt_sim))
    dt = torch.tensor((T - t0) / n_steps)
    rho = torch.tensor(params.rho)

    z = z0.clone()
    v = v0.clone()
    t = float(t0)
    lam = _intensity(z, v, params)

    t_hist = [t]
    z_hist = [z.clone()]
    v_hist = [v.clone()]
    lam_hist = [lam.clone()]
    events_t: List[float] = []
    events_x: List[torch.Tensor] = []

    sqrt_dt = torch.sqrt(dt)
    for _ in range(n_steps):
        eps1 = torch.randn(params.d_z, generator=generator)
        eps2 = torch.randn(params.d_z, generator=generator)
        dW_z = sqrt_dt * eps1
        dW_v = sqrt_dt * (rho * eps1 + torch.sqrt(1.0 - rho * rho) * eps2)

        v_floor = torch.clamp(v, min=0.0)
        sqrt_v = torch.sqrt(v_floor + 1e-12)

        z_new = z + _drift(z, params) * dt + sqrt_v * dW_z
        v_new = v + params.kappa * (params.v_bar - v) * dt + params.xi * sqrt_v * dW_v
        v_new = torch.clamp(v_new, min=0.0)

        # Thinning at step end:
        lam_new = _intensity(z_new, v_new, params)
        p_event = 1.0 - torch.exp(-lam_new * dt)
        u = torch.rand((), generator=generator)
        event_fired = bool((u < p_event).item())

        t_new = t + float(dt.item())

        if event_fired and len(events_t) < max_events:
            # Sample a mark at the left-limit state (consistent with paper §2.4).
            x_i = _sample_mark(z_new, v_new, W_x, params, generator=generator)
            events_t.append(t_new)
            events_x.append(x_i)
            # Apply jump.
            z_new = z_new + _jump(z_new, x_i, W_j, b_j, params)
            lam_new = _intensity(z_new, v_new, params)

        z, v, t = z_new, v_new, t_new
        t_hist.append(t)
        z_hist.append(z.clone())
        v_hist.append(v.clone())
        lam_hist.append(lam_new.clone())

    t_grid = torch.tensor(t_hist)
    z_grid = torch.stack(z_hist)
    v_grid = torch.stack(v_hist)
    lam_grid = torch.stack(lam_hist)

    event_times = torch.tensor(events_t) if events_t else torch.zeros(0)
    event_marks = torch.stack(events_x) if events_x else torch.zeros(0, params.d_x)

    # Guard: events must sit strictly inside (t0, T). Clip to safe interior.
    if event_times.numel() > 0:
        tiny = dt_sim * 0.5
        event_times = torch.clamp(event_times, min=t0 + tiny, max=T - tiny)

    return SyntheticSequence(
        t0=t0, T=T,
        event_times=event_times,
        event_marks=event_marks,
        t_grid=t_grid,
        z_grid=z_grid,
        v_grid=v_grid,
        lam_grid=lam_grid,
    )


def make_dataset(
    n_seq: int = 8,
    t0: float = 0.0,
    T: float = 5.0,
    dt_sim: float = 0.01,
    params: Optional[GroundTruthParams] = None,
    seed: int = 0,
) -> SyntheticDataset:
    params = params or GroundTruthParams()
    W_j, b_j, W_x = _sample_linear_weights(params)
    g = torch.Generator().manual_seed(seed)
    seqs = [
        simulate_one_sequence(
            params, W_j, b_j, W_x, t0=t0, T=T, dt_sim=dt_sim, generator=g
        )
        for _ in range(n_seq)
    ]
    return SyntheticDataset(sequences=seqs, params=params, W_j=W_j, b_j=b_j, W_x=W_x)
