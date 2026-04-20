"""Forward integration of the latent jump-diffusion (main.tex §2.2).

We integrate Itô-form dynamics via Euler-Maruyama on an adaptive grid that
includes every event time as a breakpoint. CIR positivity is enforced with the
full-truncation scheme of Lord et al. (2010): ``sqrt(v⁺)`` in the diffusion,
but the drift uses the un-truncated ``v``. After each step we clamp ``v`` to
zero from below so that the next step's sqrt is well-defined. Rationale and
alternatives are documented in ``OPEN_ISSUES.md``.

The interpretation of ``dW_z, dW_v`` is Itô. The paper writes the *general*
formulation in Stratonovich (§2.1) because that form is convenient for the
adjoint derivation, but the *specific* dynamics in eq. (2.9) have no ``∘``
symbols; we take the Itô reading because it is the standard CIR convention.
For the z-equation the diffusion coefficient ``sqrt(v)`` does not depend on
``z``, so Itô = Stratonovich there anyway. For v the correction is a constant
drift shift (``+ξ²/4``) that the user can fold into ``v̄`` if needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import math
import torch


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------


def build_grid(t0: float, event_times: torch.Tensor, T: float,
               dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a 1D time grid on ``[t0, T]`` that includes every event time.

    Returns
    -------
    t_grid : (M+1,) float tensor, strictly increasing, ``t_grid[0] == t0`` and
             ``t_grid[-1] == T``.
    event_idx : (N,) long tensor; ``event_idx[i]`` is the index into ``t_grid``
                corresponding to ``event_times[i]``. Guarantees
                ``t_grid[event_idx[i]] == event_times[i]``.

    Event times must satisfy ``t0 < event_times < T`` (strict) and be sorted.
    """
    device = event_times.device
    dtype = event_times.dtype
    events = event_times.detach().cpu().tolist()
    if any(t <= t0 or t >= T for t in events):
        raise ValueError(f"event times must lie strictly in (t0={t0}, T={T}); got {events}")
    # Breakpoints partition [t0, T]; between each pair we do fixed-size Euler steps.
    breaks = [t0] + events + [T]
    grid: List[float] = [t0]
    event_idx: List[int] = []
    for i in range(len(breaks) - 1):
        a, b = breaks[i], breaks[i + 1]
        span = b - a
        if span <= 0.0:
            raise ValueError(f"non-positive sub-interval [{a}, {b}]")
        n_sub = max(1, int(math.ceil(span / dt)))
        step = span / n_sub
        for j in range(1, n_sub + 1):
            grid.append(a + j * step)
        # After extending, the last point of this segment is an event (unless
        # it's the final T). Record its index.
        if i < len(breaks) - 2:
            event_idx.append(len(grid) - 1)
    t_grid = torch.tensor(grid, device=device, dtype=dtype)
    ei = torch.tensor(event_idx, device=device, dtype=torch.long)
    return t_grid, ei


# ---------------------------------------------------------------------------
# Correlated Brownian increments
# ---------------------------------------------------------------------------


def sample_correlated_dW(n_steps: int, d: int, dt: torch.Tensor, rho: torch.Tensor,
                         device: torch.device, dtype: torch.dtype,
                         generator: Optional[torch.Generator] = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample increments ``(dW_z, dW_v)`` with component-wise correlation ``ρ``.

    ``dt`` may be either a scalar or a (n_steps,) tensor (heterogeneous steps).
    Returns two ``(n_steps, d)`` tensors.
    """
    if dt.dim() == 0:
        sqrt_dt = torch.sqrt(dt).expand(n_steps, 1)
    else:
        sqrt_dt = torch.sqrt(dt).unsqueeze(-1)                  # (n_steps, 1)
    eps1 = torch.randn(n_steps, d, device=device, dtype=dtype, generator=generator)
    eps2 = torch.randn(n_steps, d, device=device, dtype=dtype, generator=generator)
    dW_z = sqrt_dt * eps1
    dW_v = sqrt_dt * (rho * eps1 + torch.sqrt(1.0 - rho * rho) * eps2)
    return dW_z, dW_v


# ---------------------------------------------------------------------------
# Forward result container
# ---------------------------------------------------------------------------


@dataclass
class ForwardResult:
    """Everything the loss and downstream diagnostics need from one forward pass."""

    t_grid: torch.Tensor                # (M+1,)
    z_grid: torch.Tensor                # (M+1, d_z) — z at each grid point (pre-jump)
    v_grid: torch.Tensor                # (M+1, d_v)
    lam_grid: torch.Tensor              # (M+1,) — intensity at each grid point
    z_pre_event: torch.Tensor           # (N, d_z) — left-limits z(t_i^-)
    v_pre_event: torch.Tensor           # (N, d_v) — v(t_i^-)
    lam_at_event: torch.Tensor          # (N,) — λ_g(t_i | state at t_i^-)
    survival_integral: torch.Tensor     # scalar — ∫_{t0}^{T} λ_g dt (trapezoidal)


# ---------------------------------------------------------------------------
# Forward simulator
# ---------------------------------------------------------------------------


def simulate(
    *,
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    jump: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    intensity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    kappa: torch.Tensor,            # (d_v,) or scalar, must be > 0
    v_bar: torch.Tensor,            # (d_v,) or scalar, must be > 0
    xi: torch.Tensor,               # (d_v,) or scalar, must be > 0
    rho: torch.Tensor,              # scalar in [-1, 1]
    z0: torch.Tensor,               # (d_z,)
    v0: torch.Tensor,               # (d_v,)
    t0: float,
    T: float,
    event_times: torch.Tensor,      # (N,)
    event_marks: torch.Tensor,      # (N, d_x)
    dt: float = 0.05,
    generator: Optional[torch.Generator] = None,
) -> ForwardResult:
    """Euler-Maruyama on one sequence with event-driven jumps on ``z``.

    See the module docstring for the Itô/Stratonovich discussion.
    """
    if z0.dim() != 1 or v0.dim() != 1:
        raise ValueError("z0 and v0 must be 1D (single sequence).")

    device = z0.device
    dtype = z0.dtype
    d_z = z0.shape[0]
    d_v = v0.shape[0]
    N = event_times.shape[0]

    # --- grid ---
    t_grid, event_idx = build_grid(t0, event_times.to(device), T, dt)
    M = t_grid.shape[0] - 1                                  # number of Euler steps
    # Step sizes: (M,)
    step_sizes = t_grid[1:] - t_grid[:-1]

    # --- pre-sample Brownian increments ---
    dW_z, dW_v = sample_correlated_dW(
        M, max(d_z, d_v), step_sizes, rho=rho, device=device, dtype=dtype,
        generator=generator,
    )
    # If d_z != d_v we take the first d_z / d_v components respectively; most
    # commonly d_z == d_v.
    dW_z = dW_z[:, :d_z]
    dW_v = dW_v[:, :d_v]

    # --- set up storage ---
    z_hist = [z0]
    v_hist = [v0]
    lam_hist = [intensity(z0, v0)]
    z_pre: List[torch.Tensor] = []
    v_pre: List[torch.Tensor] = []
    lam_event: List[torch.Tensor] = []

    # index into next event to be processed
    event_set = set(int(i) for i in event_idx.tolist())
    event_mark_by_idx = {int(event_idx[i]): event_marks[i] for i in range(N)}

    z = z0
    v = v0
    survival = torch.zeros((), device=device, dtype=dtype)
    lam_cur = lam_hist[0]

    for k in range(M):
        dt_k = step_sizes[k]
        t_k = t_grid[k]
        v_floor = torch.clamp(v, min=0.0)
        sqrt_v = torch.sqrt(v_floor + 1e-12)

        # z step (Itô Euler-Maruyama)
        z_next = z + drift(z, t_k) * dt_k + sqrt_v * dW_z[k]
        # v step (CIR with full truncation)
        v_next = v + kappa * (v_bar - v) * dt_k + xi * sqrt_v * dW_v[k]
        v_next = torch.clamp(v_next, min=0.0)

        lam_next_pre = intensity(z_next, v_next)
        # Trapezoidal survival contribution BEFORE applying any jump at τ_{k+1}.
        survival = survival + 0.5 * (lam_cur + lam_next_pre) * dt_k

        # Record grid state (this is the left-limit if τ_{k+1} is an event).
        z_hist.append(z_next)
        v_hist.append(v_next)

        if (k + 1) in event_set:
            x_i = event_mark_by_idx[k + 1]
            z_pre.append(z_next)
            v_pre.append(v_next)
            lam_event.append(lam_next_pre)
            # Apply jump on z; v is continuous across events (eq. 2.7).
            z_next = z_next + jump(z_next, x_i)
            # Update lam AFTER the jump for the next step's trapezoidal.
            lam_post = intensity(z_next, v_next)
            # Overwrite the grid snapshot with the post-jump z so diagnostics
            # reflect the applied event. (The pre-jump is stored in z_pre.)
            z_hist[-1] = z_next
            lam_hist.append(lam_post)
            lam_cur = lam_post
        else:
            lam_hist.append(lam_next_pre)
            lam_cur = lam_next_pre

        z, v = z_next, v_next

    return ForwardResult(
        t_grid=t_grid,
        z_grid=torch.stack(z_hist),
        v_grid=torch.stack(v_hist),
        lam_grid=torch.stack(lam_hist),
        z_pre_event=torch.stack(z_pre) if N > 0 else torch.zeros((0, d_z), device=device, dtype=dtype),
        v_pre_event=torch.stack(v_pre) if N > 0 else torch.zeros((0, d_v), device=device, dtype=dtype),
        lam_at_event=torch.stack(lam_event) if N > 0 else torch.zeros((0,), device=device, dtype=dtype),
        survival_integral=survival,
    )
