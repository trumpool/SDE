"""MPP negative log-likelihood, eq. (2.11) of main.tex.

    J(Ψ) = − Σ_i log λ_g(t_i | z(t_i^-), v(t_i^-))
           − β · Σ_i log p_η(x_i | z(t_i^-), v(t_i^-))
           + ∫_{t_0}^{T} λ_g(t | z(t), v(t)) dt
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .model import NeuralSVMPP
from .sde import ForwardResult


@dataclass
class LossComponents:
    nll_time: torch.Tensor            # − Σ log λ_g(t_i)
    nll_mark: torch.Tensor            # − Σ log p(x_i | z_i^-, v_i^-)
    survival: torch.Tensor            # + ∫ λ_g dt
    total: torch.Tensor               # total objective (with β applied to nll_mark)
    n_events: int


def compute_loss(
    model: NeuralSVMPP,
    forward: ForwardResult,
    event_marks: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-12,
) -> LossComponents:
    """Assemble the scalar training objective from a finished forward pass.

    Parameters
    ----------
    model : the owning module (needed for the mark decoder).
    forward : output of ``simulate`` (or ``NeuralSVMPP.forward_sequence``).
    event_marks : (N, d_x) tensor matching the events passed to the simulator.
    beta : weight on the mark log-prob term (paper's ``β``; default 1).
    eps : floor for ``log λ`` to avoid ``log(0)``.
    """
    lam = forward.lam_at_event                      # (N,)
    nll_time = -torch.log(lam + eps).sum() if lam.numel() > 0 else torch.zeros(
        (), device=forward.survival_integral.device, dtype=forward.survival_integral.dtype
    )

    if event_marks.numel() > 0:
        log_p = model.decoder.log_prob(
            event_marks, forward.z_pre_event, forward.v_pre_event
        )                                           # (N,)
        nll_mark = -log_p.sum()
    else:
        nll_mark = torch.zeros_like(nll_time)

    survival = forward.survival_integral

    total = nll_time + beta * nll_mark + survival
    return LossComponents(
        nll_time=nll_time,
        nll_mark=nll_mark,
        survival=survival,
        total=total,
        n_events=int(event_marks.shape[0]) if event_marks.dim() > 0 else 0,
    )
