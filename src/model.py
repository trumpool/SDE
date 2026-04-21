"""Top-level ``NeuralSVMPP`` module.

Owns all trainable networks plus the CIR parameters ``(κ, v̄, ξ, ρ)`` and
exposes a ``forward`` that runs the SDE simulator on one event sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import DriftNet, JumpNet, DualChannelIntensity, GMMMarkDecoder
from .sde import ForwardResult, simulate


@dataclass
class ModelConfig:
    d_z: int = 4
    d_v: int = 4
    d_x: int = 4
    # If set, the model owns a Linear(bert_dim → d_x) projector. Event marks
    # passed in with shape (..., bert_dim) are auto-projected down to d_x
    # before entering the SDE. This matches paper §4: x_i = W_p·h_i^[CLS]+b_p.
    bert_dim: Optional[int] = None
    drift_hidden: int = 64
    jump_hidden: int = 64
    intensity_hidden: int = 32
    gmm_hidden: int = 64
    gmm_K: int = 3
    n_time_feats: int = 4
    init_log_kappa: float = 0.0        # softplus(0) ≈ 0.693
    init_log_v_bar: float = 0.0
    init_log_xi: float = -1.0          # smaller xi for stability at init
    init_rho_raw: float = 0.0          # tanh(0) = 0


class NeuralSVMPP(nn.Module):
    """Neural marked point process with latent stochastic volatility.

    ``Ψ = {θ_μ, φ, η, κ, v̄, ξ, ρ}`` per eq. (2.3) of main.tex.

    All CIR parameters are stored as *raw* unconstrained tensors and mapped
    through ``softplus``/``tanh`` so we can use unconstrained optimizers.
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg

        self.drift = DriftNet(c.d_z, hidden=c.drift_hidden, n_time_feats=c.n_time_feats)
        self.jump = JumpNet(c.d_z, c.d_x, hidden=c.jump_hidden)
        self.intensity_net = DualChannelIntensity(c.d_z, c.d_v, d_h=c.intensity_hidden)
        self.decoder = GMMMarkDecoder(c.d_z, c.d_v, c.d_x, K=c.gmm_K, hidden=c.gmm_hidden)
        # Optional BERT [CLS] → d_x projector (paper §4).
        self.bert_proj = (
            nn.Linear(c.bert_dim, c.d_x) if c.bert_dim is not None else None
        )

        # CIR params — per-coordinate (broadcast from scalar init).
        self._raw_log_kappa = nn.Parameter(torch.full((c.d_v,), c.init_log_kappa))
        self._raw_log_v_bar = nn.Parameter(torch.full((c.d_v,), c.init_log_v_bar))
        self._raw_log_xi = nn.Parameter(torch.full((c.d_v,), c.init_log_xi))
        self._raw_rho = nn.Parameter(torch.tensor(c.init_rho_raw))

        # Initial state — learnable, initialized at zero / v̄.
        self.z0 = nn.Parameter(torch.zeros(c.d_z))
        # Use a buffer to match v̄ at init; we treat v0 as non-learnable initially.
        self.register_buffer("_v0_default", torch.ones(c.d_v))

    # ------- constrained params -------
    @property
    def kappa(self) -> torch.Tensor:
        return F.softplus(self._raw_log_kappa) + 1e-4

    @property
    def v_bar(self) -> torch.Tensor:
        return F.softplus(self._raw_log_v_bar) + 1e-4

    @property
    def xi(self) -> torch.Tensor:
        return F.softplus(self._raw_log_xi) + 1e-4

    @property
    def rho(self) -> torch.Tensor:
        return torch.tanh(self._raw_rho)

    @property
    def v0(self) -> torch.Tensor:
        # Default initial volatility = current mean-revert level v̄.
        return self.v_bar.detach()

    # ------- forward -------
    def project_marks(self, event_marks: torch.Tensor) -> torch.Tensor:
        """Apply ``bert_proj`` if configured and the input is BERT-dim.

        - If ``bert_proj`` is ``None`` → pass-through.
        - If ``event_marks`` already has the ``d_x`` trailing dim → pass-through
          (lets callers feed raw d_x marks even in BERT-configured models).
        - Otherwise project ``(..., bert_dim) → (..., d_x)`` through the
          learnable linear layer.
        """
        if self.bert_proj is None:
            return event_marks
        if event_marks.shape[-1] == self.cfg.d_x:
            return event_marks
        # Cast to the projector's dtype in case the cache is fp16.
        target_dtype = self.bert_proj.weight.dtype
        if event_marks.dtype != target_dtype:
            event_marks = event_marks.to(target_dtype)
        return self.bert_proj(event_marks)

    def forward_sequence(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        t0: float,
        T: float,
        dt: float = 0.05,
        generator: Optional[torch.Generator] = None,
    ) -> ForwardResult:
        event_marks = self.project_marks(event_marks)
        return simulate(
            drift=self.drift,
            jump=self.jump,
            intensity=self.intensity_net,
            kappa=self.kappa,
            v_bar=self.v_bar,
            xi=self.xi,
            rho=self.rho,
            z0=self.z0,
            v0=self.v0,
            t0=t0,
            T=T,
            event_times=event_times,
            event_marks=event_marks,
            dt=dt,
            generator=generator,
        )
