"""Neural network modules referenced in main.tex §2.

All modules are plain ``torch.nn.Module`` instances; they carry no state and
are called pointwise (no time-unrolling here, which is done in ``sde.py``).

Architectures are NOT specified by the paper; defaults are documented in
``OPEN_ISSUES.md``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int = 2,
         act: nn.Module = nn.Tanh) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class DriftNet(nn.Module):
    """``μ_θ(z, t)`` in eq. (2.5) of main.tex.

    Takes the latent state and the current (scalar) time and returns the drift
    vector in ``R^{d_z}``. The time is concatenated after a sinusoidal feature
    expansion so the network does not need to learn a scale for ``t``.
    """

    def __init__(self, d_z: int, hidden: int = 64, n_layers: int = 2,
                 n_time_feats: int = 4):
        super().__init__()
        self.d_z = d_z
        self.n_time_feats = n_time_feats
        self.net = _mlp(d_z + 2 * n_time_feats, d_z, hidden, n_layers)

    def _time_features(self, t: torch.Tensor) -> torch.Tensor:
        # t: (...,) scalar time. Emit sin/cos at geometric freqs.
        freqs = torch.pow(2.0, torch.arange(self.n_time_feats, device=t.device, dtype=t.dtype))
        angles = t.unsqueeze(-1) * freqs  # (..., n_time_feats)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: (..., d_z); t: (...,) or broadcastable scalar.
        if t.dim() == 0:
            t = t.expand(z.shape[:-1])
        feats = self._time_features(t)
        x = torch.cat([z, feats], dim=-1)
        return self.net(x)


class JumpNet(nn.Module):
    """``J_φ(z_-, x)`` in eq. (2.7): additive jump applied at event times.

    ``z(t_i^+) = z(t_i^-) + J_φ(z(t_i^-), x_i)``.
    """

    def __init__(self, d_z: int, d_x: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(d_z + d_x, d_z, hidden, n_layers)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, x], dim=-1))


class DualChannelIntensity(nn.Module):
    """``λ_g(t) = Softplus(a_tr·φ_tr(z) + a_vol·φ_vol(v) + b_λ)`` in eq. (2.10).

    The paper leaves ``φ_tr`` and ``φ_vol`` unspecified beyond "neural features".
    We use small tanh MLPs and then inner-product with trainable vectors
    ``a_tr, a_vol``. Softplus guarantees positivity.
    """

    def __init__(self, d_z: int, d_v: int, d_h: int = 32, n_layers: int = 2):
        super().__init__()
        self.phi_tr = _mlp(d_z, d_h, hidden=d_h, n_layers=n_layers)
        self.phi_vol = _mlp(d_v, d_h, hidden=d_h, n_layers=n_layers)
        self.a_tr = nn.Parameter(torch.zeros(d_h))
        self.a_vol = nn.Parameter(torch.zeros(d_h))
        # Small positive bias so the initial intensity is not vanishing.
        self.b_lam = nn.Parameter(torch.tensor(0.0))
        nn.init.normal_(self.a_tr, std=1.0 / math.sqrt(d_h))
        nn.init.normal_(self.a_vol, std=1.0 / math.sqrt(d_h))

    def forward(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # Returns: (...,) positive intensity λ_g.
        lam_tr = (self.phi_tr(z) * self.a_tr).sum(dim=-1)
        lam_vol = (self.phi_vol(v) * self.a_vol).sum(dim=-1)
        return F.softplus(lam_tr + lam_vol + self.b_lam)


class GMMMarkDecoder(nn.Module):
    """Volatility-modulated Gaussian-mixture decoder, eq. (2.10) of main.tex.

    ``p(x | z, v) = Σ π_k(z) · N(x; μ_k(z), Diag(σ_k²(z) · s(v)))``
    with ``s(v) = 1 + v`` elementwise (documented in OPEN_ISSUES.md).

    Returns per-event log-probabilities.
    """

    def __init__(self, d_z: int, d_v: int, d_x: int, K: int = 3,
                 hidden: int = 64, n_layers: int = 2,
                 min_log_var: float = -6.0, max_log_var: float = 4.0):
        super().__init__()
        self.K = K
        self.d_x = d_x
        # Shared trunk on z; three heads for (log π, μ, log σ²).
        self.trunk = _mlp(d_z, hidden, hidden, n_layers=n_layers)
        self.head_logits = nn.Linear(hidden, K)
        self.head_mu = nn.Linear(hidden, K * d_x)
        self.head_logvar = nn.Linear(hidden, K * d_x)
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

    @staticmethod
    def _s_of_v(v: torch.Tensor) -> torch.Tensor:
        # Positive, monotone, ≥ 1. v is already ≥ 0 in our CIR.
        return 1.0 + v

    def _params(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        log_pi = F.log_softmax(self.head_logits(h), dim=-1)                       # (..., K)
        mu = self.head_mu(h).view(*z.shape[:-1], self.K, self.d_x)                # (..., K, d_x)
        lv = self.head_logvar(h).view(*z.shape[:-1], self.K, self.d_x)
        lv = lv.clamp(self.min_log_var, self.max_log_var)
        return log_pi, mu, lv

    def log_prob(self, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Log-likelihood of ``x`` under the decoder conditioned on ``(z, v)``.

        Shapes
        ------
        x : (N, d_x)
        z : (N, d_z)
        v : (N, d_v) with d_v == d_x required (per-coordinate scaling); if
            d_v != d_x we fallback to scalar scaling by the first coordinate.
        """
        log_pi, mu, lv = self._params(z)                  # (..., K), (..., K, d_x), (..., K, d_x)
        s_v = self._s_of_v(v)                             # (..., d_v)
        if s_v.shape[-1] != self.d_x:
            s_v = s_v.mean(dim=-1, keepdim=True)          # broadcast a scalar
        log_var = lv + torch.log(s_v).unsqueeze(-2)       # (..., K, d_x)
        # Component-wise log N(x; μ_k, diag(exp(log_var_k))).
        x_ = x.unsqueeze(-2)                              # (..., 1, d_x)
        quad = -0.5 * ((x_ - mu) ** 2 / torch.exp(log_var)).sum(dim=-1)
        norm = -0.5 * (log_var.sum(dim=-1) + self.d_x * math.log(2 * math.pi))
        log_comp = quad + norm                            # (..., K)
        return torch.logsumexp(log_pi + log_comp, dim=-1)

    def sample(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sample one mark per event. Used only for simulation / debugging."""
        log_pi, mu, lv = self._params(z)
        s_v = self._s_of_v(v)
        if s_v.shape[-1] != self.d_x:
            s_v = s_v.mean(dim=-1, keepdim=True)
        log_var = lv + torch.log(s_v).unsqueeze(-2)
        cat = torch.distributions.Categorical(logits=log_pi)
        k = cat.sample()                                  # (...,)
        mu_k = mu.gather(-2, k[..., None, None].expand(*k.shape, 1, self.d_x)).squeeze(-2)
        lv_k = log_var.gather(-2, k[..., None, None].expand(*k.shape, 1, self.d_x)).squeeze(-2)
        eps = torch.randn_like(mu_k)
        return mu_k + eps * torch.exp(0.5 * lv_k)
