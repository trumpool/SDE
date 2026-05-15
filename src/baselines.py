"""Closed-form / lightweight baselines for the marked point process.

All baselines expose the same surface:

    base = Baseline(...)
    base.fit(train_sequences)
    metrics = base.evaluate(test_sequences)   # returns NLLMetrics

where ``train_sequences`` and ``test_sequences`` are lists of objects with
the attributes ``event_times`` (1-D tensor of strictly-increasing event
times in [t0, T]), ``event_marks`` (N × d_x tensor), ``t0``, ``T``.

Two families are implemented:

* **HomogeneousPoisson** — constant ground intensity, MLE in closed form.
* **HawkesExponential** — ``λ(t) = μ + Σ α·exp(-β·(t - t_i))`` with three
  scalar parameters fit by Adam on the analytical log-likelihood.

Both pair with the **GMMMarkModel**, a stand-alone unconditional Gaussian
mixture fit to all training marks (decoupled from time / latent state).

These are deliberately the simplest sensible baselines: any reasonable
neural marked point process should comfortably beat both on real-world
event data. They are useful as a *floor* to confirm the main model is
actually learning something.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import GMMMarkDecoder  # we re-purpose the K-component GMM


# ---------------------------------------------------------------------------
# Metric container
# ---------------------------------------------------------------------------


@dataclass
class NLLMetrics:
    """Per-event averages of the three NLL components on a held-out set."""

    nll_time_per_event: float
    nll_mark_per_event: float
    total_per_event: float            # nll_time + β·nll_mark (β recorded for ref)
    survival_per_event: float
    n_events: int
    n_sequences: int
    beta: float = 1.0

    def as_row(self, name: str) -> dict:
        return {
            "model": name,
            "n_seq": self.n_sequences,
            "n_events": self.n_events,
            "nll_time/ev": self.nll_time_per_event,
            "nll_mark/ev": self.nll_mark_per_event,
            "surv/ev": self.survival_per_event,
            "total/ev": self.total_per_event,
            "β": self.beta,
        }


# ---------------------------------------------------------------------------
# Mark model — unconditional GMM
# ---------------------------------------------------------------------------


class GMMMarkModel(nn.Module):
    """Unconditional GMM over the d_x mark space.

    Fit by 100 Adam steps of mini-batch negative log-likelihood on all
    training marks pooled together. Used by both Poisson and Hawkes
    baselines so they share the same mark fit.
    """

    def __init__(self, d_x: int, K: int = 3,
                 min_log_var: float = -6.0, max_log_var: float = 4.0):
        super().__init__()
        self.d_x = d_x
        self.K = K
        self.log_pi = nn.Parameter(torch.zeros(K))
        self.mu = nn.Parameter(torch.randn(K, d_x) * 0.1)
        self.logvar = nn.Parameter(torch.zeros(K, d_x))
        self.min_log_var = float(min_log_var)
        self.max_log_var = float(max_log_var)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (N, d_x). Returns (N,) log-probabilities."""
        log_pi = F.log_softmax(self.log_pi, dim=-1)              # (K,)
        lv = self.logvar.clamp(self.min_log_var, self.max_log_var)  # (K, d_x)
        x_ = x.unsqueeze(-2)                                     # (N, 1, d_x)
        diff = x_ - self.mu                                      # (N, K, d_x)
        quad = -0.5 * (diff ** 2 / torch.exp(lv)).sum(dim=-1)
        norm = -0.5 * (lv.sum(dim=-1) + self.d_x * math.log(2 * math.pi))
        log_comp = quad + norm                                   # (N, K)
        return torch.logsumexp(log_pi + log_comp, dim=-1)

    def fit(self, marks: torch.Tensor, steps: int = 200, lr: float = 5e-2,
            verbose: bool = False) -> None:
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        for s in range(steps):
            opt.zero_grad()
            loss = -self.log_prob(marks).mean()
            loss.backward()
            opt.step()
            if verbose and (s % 50 == 0 or s == steps - 1):
                print(f"  [GMMMarkModel] step {s:3d}  NLL/ev = {loss.item():+.3f}")


# ---------------------------------------------------------------------------
# Baseline 1: Homogeneous Poisson
# ---------------------------------------------------------------------------


class HomogeneousPoisson:
    """λ(t) = μ. MLE: μ̂ = total_events / total_window."""

    def __init__(self, d_x: int, gmm_K: int = 3, mark_steps: int = 200):
        self.d_x = d_x
        self.gmm_K = gmm_K
        self.mark_steps = mark_steps
        self.mu: Optional[torch.Tensor] = None
        self.mark_model: Optional[GMMMarkModel] = None

    def fit(self, sequences) -> None:
        # MLE for intensity.
        n_events = sum(int(len(s.event_times)) for s in sequences)
        total_window = sum(float(s.T - s.t0) for s in sequences)
        self.mu = torch.tensor(max(n_events / max(total_window, 1e-9), 1e-9))
        # MLE for GMM marks (pool all training marks).
        marks = torch.cat([s.event_marks for s in sequences], dim=0)
        # If marks happen to be high-dim BERT cache, fit on them directly.
        self.mark_model = GMMMarkModel(d_x=marks.shape[-1], K=self.gmm_K)
        self.mark_model.fit(marks, steps=self.mark_steps)

    @torch.no_grad()
    def evaluate(self, sequences, beta: float = 1.0) -> NLLMetrics:
        assert self.mu is not None and self.mark_model is not None
        total_nll_time = 0.0
        total_nll_mark = 0.0
        total_survival = 0.0
        n_events = 0
        for s in sequences:
            nt = len(s.event_times)
            n_events += nt
            if nt > 0:
                # log λ at every event = log μ (constant).
                total_nll_time += -nt * float(torch.log(self.mu))
                total_nll_mark += -float(self.mark_model.log_prob(s.event_marks).sum())
            total_survival += float(self.mu) * float(s.T - s.t0)
        n_seqs = len(sequences)
        n_events = max(n_events, 1)
        return NLLMetrics(
            nll_time_per_event=total_nll_time / n_events,
            nll_mark_per_event=total_nll_mark / n_events,
            survival_per_event=total_survival / n_events,
            total_per_event=(total_nll_time + beta * total_nll_mark + total_survival) / n_events,
            n_events=n_events, n_sequences=n_seqs, beta=beta,
        )


# ---------------------------------------------------------------------------
# Baseline 2: Hawkes with exponential kernel
# ---------------------------------------------------------------------------


class HawkesExponential(nn.Module):
    """``λ(t) = μ + Σ_{t_i < t} α·exp(-β·(t - t_i))``, exponential kernel.

    Negative log-likelihood is exact in closed form for the exponential
    kernel because the survival integral telescopes. We fit ``(log μ,
    log α, log β)`` (so all stay positive) with Adam on this exact NLL.
    """

    def __init__(self, d_x: int, gmm_K: int = 3, mark_steps: int = 200):
        super().__init__()
        self.d_x = d_x
        self.gmm_K = gmm_K
        self.mark_steps = mark_steps
        # Reasonable inits: μ ~ 1 event/day, α ~ 0.3, β ~ 1/day.
        self.log_mu = nn.Parameter(torch.tensor(0.0))
        self.log_alpha = nn.Parameter(torch.tensor(-1.2))   # softplus^-1(0.3) ish
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        self.mark_model: Optional[GMMMarkModel] = None

    @property
    def mu(self) -> torch.Tensor: return F.softplus(self.log_mu) + 1e-6
    @property
    def alpha(self) -> torch.Tensor: return F.softplus(self.log_alpha) + 1e-6
    @property
    def beta(self) -> torch.Tensor: return F.softplus(self.log_beta) + 1e-6

    def _seq_loglik(self, t: torch.Tensor, t0: float, T: float) -> torch.Tensor:
        """Exact log-likelihood ``Σ log λ(t_i) − ∫_{t0}^{T} λ dt`` (no marks)."""
        mu, alpha, beta = self.mu, self.alpha, self.beta
        if t.numel() == 0:
            return -mu * (T - t0)

        # Recurrence for the kernel sum at each event:
        #   λ(t_i) = μ + α · κ_i,    κ_i = exp(-β·(t_i - t_{i-1})) · (κ_{i-1} + 1)
        # (κ_0 = 0 implicitly: at t_1, the sum is empty.)
        dts = torch.empty_like(t)
        dts[0] = 0.0
        dts[1:] = t[1:] - t[:-1]
        kappa = torch.zeros_like(t)
        # Iterative (loop fine — N is small here).
        for i in range(1, len(t)):
            kappa[i] = torch.exp(-beta * dts[i]) * (kappa[i - 1] + 1.0)
        lam = mu + alpha * kappa
        # Survival integral: μ·(T-t0)  +  α/β · Σ_i (1 − exp(-β·(T - t_i)))
        surv_base = mu * (T - t0)
        decays = 1.0 - torch.exp(-beta * (T - t))
        surv_excite = (alpha / beta) * decays.sum()
        return torch.log(lam + 1e-12).sum() - (surv_base + surv_excite)

    def _seq_nll(self, t, t0, T) -> torch.Tensor:
        return -self._seq_loglik(t, t0, T)

    def fit(self, sequences, steps: int = 200, lr: float = 5e-2,
            verbose: bool = False) -> None:
        opt = torch.optim.Adam(
            [self.log_mu, self.log_alpha, self.log_beta], lr=lr
        )
        for s in range(steps):
            opt.zero_grad()
            loss = sum(
                self._seq_nll(seq.event_times.detach(), seq.t0, seq.T)
                for seq in sequences
            )
            loss = loss / max(sum(len(seq.event_times) for seq in sequences), 1)
            loss.backward()
            opt.step()
            if verbose and (s % 50 == 0 or s == steps - 1):
                print(f"  [Hawkes]   step {s:3d}  NLL/ev = {loss.item():+.3f}  "
                      f"μ={self.mu.item():.3f}  α={self.alpha.item():.3f}  "
                      f"β={self.beta.item():.3f}")
        # Then fit marks (no timing dependence).
        marks = torch.cat([s.event_marks for s in sequences], dim=0)
        self.mark_model = GMMMarkModel(d_x=marks.shape[-1], K=self.gmm_K)
        self.mark_model.fit(marks, steps=self.mark_steps)

    @torch.no_grad()
    def evaluate(self, sequences, beta_loss: float = 1.0) -> NLLMetrics:
        total_nll_time = 0.0
        total_nll_mark = 0.0
        total_survival = 0.0
        n_events = 0
        for seq in sequences:
            t = seq.event_times.detach()
            nt = len(t)
            n_events += nt
            mu, alpha, beta = self.mu, self.alpha, self.beta
            # Reuse the recurrence to compute per-event log λ and survival.
            if nt > 0:
                kappa = torch.zeros_like(t)
                for i in range(1, nt):
                    dt = t[i] - t[i - 1]
                    kappa[i] = torch.exp(-beta * dt) * (kappa[i - 1] + 1.0)
                lam = mu + alpha * kappa
                total_nll_time += -float(torch.log(lam + 1e-12).sum())
                total_nll_mark += -float(self.mark_model.log_prob(seq.event_marks).sum())
            surv_base = float(mu * (seq.T - seq.t0))
            decays = 1.0 - torch.exp(-beta * (seq.T - t))
            surv_excite = float((alpha / beta) * decays.sum()) if nt > 0 else 0.0
            total_survival += surv_base + surv_excite
        n_events = max(n_events, 1)
        return NLLMetrics(
            nll_time_per_event=total_nll_time / n_events,
            nll_mark_per_event=total_nll_mark / n_events,
            survival_per_event=total_survival / n_events,
            total_per_event=(total_nll_time + beta_loss * total_nll_mark + total_survival) / n_events,
            n_events=n_events, n_sequences=len(sequences), beta=beta_loss,
        )
