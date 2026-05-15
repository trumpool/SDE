"""Sanity checks for the baselines:

- Homogeneous Poisson MLE matches the closed-form total_events / total_window.
- Hawkes log-likelihood matches a brute-force trapezoid evaluation of the
  survival integral on a known sequence.
- GMM mark model gives finite log-prob and is well-shaped.
- ``IntensityNet.freeze_volatility_channel`` truly zeroes ``a_vol`` and
  marks its parameter not-requires_grad.
"""
import math

import torch

from src.baselines import (
    GMMMarkModel, HawkesExponential, HomogeneousPoisson, NLLMetrics,
)
from src.nets import DualChannelIntensity


def _fake_seq(times, marks, t0=0.0, T=10.0):
    class S: ...
    s = S()
    s.event_times = torch.tensor(times, dtype=torch.float32)
    s.event_marks = torch.tensor(marks, dtype=torch.float32)
    s.t0 = t0
    s.T = T
    return s


def test_poisson_mle_closed_form():
    # 3 sequences: 4 events on [0,2], 2 events on [0,2], 0 on [0,1] → 6 / 5 = 1.2
    seqs = [
        _fake_seq([0.5, 1.0, 1.4, 1.9], [[0.0]] * 4, t0=0.0, T=2.0),
        _fake_seq([0.3, 1.7], [[0.0]] * 2, t0=0.0, T=2.0),
        _fake_seq([], [], t0=0.0, T=1.0),
    ]
    # fake empty event_marks tensor
    seqs[2].event_marks = torch.zeros(0, 1)
    p = HomogeneousPoisson(d_x=1, gmm_K=2, mark_steps=20)
    p.fit(seqs)
    assert abs(p.mu.item() - 6.0 / 5.0) < 1e-6


def test_hawkes_loglik_matches_trapezoid():
    """For a fixed (μ, α, β), check that the integrated intensity from the
    recurrence equals a fine-grained trapezoidal integration."""
    torch.manual_seed(0)
    t = torch.tensor([0.7, 1.4, 1.6, 2.5, 4.0], dtype=torch.float64)
    t0, T = 0.0, 5.0
    h = HawkesExponential(d_x=1)
    # Pin parameters.
    with torch.no_grad():
        h.log_mu.copy_(torch.tensor(0.4))      # μ ≈ softplus(0.4) ≈ 0.913
        h.log_alpha.copy_(torch.tensor(-0.3))  # α ≈ 0.554
        h.log_beta.copy_(torch.tensor(0.6))    # β ≈ 1.037
    h = h.double()
    t = t.double()

    ll = h._seq_loglik(t, t0, T).item()

    # Trapezoid reference for the survival term.
    mu = h.mu.item(); alpha = h.alpha.item(); beta = h.beta.item()
    grid = torch.linspace(t0, T, 200_001, dtype=torch.float64)
    lam = mu + alpha * sum(torch.where(grid > ti, torch.exp(-beta * (grid - ti)), torch.zeros_like(grid))
                           for ti in t.tolist())
    surv_ref = torch.trapz(lam, grid).item()
    # log λ at events.
    kappa = torch.zeros_like(t)
    for i in range(1, len(t)):
        kappa[i] = math.exp(-beta * (t[i] - t[i - 1])) * (kappa[i - 1] + 1.0)
    lam_at = mu + alpha * kappa
    sum_loglam = torch.log(lam_at).sum().item()
    ll_ref = sum_loglam - surv_ref

    assert abs(ll - ll_ref) < 1e-3, (ll, ll_ref)


def test_gmm_mark_model_log_prob_finite():
    torch.manual_seed(0)
    gmm = GMMMarkModel(d_x=4, K=3)
    x = torch.randn(20, 4)
    gmm.fit(x, steps=20)
    lp = gmm.log_prob(x)
    assert lp.shape == (20,)
    assert torch.isfinite(lp).all()


def test_intensity_freeze_volatility_channel():
    net = DualChannelIntensity(d_z=3, d_v=3, d_h=4)
    net.freeze_volatility_channel()
    # a_vol is zero and not learnable.
    assert torch.all(net.a_vol == 0)
    assert not net.a_vol.requires_grad
    # phi_vol params are frozen.
    for p in net.phi_vol.parameters():
        assert not p.requires_grad
    # Forward still works.
    z = torch.randn(5, 3)
    v = torch.rand(5, 3)
    lam = net(z, v)
    assert lam.shape == (5,)
    # Forward must NOT depend on v in any way once frozen.
    v2 = torch.rand(5, 3)
    lam2 = net(z, v2)
    assert torch.allclose(lam, lam2)
