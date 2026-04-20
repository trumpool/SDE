"""Correctness tests for the SDE integrator and grid."""
import math

import pytest
import torch

from src.sde import build_grid, sample_correlated_dW, simulate


def test_grid_contains_events():
    event_times = torch.tensor([1.0, 2.5, 3.7])
    t_grid, ei = build_grid(0.0, event_times, 5.0, dt=0.1)
    # events strictly interior
    assert t_grid[0].item() == 0.0
    assert t_grid[-1].item() == 5.0
    # event indices point at the right times
    for k, e in enumerate(event_times.tolist()):
        assert abs(t_grid[ei[k]].item() - e) < 1e-6


def test_grid_strictly_increasing():
    event_times = torch.tensor([0.3, 0.9, 1.0, 2.0])
    t_grid, _ = build_grid(0.0, event_times, 3.0, dt=0.25)
    diffs = t_grid[1:] - t_grid[:-1]
    assert (diffs > 0).all()


def test_correlated_brownian_correlation():
    torch.manual_seed(0)
    d = 4
    n = 200_000
    dt = torch.tensor(0.01)
    rho = torch.tensor(0.7)
    dW_z, dW_v = sample_correlated_dW(n, d, dt, rho=rho,
                                      device=torch.device("cpu"),
                                      dtype=torch.float64)
    # Per-coordinate: var = dt, cov = rho*dt.
    var_z = (dW_z ** 2).mean(dim=0).mean().item()
    var_v = (dW_v ** 2).mean(dim=0).mean().item()
    cov = (dW_z * dW_v).mean(dim=0).mean().item()
    assert abs(var_z - 0.01) < 5e-4
    assert abs(var_v - 0.01) < 5e-4
    assert abs(cov - 0.7 * 0.01) < 5e-4


def test_cir_mean_reverts_in_expectation():
    """Run many paths with no events; E[v(T)] should be close to v̄."""
    torch.manual_seed(1)
    d_z = 1
    d_v = 1

    def drift(z, t):
        return torch.zeros_like(z)

    def jump(z, x):
        return torch.zeros_like(z)

    class ConstantIntensity(torch.nn.Module):
        def forward(self, z, v):
            return torch.zeros(z.shape[:-1]).to(z)  # λ=0 → no survival contribution

    intensity = ConstantIntensity()

    kappa = torch.tensor([2.0])
    v_bar = torch.tensor([0.5])
    xi = torch.tensor([0.3])
    rho = torch.tensor(0.0)

    v_inits = [torch.tensor([0.05]), torch.tensor([2.0])]  # below and above v̄
    T = 5.0
    for v0 in v_inits:
        ends = []
        for _ in range(200):
            res = simulate(
                drift=drift, jump=jump, intensity=intensity,
                kappa=kappa, v_bar=v_bar, xi=xi, rho=rho,
                z0=torch.zeros(d_z), v0=v0,
                t0=0.0, T=T,
                event_times=torch.zeros(0),
                event_marks=torch.zeros(0, 1),
                dt=0.02,
            )
            ends.append(res.v_grid[-1].item())
        mean_end = sum(ends) / len(ends)
        # Theoretical mean of CIR at time T: v0 * exp(-κT) + v̄ * (1 - exp(-κT))
        expected = (float(v0.item()) * math.exp(-2.0 * T)
                    + 0.5 * (1 - math.exp(-2.0 * T)))
        assert abs(mean_end - expected) < 0.05, (mean_end, expected)


def test_v_nonnegative():
    torch.manual_seed(2)
    def drift(z, t): return torch.zeros_like(z)
    def jump(z, x): return torch.zeros_like(z)
    class ZeroIntensity(torch.nn.Module):
        def forward(self, z, v):
            return torch.zeros(z.shape[:-1]).to(z)
    intensity = ZeroIntensity()
    res = simulate(
        drift=drift, jump=jump, intensity=intensity,
        kappa=torch.tensor([0.1]), v_bar=torch.tensor([0.01]),
        xi=torch.tensor([2.0]),  # big vol-of-vol to stress Feller
        rho=torch.tensor(0.0),
        z0=torch.zeros(1), v0=torch.tensor([0.1]),
        t0=0.0, T=2.0,
        event_times=torch.zeros(0), event_marks=torch.zeros(0, 1),
        dt=0.01,
    )
    assert (res.v_grid >= 0).all()


def test_event_jump_is_applied():
    """After one event, post-jump z should equal pre-jump z + jump value."""
    def drift(z, t): return torch.zeros_like(z)

    JUMP_VAL = torch.tensor([1.5])

    def jump(z, x):
        return JUMP_VAL

    class ZeroIntensity(torch.nn.Module):
        def forward(self, z, v):
            return torch.zeros(z.shape[:-1]).to(z)

    intensity = ZeroIntensity()
    # v0 = v̄ = 0 → sqrt(v) = 0 → z has no diffusion either.
    res = simulate(
        drift=drift, jump=jump, intensity=intensity,
        kappa=torch.tensor([1.0]), v_bar=torch.tensor([0.0]),
        xi=torch.tensor([0.0]),
        rho=torch.tensor(0.0),
        z0=torch.zeros(1), v0=torch.tensor([0.0]),
        t0=0.0, T=2.0,
        event_times=torch.tensor([1.0]), event_marks=torch.zeros(1, 1),
        dt=0.25,
    )
    # Pre-event z should be ~0 (no drift, negligible diffusion from the
    # sqrt(v+1e-12) epsilon floor), post-event ≈ pre + 1.5.
    assert torch.allclose(res.z_pre_event[0], torch.tensor([0.0]), atol=1e-4)
    assert torch.allclose(res.z_grid[-1] - res.z_pre_event[0], JUMP_VAL, atol=1e-4)
