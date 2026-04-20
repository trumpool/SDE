"""Autograd vs finite-difference gradient check on a small scalar parameter.

This is our single strongest correctness check: it verifies that the entire
forward+backward chain (SDE integrator, intensity, GMM decoder, survival
integral) is consistent with autograd. We freeze the Brownian path via a
seeded ``torch.Generator`` so the same noise realization is used for both
forward evaluations.
"""
import torch

from src.model import NeuralSVMPP, ModelConfig
from src.loss import compute_loss
from src.synth import make_dataset


def _loss_of_b(model: NeuralSVMPP, seq, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    res = model.forward_sequence(
        seq.event_times, seq.event_marks, t0=seq.t0, T=seq.T, dt=0.05,
        generator=gen,
    )
    return compute_loss(model, res, seq.event_marks).total


def test_finite_difference_vs_autograd_on_b_lambda():
    torch.manual_seed(0)
    ds = make_dataset(n_seq=1, T=1.0, dt_sim=0.02)
    seq = ds.sequences[0]
    cfg = ModelConfig(d_z=ds.params.d_z, d_v=ds.params.d_v, d_x=ds.params.d_x)
    # Use float64 for a tight tolerance.
    model = NeuralSVMPP(cfg).to(dtype=torch.float64)
    seq_d = seq
    # Move sequence to float64 as well.
    seq_d.event_times = seq.event_times.to(torch.float64)
    seq_d.event_marks = seq.event_marks.to(torch.float64)
    # Keep a seeded generator so the Brownian noise is reproducible.
    SEED = 123

    # Autograd gradient w.r.t. b_λ (scalar).
    loss = _loss_of_b(model, seq_d, seed=SEED)
    grads = torch.autograd.grad(loss, [model.intensity_net.b_lam], retain_graph=False)
    g_auto = grads[0].item()

    # Finite difference.
    eps = 1e-5
    with torch.no_grad():
        model.intensity_net.b_lam.add_(eps)
        loss_plus = _loss_of_b(model, seq_d, seed=SEED).item()
        model.intensity_net.b_lam.sub_(2 * eps)
        loss_minus = _loss_of_b(model, seq_d, seed=SEED).item()
        model.intensity_net.b_lam.add_(eps)  # restore
    g_fd = (loss_plus - loss_minus) / (2 * eps)

    rel = abs(g_auto - g_fd) / max(1.0, abs(g_fd))
    assert rel < 1e-3, f"autograd={g_auto:.6f}, fd={g_fd:.6f}, rel={rel:.2e}"


def test_finite_difference_vs_autograd_on_log_kappa():
    torch.manual_seed(1)
    ds = make_dataset(n_seq=1, T=1.0, dt_sim=0.02)
    seq = ds.sequences[0]
    cfg = ModelConfig(d_z=ds.params.d_z, d_v=ds.params.d_v, d_x=ds.params.d_x)
    model = NeuralSVMPP(cfg).to(dtype=torch.float64)
    seq.event_times = seq.event_times.to(torch.float64)
    seq.event_marks = seq.event_marks.to(torch.float64)
    SEED = 321

    p = model._raw_log_kappa
    loss = _loss_of_b(model, seq, seed=SEED)
    grads = torch.autograd.grad(loss, [p])
    g_auto = grads[0].clone()

    g_fd = torch.zeros_like(p)
    eps = 1e-5
    with torch.no_grad():
        for i in range(p.numel()):
            orig = p.data[i].item()
            p.data[i] = orig + eps
            loss_plus = _loss_of_b(model, seq, seed=SEED).item()
            p.data[i] = orig - eps
            loss_minus = _loss_of_b(model, seq, seed=SEED).item()
            p.data[i] = orig
            g_fd[i] = (loss_plus - loss_minus) / (2 * eps)

    num = (g_auto - g_fd).abs().max().item()
    denom = max(1.0, g_fd.abs().max().item())
    rel = num / denom
    assert rel < 5e-3, f"autograd={g_auto.tolist()}, fd={g_fd.tolist()}, rel={rel:.2e}"
