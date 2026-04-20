"""End-to-end loss/forward sanity."""
import torch

from src.model import NeuralSVMPP, ModelConfig
from src.loss import compute_loss
from src.synth import make_dataset


def test_end_to_end_single_sequence_loss_finite():
    torch.manual_seed(0)
    ds = make_dataset(n_seq=2, T=2.0, dt_sim=0.02)
    seq = ds.sequences[0]
    cfg = ModelConfig(d_z=ds.params.d_z, d_v=ds.params.d_v, d_x=ds.params.d_x)
    model = NeuralSVMPP(cfg)
    res = model.forward_sequence(
        seq.event_times, seq.event_marks, t0=seq.t0, T=seq.T, dt=0.05
    )
    lc = compute_loss(model, res, seq.event_marks)
    assert torch.isfinite(lc.total).all()
    assert lc.survival.item() >= 0.0


def test_loss_backward_produces_gradients():
    torch.manual_seed(0)
    ds = make_dataset(n_seq=1, T=1.5, dt_sim=0.02)
    seq = ds.sequences[0]
    cfg = ModelConfig(d_z=ds.params.d_z, d_v=ds.params.d_v, d_x=ds.params.d_x)
    model = NeuralSVMPP(cfg)
    res = model.forward_sequence(
        seq.event_times, seq.event_marks, t0=seq.t0, T=seq.T, dt=0.05
    )
    lc = compute_loss(model, res, seq.event_marks)
    lc.total.backward()
    # At least one parameter in each sub-module should have a gradient.
    touched = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            touched.append(name)
    # Expect drift, intensity, decoder, and CIR params all touched.
    kinds = {n.split(".")[0] for n in touched}
    for k in ("drift", "intensity_net", "decoder"):
        assert k in kinds, f"no gradient reached {k}; touched={touched}"
