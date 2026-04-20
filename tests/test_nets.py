"""Shape + basic-sanity tests for the neural components."""
import torch

from src.nets import DriftNet, JumpNet, DualChannelIntensity, GMMMarkDecoder


def test_drift_shape():
    net = DriftNet(d_z=4)
    z = torch.randn(7, 4)
    t = torch.rand(7)
    out = net(z, t)
    assert out.shape == (7, 4)


def test_drift_scalar_time():
    net = DriftNet(d_z=4)
    z = torch.randn(5, 4)
    out = net(z, torch.tensor(0.3))  # scalar time broadcast
    assert out.shape == (5, 4)


def test_jump_shape():
    net = JumpNet(d_z=3, d_x=5)
    z = torch.randn(6, 3)
    x = torch.randn(6, 5)
    assert net(z, x).shape == (6, 3)


def test_intensity_is_positive():
    net = DualChannelIntensity(d_z=3, d_v=3)
    z = torch.randn(128, 3)
    v = torch.rand(128, 3) * 2.0
    lam = net(z, v)
    assert lam.shape == (128,)
    assert (lam > 0).all()


def test_gmm_log_prob_finite():
    dec = GMMMarkDecoder(d_z=3, d_v=3, d_x=4, K=3)
    x = torch.randn(10, 4)
    z = torch.randn(10, 3)
    v = torch.rand(10, 3) * 1.5
    lp = dec.log_prob(x, z, v)
    assert lp.shape == (10,)
    assert torch.isfinite(lp).all()


def test_gmm_sample_shape():
    dec = GMMMarkDecoder(d_z=3, d_v=3, d_x=4, K=2)
    z = torch.randn(5, 3)
    v = torch.rand(5, 3)
    xs = dec.sample(z, v)
    assert xs.shape == (5, 4)
