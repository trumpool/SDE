"""Check the end-to-end pipeline when marks come from a BERT-dim cache.

Does not actually load BERT; synthesizes a fake cache that maps each row's
_id → a random 768-d tensor. Ensures the Linear(768→d_x) projection is wired
correctly and gradients flow all the way back to the BERT projector weights.
"""
import os

import pytest
import torch

from src.loss import compute_loss
from src.model import ModelConfig, NeuralSVMPP
from src.weibo_data import build_sequences, load_csv

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "2019-12.csv")
_HAS_CSV = os.path.exists(_CSV)


@pytest.mark.skipif(not _HAS_CSV, reason="raw CSV not downloaded")
def test_bert_projection_forward_and_backward():
    torch.manual_seed(0)
    df = load_csv(_CSV)

    # Build a fake 768-d cache keyed by _id. Just random floats.
    ids = df["_id"].astype(str).tolist()
    fake_embeds = torch.randn(len(ids), 768, dtype=torch.float32) * 0.1
    cache = ({s: i for i, s in enumerate(ids)}, fake_embeds)

    seqs = build_sequences(df, min_length=5, max_sequences=3, bert_cache=cache, seed=0)
    assert len(seqs) > 0
    assert seqs[0].event_marks.shape[-1] == 768

    cfg = ModelConfig(d_z=4, d_v=4, d_x=32, bert_dim=768)
    model = NeuralSVMPP(cfg)

    # Run one sequence end-to-end.
    seq = seqs[0]
    res = model.forward_sequence(
        seq.event_times, seq.event_marks, t0=seq.t0, T=seq.T, dt=0.1
    )
    # z_pre_event should be in the d_z space.
    assert res.z_pre_event.shape == (len(seq.event_times), cfg.d_z)
    lc = compute_loss(model, res, seq.event_marks)
    assert torch.isfinite(lc.total).all()

    # Backprop and check the bert projector gradient is non-zero.
    lc.total.backward()
    assert model.bert_proj is not None
    grad = model.bert_proj.weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0, "bert_proj did not receive gradient"


@pytest.mark.skipif(not _HAS_CSV, reason="raw CSV not downloaded")
def test_pass_through_when_already_dx():
    """If marks arrive already in d_x, projector should be a no-op."""
    torch.manual_seed(1)
    cfg = ModelConfig(d_z=4, d_v=4, d_x=32, bert_dim=768)
    model = NeuralSVMPP(cfg)
    dx_marks = torch.randn(5, 32)
    out = model.project_marks(dx_marks)
    assert torch.equal(out, dx_marks)
