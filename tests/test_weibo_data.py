"""Sanity checks for the Weibo CSV → sequence pipeline."""
import os

import pandas as pd
import pytest
import torch

from src.weibo_data import (
    MARK_DIM, build_sequences, load_csv, sequences_from_path, summarize,
)

_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "2019-12.csv")
_HAS_CSV = os.path.exists(_CSV)


@pytest.mark.skipif(not _HAS_CSV, reason="raw CSV not downloaded")
def test_load_csv_basic():
    df = load_csv(_CSV)
    for col in ("_id", "user_id", "created_at", "content", "origin_weibo"):
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert len(df) > 40_000


@pytest.mark.skipif(not _HAS_CSV, reason="raw CSV not downloaded")
def test_build_sequences_min_length():
    df = load_csv(_CSV)
    seqs = build_sequences(df, min_length=5, max_sequences=5, seed=0)
    assert 0 < len(seqs) <= 5
    for s in seqs:
        assert s.event_times.dtype == torch.float32
        assert s.event_marks.shape == (len(s.event_times), MARK_DIM)
        # Strictly interior to (0, T).
        assert 0.0 < s.event_times.min().item()
        assert s.event_times.max().item() < s.T
        # Strictly increasing after jitter.
        diffs = s.event_times[1:] - s.event_times[:-1]
        assert (diffs > 0).all()


@pytest.mark.skipif(not _HAS_CSV, reason="raw CSV not downloaded")
def test_summarize_runs():
    seqs = sequences_from_path(_CSV, min_length=5, max_sequences=5)
    out = summarize(seqs)
    assert "sequences" in out and "events" in out
