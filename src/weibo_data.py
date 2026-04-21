"""Weibo-COV 2.0 CSV → MPP event sequences.

Schema (per the dataset readme, verified against ``data/raw/2019-12.csv``):

    _id, user_id, crawl_time, created_at, like_num, repost_num,
    comment_num, content, origin_weibo, geo_info

``created_at`` turns out to carry SECOND precision (``YYYY-MM-DD HH:MM:SS``)
in practice, despite the project README saying minute precision. We still
jitter by ``U(-0.5s, +0.5s)`` after parsing to break any residual ties.

Design decisions (discussed with the user on 2026-04-20):

- Sequences are per ``user_id``; filter users with fewer than ``min_length``
  events.
- Time axis is **days since the user's first event** (after jitter). This
  keeps ``dt`` in the SDE on a natural scale.
- Repost handling: keep the event, but emit an ``is_repost`` flag as part of
  the mark.
- Marks are a small, fixed 8-dim feature vector (no BERT yet). Phase B will
  swap in RoBERTa-wwm-ext embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import math
import re

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Mark featurizer (placeholder until BERT)
# ---------------------------------------------------------------------------


MARK_FIELD_NAMES = [
    "content_len_log",
    "is_repost",
    "like_log1p",
    "repost_log1p",
    "comment_log1p",
    "hashtag_log1p",
    "url_log1p",
    "bias",  # constant 1.0 so downstream linear layers have a reachable bias
]
MARK_DIM = len(MARK_FIELD_NAMES)

_URL_RE = re.compile(r"https?://\S+")
_HASHTAG_RE = re.compile(r"#[^#]+#")


def _extract_marks(df: pd.DataFrame) -> np.ndarray:
    """Turn the DataFrame rows into a (N, MARK_DIM) float32 array."""
    content = df["content"].fillna("").astype(str)
    is_repost = df["origin_weibo"].notna().astype(np.float32).to_numpy()
    content_len = content.str.len().to_numpy()
    hashtag_count = content.str.count(_HASHTAG_RE.pattern).to_numpy()
    url_count = content.str.count(_URL_RE.pattern).to_numpy()

    def log1p(x):
        return np.log1p(x.astype(np.float64)).astype(np.float32)

    out = np.stack([
        log1p(content_len),
        is_repost,
        log1p(df["like_num"].to_numpy()),
        log1p(df["repost_num"].to_numpy()),
        log1p(df["comment_num"].to_numpy()),
        log1p(hashtag_count),
        log1p(url_count),
        np.ones(len(df), dtype=np.float32),
    ], axis=1).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class WeiboSequence:
    """One user's posting sequence, in the MPP convention.

    - ``event_times``: days since ``t0`` (strictly > 0, < T).
    - ``event_marks``: (N, MARK_DIM) features.
    - ``t0, T``: observation window in days (``t0`` < first event, last event < ``T``).
    - ``user_id``: kept for diagnostics.
    - ``abs_times``: original absolute timestamps as ``pd.Timestamp`` list for
      diagnostics (not used by the model).
    """

    event_times: torch.Tensor
    event_marks: torch.Tensor
    t0: float
    T: float
    user_id: str
    abs_times: Optional[List[pd.Timestamp]] = None


# ---------------------------------------------------------------------------
# Loading + grouping
# ---------------------------------------------------------------------------


SECONDS_PER_DAY = 86400.0


def load_csv(path: str) -> pd.DataFrame:
    """Read the Weibo CSV and parse timestamps. Keeps only columns we need."""
    df = pd.read_csv(
        path,
        usecols=["_id", "user_id", "created_at", "like_num", "repost_num",
                 "comment_num", "content", "origin_weibo"],
        low_memory=False,
    )
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"]).reset_index(drop=True)
    return df


def build_sequences(
    df: pd.DataFrame,
    *,
    min_length: int = 5,
    jitter_seconds: float = 1.0,
    pad_head: float = 600.0,    # seconds of pad before first event → t0
    pad_tail: float = 600.0,    # seconds of pad after last event → T
    max_sequences: Optional[int] = None,
    seed: int = 0,
) -> List[WeiboSequence]:
    """Group posts by user, filter short sequences, jitter, emit ``WeiboSequence``.

    Each user's time axis is re-anchored so ``t0 == 0``; event times are in
    **days** relative to ``t0``.

    Parameters
    ----------
    jitter_seconds : uniform ``U(-jitter/2, +jitter/2)`` noise added to each
        event's absolute timestamp *before* rescaling, to break ties that the
        paper's event-time-distinct assumption forbids.
    pad_head / pad_tail : seconds of buffer added on either side of a user's
        first/last events, so the MPP constraint ``t0 < t_1`` and ``t_N < T``
        holds strictly. 10 minutes each is a reasonable default.
    """
    rng = np.random.default_rng(seed)

    per_user: dict[str, list[int]] = {}
    for idx, uid in enumerate(df["user_id"].to_numpy()):
        per_user.setdefault(uid, []).append(idx)

    # Filter minimum length.
    kept_uids = [u for u, idxs in per_user.items() if len(idxs) >= min_length]
    kept_uids.sort(key=lambda u: -len(per_user[u]))  # longest first
    if max_sequences is not None:
        kept_uids = kept_uids[:max_sequences]

    all_marks = _extract_marks(df)               # (N_total, MARK_DIM)
    ts_ns = df["created_at"].astype("int64").to_numpy()  # nanoseconds since epoch

    sequences: List[WeiboSequence] = []
    for uid in kept_uids:
        idxs = np.array(sorted(per_user[uid], key=lambda i: ts_ns[i]))
        abs_secs = ts_ns[idxs] / 1e9
        # Jitter (U(-0.5j, +0.5j) in seconds).
        if jitter_seconds > 0:
            abs_secs = abs_secs + rng.uniform(
                -0.5 * jitter_seconds, 0.5 * jitter_seconds, size=abs_secs.shape
            )
            # Re-sort in case jitter flipped adjacent events.
            order = np.argsort(abs_secs)
            idxs = idxs[order]
            abs_secs = abs_secs[order]

        t0_abs = abs_secs[0] - pad_head
        T_abs = abs_secs[-1] + pad_tail
        # Convert to DAYS relative to t0_abs.
        t_rel = (abs_secs - t0_abs) / SECONDS_PER_DAY
        T_rel = (T_abs - t0_abs) / SECONDS_PER_DAY
        marks = all_marks[idxs]

        # Sanity: strict interior.
        assert 0.0 < t_rel.min() and t_rel.max() < T_rel, (uid, t_rel, T_rel)

        sequences.append(
            WeiboSequence(
                event_times=torch.from_numpy(t_rel).to(torch.float32),
                event_marks=torch.from_numpy(marks).to(torch.float32),
                t0=0.0,
                T=float(T_rel),
                user_id=str(uid),
                abs_times=[pd.Timestamp(int(s * 1e9)) for s in abs_secs],
            )
        )
    return sequences


def sequences_from_path(
    path: str, *, min_length: int = 5, max_sequences: Optional[int] = None,
    **kwargs,
) -> List[WeiboSequence]:
    """One-shot convenience: CSV path → list of ``WeiboSequence``."""
    df = load_csv(path)
    return build_sequences(
        df, min_length=min_length, max_sequences=max_sequences, **kwargs
    )


# ---------------------------------------------------------------------------
# Simple summary
# ---------------------------------------------------------------------------


def summarize(sequences: Iterable[WeiboSequence]) -> str:
    seqs = list(sequences)
    if not seqs:
        return "(no sequences)"
    lens = [len(s.event_times) for s in seqs]
    Ts = [s.T for s in seqs]
    lines = [
        f"sequences: {len(seqs)}",
        f"events total: {sum(lens)}",
        f"events/seq: min={min(lens)}, max={max(lens)}, mean={sum(lens)/len(lens):.1f}",
        f"horizon T (days): min={min(Ts):.2f}, max={max(Ts):.2f}, mean={sum(Ts)/len(Ts):.2f}",
    ]
    return "\n".join(lines)
