"""RoBERTa-wwm-ext text encoder for Weibo marks (paper §4).

We load ``hfl/chinese-roberta-wwm-ext`` from HuggingFace and expose an
``encode`` method that takes a list of strings and returns the ``[CLS]``
representation of each, as a ``(N, 768)`` tensor.

The model is held on the configured device (MPS by default on Apple Silicon)
and set to ``eval()``; inference runs under ``torch.no_grad()``. Encoding is
batched and supports iterating over large lists via a progress counter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import math
import torch

_MODEL_NAME_DEFAULT = "hfl/chinese-roberta-wwm-ext"
_CLS_DIM = 768


@dataclass
class EncoderConfig:
    model_name: str = _MODEL_NAME_DEFAULT
    max_length: int = 256
    batch_size: int = 32
    device: Optional[str] = None     # "mps", "cuda", "cpu"; auto if None
    dtype: str = "float16"           # float16 for storage, float32 for model


class WeiboTextEncoder:
    """Batched [CLS] encoder for Chinese Weibo posts.

    Loads lazily so tests can import the module without downloading the model.
    """

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        self.cfg = cfg or EncoderConfig()
        self._model = None
        self._tokenizer = None

    # ---- lazy loaders ----
    def _load(self):
        if self._model is not None:
            return
        # Import lazily so that missing transformers doesn't break our tests.
        from transformers import AutoTokenizer, AutoModel

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name)
        model = AutoModel.from_pretrained(self.cfg.model_name)
        model.eval()
        device = self.cfg.device or (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model = model.to(device)
        self._model = model
        self._device = device

    # ---- public API ----
    @property
    def device(self) -> str:
        self._load()
        return self._device

    @torch.no_grad()
    def encode(self, texts: List[str], progress: bool = False) -> torch.Tensor:
        """Return ``(len(texts), 768)`` [CLS] embeddings, always on CPU.

        We move outputs to CPU and cast to ``cfg.dtype`` immediately after
        each batch so peak GPU memory stays bounded and the caller can
        freely concatenate without device thrash.
        """
        self._load()
        if not texts:
            return torch.empty(0, _CLS_DIM)

        dtype = getattr(torch, self.cfg.dtype)
        n = len(texts)
        out = torch.empty(n, _CLS_DIM, dtype=dtype)
        bs = self.cfg.batch_size
        n_batches = math.ceil(n / bs)

        for b in range(n_batches):
            lo, hi = b * bs, min((b + 1) * bs, n)
            batch_texts = texts[lo:hi]
            # Strings may contain None / NaN from pandas — coerce.
            batch_texts = [str(t) if t is not None else "" for t in batch_texts]
            enc = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self._device)
            outputs = self._model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            out[lo:hi] = cls.to("cpu").to(dtype)
            if progress and (b % 20 == 0 or b == n_batches - 1):
                print(f"  [encode] batch {b+1}/{n_batches} ({hi}/{n} texts)")
        return out

    def encode_iter(self, texts: Iterable[str]) -> torch.Tensor:
        return self.encode(list(texts))
