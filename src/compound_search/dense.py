"""Dense retrieval with multilingual-e5-small (used as published, no fine-tuning).

Passages are prefixed with 'passage: ' and queries with 'query: ', encoded with
average pooling over the attention mask and L2 normalisation (Wang et al., 2024).
Because passage embeddings depend only on the passage pool, they can be cached
on disk under data/cache/ so that repeated dev runs do not re-encode the pool.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LENGTH = 512
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
BATCH_SIZE = 64


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str], prefix: str) -> np.ndarray: ...


def mean_pool_and_normalize(last_hidden_state, attention_mask):
    """Average token embeddings over non-padding positions, then L2-normalise each row."""
    import torch
    import torch.nn.functional as F

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    pooled = summed / counts
    return F.normalize(pooled, p=2, dim=1)


class E5Encoder:
    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH, batch_size: int = BATCH_SIZE, device: str | None = None):
        import os

        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_TORCH", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)

        if device is None:
            if torch.backends.mps.is_available():
                try:
                    model = model.to("mps")
                    device = "mps"
                except RuntimeError:
                    model = model.to("cpu")
                    device = "cpu"
            else:
                model = model.to("cpu")
                device = "cpu"
        else:
            model = model.to(device)
        self.device = device
        self._model = model
        self._model.eval()

    def encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        import sys
        import time
        import torch

        if not texts:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float32)
        prefixed = [prefix + t for t in texts]
        # Sort by character length so batches pad to similar lengths, then restore order.
        order = sorted(range(len(prefixed)), key=lambda i: len(prefixed[i]))
        ordered_texts = [prefixed[i] for i in order]
        chunks: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.inference_mode():
            for start in range(0, len(ordered_texts), self.batch_size):
                batch = ordered_texts[start : start + self.batch_size]
                encoded = self._tokenizer(
                    batch,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)
                out = self._model(**encoded)
                emb = mean_pool_and_normalize(out.last_hidden_state, encoded["attention_mask"])
                chunks.append(emb.cpu().numpy().astype(np.float32, copy=False))
                if self.device == "mps" and (start // self.batch_size) % 16 == 15:
                    torch.mps.empty_cache()
                done = min(start + self.batch_size, len(ordered_texts))
                if len(ordered_texts) >= 1000 and (done % (self.batch_size * 20) == 0 or done == len(ordered_texts)):
                    elapsed = time.perf_counter() - t0
                    rate = done / max(elapsed, 1e-6)
                    print(f"  [dense:{self.device}] encoded {done}/{len(ordered_texts)} ({rate:.1f} texts/s)", file=sys.stderr, flush=True)
        sorted_emb = np.vstack(chunks)
        restored = np.empty_like(sorted_emb)
        restored[order] = sorted_emb
        return restored


def _cache_path(cache_dir: Path, encoder_name: str, prefix: str, texts: Sequence[str]) -> Path:
    h = hashlib.sha1()
    h.update(encoder_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(prefix.encode("utf-8"))
    h.update(b"\x00")
    for text in texts:
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    slug = encoder_name.replace("/", "__")
    kind = prefix.strip(": ").replace(" ", "_") or "text"
    return cache_dir / f"dense_{slug}_{kind}_{h.hexdigest()[:12]}.npy"


class DenseIndex:
    def __init__(self, encoder: Encoder | None = None, cache_dir: Path | None = None):
        self._encoder = encoder
        self.cache_dir = cache_dir
        self._ids: list[str] = []
        self._embeddings: np.ndarray | None = None

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = E5Encoder()
        return self._encoder

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        encoder_name = self._encoder.name if self._encoder is not None else MODEL_NAME
        cache_file = (
            _cache_path(self.cache_dir, encoder_name, prefix, texts)
            if self.cache_dir is not None
            else None
        )
        if cache_file is not None and cache_file.exists():
            return np.load(cache_file)
        emb = np.asarray(self.encoder.encode(texts, prefix), dtype=np.float32)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, emb)
        return emb

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "DenseIndex":
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self._ids = list(ids)
        self._embeddings = self._encode(texts, PASSAGE_PREFIX)
        return self

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """For each query, the top k (passage id, cosine score) pairs, best first."""
        if self._embeddings is None:
            raise RuntimeError("call build() first")
        k = min(k, len(self._ids))
        q_emb = self._encode(queries, QUERY_PREFIX)
        scores_matrix = q_emb @ self._embeddings.T
        results: list[list[tuple[str, float]]] = []
        for row in scores_matrix:
            if k < len(self._ids):
                top_idx = np.argpartition(-row, k - 1)[:k]
            else:
                top_idx = np.arange(len(self._ids))
            hits = [(self._ids[int(i)], float(row[int(i)])) for i in top_idx]
            hits.sort(key=lambda hit: (-hit[1], hit[0]))
            results.append(hits)
        return results
