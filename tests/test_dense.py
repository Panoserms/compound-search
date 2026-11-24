"""Unit tests for the dense retrieval wrapper and E5 mean-pooling helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compound_search.dense import PASSAGE_PREFIX, QUERY_PREFIX, DenseIndex, mean_pool_and_normalize

KEYWORDS = ("karlsruhe", "berlin", "bodensee", "tabak")


class ToyEncoder:
    """Maps each text to a unit vector over 4 keywords, recording every prefix seen."""

    name = "toy-e5"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def encode(self, texts, prefix: str) -> np.ndarray:
        self.calls.append((prefix, list(texts)))
        rows = []
        for text in texts:
            lower = text.lower()
            vec = np.array([1.0 if kw in lower else 0.0 for kw in KEYWORDS], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            rows.append(vec / norm if norm > 0 else vec)
        return np.vstack(rows) if rows else np.zeros((0, len(KEYWORDS)), dtype=np.float32)


DOCS = {
    "d1": "Das Bundesverfassungsgericht sitzt in Karlsruhe.",
    "d2": "Die Hauptstadt von Deutschland ist Berlin.",
    "d3": "Im Bodensee leben viele Fische.",
    "d4": "Tabakindustrie in Berlin und Karlsruhe.",
}


def test_mean_pool_and_normalize_ignores_padding_and_unit_normalises():
    hidden = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 0.0], [99.0, 99.0]],
            [[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = mean_pool_and_normalize(hidden, mask).numpy()
    assert out[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert out[1] == pytest.approx([0.0, 1.0], abs=1e-6)


def test_dense_index_uses_e5_prefixes_and_ranks_by_cosine():
    enc = ToyEncoder()
    index = DenseIndex(encoder=enc).build(list(DOCS), list(DOCS.values()))
    hits = index.search(["Wo liegt Karlsruhe?"], k=2)[0]
    assert enc.calls[0][0] == PASSAGE_PREFIX
    assert enc.calls[1][0] == QUERY_PREFIX
    assert hits[0][0] == "d1"
    assert hits[0][1] == pytest.approx(1.0)
    assert hits[1][0] == "d4"
    assert hits[1][1] == pytest.approx(1.0 / np.sqrt(3), abs=1e-5)


def test_equal_scores_are_ordered_by_passage_id():
    ids = ["p3", "p1", "p2"]
    texts = ["Berlin", "Berlin", "Berlin"]
    index = DenseIndex(encoder=ToyEncoder()).build(ids, texts)
    assert [pid for pid, _ in index.search(["Berlin"], k=3)[0]] == ["p1", "p2", "p3"]


def test_passage_embeddings_are_cached_on_disk(tmp_path):
    enc1 = ToyEncoder()
    DenseIndex(encoder=enc1, cache_dir=tmp_path).build(list(DOCS), list(DOCS.values()))
    assert len(enc1.calls) == 1

    enc2 = ToyEncoder()
    index2 = DenseIndex(encoder=enc2, cache_dir=tmp_path).build(list(DOCS), list(DOCS.values()))
    assert len(enc2.calls) == 0  # loaded from disk without calling encode() on passages
    hits = index2.search(["Bodensee"], k=1)[0]
    assert hits[0][0] == "d3"


def test_search_before_build_is_an_error():
    with pytest.raises(RuntimeError):
        DenseIndex(encoder=ToyEncoder()).search(["Berlin"], k=1)


def test_mismatched_ids_and_texts_is_an_error():
    with pytest.raises(ValueError):
        DenseIndex(encoder=ToyEncoder()).build(["d1"], ["a", "b"])
