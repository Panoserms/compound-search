"""BM25 over pre-tokenised text. A thin wrapper around bm25s, so that every
lexical system differs only in its tokeniser."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import bm25s

# Lucene / Elasticsearch defaults. Fixed before any result was seen; not tuned.
K1 = 1.2
B = 0.75


class BM25Index:
    def __init__(self, tokenizer: Callable[[str], list[str]], k1: float = K1, b: float = B):
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._bm25: bm25s.BM25 | None = None

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "BM25Index":
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        self._ids = list(ids)
        self._bm25 = bm25s.BM25(method="lucene", k1=self.k1, b=self.b)
        self._bm25.index([self.tokenizer(t) for t in texts], show_progress=False)
        return self

    def search(self, queries: Sequence[str], k: int) -> list[list[tuple[str, float]]]:
        """For each query, the top k (passage id, score) pairs, best first."""
        if self._bm25 is None:
            raise RuntimeError("call build() first")
        k = min(k, len(self._ids))
        tokens = [self.tokenizer(q) for q in queries]
        doc_idx, scores = self._bm25.retrieve(tokens, k=k, show_progress=False)
        results = []
        for row_idx, row_scores in zip(doc_idx, scores):
            hits = [(self._ids[int(i)], float(s)) for i, s in zip(row_idx, row_scores)]
            # Equal scores are ordered by passage id, so a rerun gives the same ranking.
            hits.sort(key=lambda hit: (-hit[1], hit[0]))
            results.append(hits)
        return results
