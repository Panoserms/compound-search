"""Ranking metrics. Each function returns one value per query, so that
bootstrap intervals and paired tests can be computed from the same numbers."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of the gold passages that appear in the top k.

    With one gold passage per query (the usual case in GermanDPR) this is
    1.0 for a hit and 0.0 for a miss.
    """
    if not gold:
        raise ValueError("a query needs at least one gold passage")
    top = set(ranked[:k])
    return sum(1 for g in set(gold) if g in top) / len(set(gold))


def reciprocal_rank_at_k(ranked: Sequence[str], gold: Sequence[str], k: int) -> float:
    """1 / rank of the first gold passage in the top k, or 0.0 if there is none."""
    if not gold:
        raise ValueError("a query needs at least one gold passage")
    gold_set = set(gold)
    for rank, pid in enumerate(ranked[:k], start=1):
        if pid in gold_set:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("no values")
    return sum(values) / len(values)
