"""Stage 0 scoring: the two pools and the hard-negative diagnosis."""

from __future__ import annotations

from compound_search.bm25 import BM25Index
from compound_search.data import Passage, Query, passage_id
from compound_search.stage0 import evaluate
from compound_search.text import plain_tokens

TEXTS = [
    "Karlsruhe\n\nDas Gericht sitzt in Karlsruhe.",
    "Gericht\n\nGericht Gericht Gericht sitzt sitzt Karlsruhe Karlsruhe, das Gericht sitzt.",
    "Berlin\n\nDie Hauptstadt von Deutschland ist Berlin.",
]
PASSAGES = {passage_id(t): Passage(passage_id(t), t.split("\n")[0], t) for t in TEXTS}
GOLD, PLANTED, OTHER = (passage_id(t) for t in TEXTS)
QUERIES = [
    Query("q1", "Wo sitzt das Gericht?", (GOLD,), ("Karlsruhe",), (PLANTED,)),
    Query("q2", "Was ist die Hauptstadt von Deutschland?", (OTHER,), ("Berlin",), ()),
]


def _index() -> BM25Index:
    return BM25Index(plain_tokens).build(list(PASSAGES), [p.text for p in PASSAGES.values()])


def test_as_shipped_keeps_the_hard_negative_diagnosis():
    out = evaluate(_index(), QUERIES, PASSAGES, drop_own_hard_negatives=False)
    stats = out["own_hard_negatives"]
    # q1's planted passage repeats the query words, so it outranks the gold passage; q2 has none.
    assert stats["share_of_queries_where_one_outranks_gold"] == 0.5
    assert stats["share_of_queries_where_top1_is_one"] == 0.5
    assert out["n_queries"] == 2
    assert out["recall@10"] == 1.0


def test_dropping_own_hard_negatives_moves_the_gold_passage_up():
    shipped = evaluate(_index(), QUERIES, PASSAGES, drop_own_hard_negatives=False)
    dropped = evaluate(_index(), QUERIES, PASSAGES, drop_own_hard_negatives=True)
    assert "own_hard_negatives" not in dropped
    assert dropped["mrr@10"] > shipped["mrr@10"]
    assert dropped["recall@1"] == 1.0


def test_intervals_resample_gold_passages():
    out = evaluate(_index(), QUERIES, PASSAGES, drop_own_hard_negatives=True)
    lo, hi = out["recall@10_ci95"]
    assert lo <= out["recall@10"] <= hi
