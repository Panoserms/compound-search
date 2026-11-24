"""The BM25 wrapper is checked against a slow reference written from the Lucene formula."""

import math

import pytest

from compound_search.bm25 import B, K1, BM25Index
from compound_search.text import plain_tokens

DOCS = {
    "d1": "Das Bundesverfassungsgericht sitzt in Karlsruhe.",
    "d2": "Karlsruhe liegt in Baden-Württemberg. Karlsruhe ist eine Stadt.",
    "d3": "Die Hauptstadt von Deutschland ist Berlin.",
    "d4": "In Berlin sitzt der Bundestag, in Karlsruhe das Gericht.",
    "d5": "Ein völlig anderer Text über Vögel und Fische im Bodensee.",
}


def reference_scores(query: str) -> dict[str, float]:
    """Lucene BM25: idf = ln(1 + (N - df + 0.5) / (df + 0.5)); tf part = tf / (tf + k1 * (1 - b + b * dl / avgdl))."""
    docs = {pid: plain_tokens(text) for pid, text in DOCS.items()}
    n = len(docs)
    avgdl = sum(len(t) for t in docs.values()) / n
    scores = {}
    for pid, toks in docs.items():
        score = 0.0
        for term in plain_tokens(query):
            df = sum(1 for t in docs.values() if term in t)
            if df == 0:
                continue
            tf = toks.count(term)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            score += idf * tf / (tf + K1 * (1 - B + B * len(toks) / avgdl))
        scores[pid] = score
    return scores


@pytest.fixture(scope="module")
def index() -> BM25Index:
    return BM25Index(plain_tokens).build(list(DOCS), list(DOCS.values()))


@pytest.mark.parametrize("query", ["Wo sitzt das Bundesverfassungsgericht?", "Karlsruhe", "Hauptstadt Berlin Bundestag"])
def test_scores_match_reference(index, query):
    got = dict(index.search([query], k=len(DOCS))[0])
    want = reference_scores(query)
    for pid, score in want.items():
        assert got[pid] == pytest.approx(score, abs=1e-5), pid


def test_results_are_sorted_best_first(index):
    hits = index.search(["Karlsruhe Stadt"], k=3)[0]
    assert [pid for pid, _ in hits][0] == "d2"
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_equal_scores_are_ordered_by_id():
    ids = ["p3", "p1", "p2", "p0"]
    texts = ["gleicher Text hier", "gleicher Text hier", "gleicher Text hier", "etwas ganz anderes"]
    index = BM25Index(plain_tokens).build(ids, texts)
    for k in (3, 4):
        assert [pid for pid, _ in index.search(["gleicher Text"], k=k)[0]][:3] == ["p1", "p2", "p3"]


def test_unknown_words_do_not_crash(index):
    hits = index.search(["Xylophonquarkstrudel"], k=3)[0]
    assert len(hits) == 3
    assert all(score == 0.0 for _, score in hits)


def test_k_larger_than_corpus_is_capped(index):
    assert len(index.search(["Berlin"], k=50)[0]) == len(DOCS)


def test_search_before_build_is_an_error():
    with pytest.raises(RuntimeError):
        BM25Index(plain_tokens).search(["x"], k=1)
