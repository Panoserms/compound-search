import pytest

from compound_search.data import Passage, Query, passage_id
from compound_search.pipeline import BASELINE, Workbench
from compound_search.runs import DROPPED, SHIPPED

TEXTS = [
    "Karlsruhe\n\nDas Bundesverfassungsgericht sitzt in Karlsruhe.",
    "Karlsruhe\n\nKarlsruhe ist eine Stadt in Baden-Württemberg mit vielen Gerichten.",
    "Berlin\n\nDie Hauptstadt von Deutschland ist Berlin.",
    "Bodensee\n\nIm Bodensee leben viele Fische. Die Häuser am Ufer sind alt.",
]
PASSAGES = {passage_id(t): Passage(passage_id(t), t.split("\n")[0], t) for t in TEXTS}
QUERIES = [
    Query("q1", "Wo sitzt das Bundesverfassungsgericht?", (passage_id(TEXTS[0]),), ("Karlsruhe",), (passage_id(TEXTS[1]),)),
    Query("q2", "In welchem Haus am Bodensee?", (passage_id(TEXTS[3]),), ("x",), ()),
]


@pytest.fixture()
def bench():
    return Workbench(PASSAGES, QUERIES)


def test_baseline_finds_the_gold_passage(bench):
    view = bench.views(BASELINE)[SHIPPED]
    assert view["q1"][0][0] == passage_id(TEXTS[0])


def test_views_respect_the_pool(bench):
    dropped = [pid for pid, _ in bench.views(BASELINE)[DROPPED]["q1"]]
    shipped = [pid for pid, _ in bench.views(BASELINE)[SHIPPED]["q1"]]
    assert passage_id(TEXTS[1]) in shipped
    assert passage_id(TEXTS[1]) not in dropped


def test_stemming_matches_a_form_that_plain_tokens_miss(bench):
    # "Haus" in the question, "Häuser" in the passage.
    plain_score = dict(bench.views(BASELINE)[SHIPPED]["q2"])[passage_id(TEXTS[3])]
    stem_score = dict(bench.views("bm25_stem")[SHIPPED]["q2"])[passage_id(TEXTS[3])]
    assert stem_score > plain_score


def test_report_compares_a_system_with_the_baseline(bench):
    report = bench.report("bm25_stem")
    assert report["system"] == "bm25_stem"
    assert BASELINE in report["pools"][DROPPED]["versus"]


def test_baseline_report_has_no_self_comparison(bench):
    assert bench.report(BASELINE)["pools"][DROPPED]["versus"] == {}


def test_unknown_system_is_an_error(bench):
    with pytest.raises(KeyError):
        bench.views("no_such_system")


# --- compound splitting systems -------------------------------------------------

from compound_search.splitter import Lexicon  # noqa: E402

SPLIT_TEXTS = [
    "Industrie\n\nDie Konzerne der Tabakindustrie wachsen seit Jahren.",
    "Berlin\n\nDie Hauptstadt von Deutschland ist Berlin, seit Jahren.",
    "Bodensee\n\nIm Bodensee leben viele Fische, seit Jahren.",
]
SPLIT_PASSAGES = {passage_id(t): Passage(passage_id(t), t.split("\n")[0], t) for t in SPLIT_TEXTS}
SPLIT_QUERY = Query("q1", "Seit wann wächst der Tabakkonzern?", (passage_id(SPLIT_TEXTS[0]),), ("x",), ())
TOY_LEXICON = Lexicon(
    freq={"tabak": 50, "konzern": 50, "konzerne": 20, "industrie": 80},
    nouns={"tabak", "konzern", "konzerne", "industrie"},
)


def gold_score(bench, system):
    return dict(bench.views(system)[SHIPPED]["q1"])[passage_id(SPLIT_TEXTS[0])]


def test_splitting_matches_a_compound_through_its_parts():
    bench = Workbench(SPLIT_PASSAGES, [SPLIT_QUERY], lexicon=TOY_LEXICON)
    # Without splitting, "Tabakkonzern" and "Tabakindustrie" share no term.
    assert gold_score(bench, "bm25_split_stem") > gold_score(bench, "bm25_stem")
    assert bench.views("bm25_split_stem")[SHIPPED]["q1"][0][0] == passage_id(SPLIT_TEXTS[0])


def test_every_registered_lexical_system_runs():
    from compound_search.pipeline import LEXICAL

    bench = Workbench(SPLIT_PASSAGES, [SPLIT_QUERY], lexicon=TOY_LEXICON)
    for system in LEXICAL:
        assert len(bench.views(system)[DROPPED]["q1"]) == len(SPLIT_PASSAGES)
    assert {"bm25_split", "bm25_split_stem", "bm25_split_lemma", "bm25_parts_stem", "bm25_hanta_stem"} <= set(LEXICAL)


# --- dense and hybrid systems ---------------------------------------------------

import numpy as np  # noqa: E402
from compound_search.pipeline import DENSE_SYSTEM, HYBRIDS, REWRITTEN_HYBRIDS  # noqa: E402


class ToyEncoder:
    name = "toy-dense"

    def encode(self, texts, prefix: str):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array(
                [
                    1.0 if "bundesverfassungsgericht" in low or "karlsruhe" in low else 0.0,
                    1.0 if "bodensee" in low or "fisch" in low else 0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            v /= np.linalg.norm(v)
            out.append(v)
        return np.vstack(out)


def test_dense_and_hybrid_systems_run_and_fuse_rankings(tmp_path):
    bench = Workbench(PASSAGES, QUERIES, dense_encoder=ToyEncoder(), cache_dir=tmp_path)
    dense_view = bench.views(DENSE_SYSTEM)[DROPPED]
    assert dense_view["q1"][0][0] == passage_id(TEXTS[0])

    for name in HYBRIDS:
        hybrid_view = bench.views(name)[DROPPED]
        assert hybrid_view["q1"][0][0] == passage_id(TEXTS[0])
        report = bench.report(name, versus=(BASELINE, DENSE_SYSTEM))
        assert BASELINE in report["pools"][DROPPED]["versus"]
        assert DENSE_SYSTEM in report["pools"][DROPPED]["versus"]


class ToyRewriter:
    name = "toy-rewriter"

    def __init__(self):
        self.seen = []

    def rewrite(self, query_text: str) -> str:
        self.seen.append(query_text)
        return f"{query_text} Karlsruhe Bundesverfassungsgericht"


def test_rewritten_hybrid_runs_and_expands_only_uncertain_queries(tmp_path):
    rewriter = ToyRewriter()
    bench = Workbench(PASSAGES, QUERIES, dense_encoder=ToyEncoder(), rewriter=rewriter, cache_dir=tmp_path)
    for name in REWRITTEN_HYBRIDS:
        view = bench.views(name)[DROPPED]
        assert view["q1"][0][0] == passage_id(TEXTS[0])
    # q1 has rank-1 agreement between lexical and dense, so it must not be sent to the LLM.
    assert "Wo sitzt das Bundesverfassungsgericht?" not in rewriter.seen




def test_rewritten_hybrid_report_records_which_queries_were_routed(tmp_path):
    rewriter = ToyRewriter()
    bench = Workbench(PASSAGES, QUERIES, dense_encoder=ToyEncoder(), rewriter=rewriter, cache_dir=tmp_path)
    name = next(iter(REWRITTEN_HYBRIDS))
    report = bench.report(name)
    for pool in (DROPPED, SHIPPED):
        routing = report["routing"][pool]
        assert routing["n_routed"] == len(routing["qids"])
        assert routing["share"] == routing["n_routed"] / len(QUERIES)
        assert "q1" not in routing["qids"]  # lexical and dense agree on rank 1
    # Every query the LLM saw is recorded as routed in at least one pool, and nothing else is.
    routed_anywhere = set(report["routing"][DROPPED]["qids"]) | set(report["routing"][SHIPPED]["qids"])
    assert {q.text for q in QUERIES if q.qid in routed_anywhere} == set(rewriter.seen)
    # Systems without a router have no routing block.
    assert "routing" not in bench.report(BASELINE)
