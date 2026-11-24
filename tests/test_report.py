import pytest

from compound_search.data import Query
from compound_search.report import build_report, views_of_run
from compound_search.runs import DROPPED, POOLS, SHIPPED, Run

Q1 = Query("q1", "Frage eins?", gold=("g1",), answers=("a",), own_hard_negatives=("n1",))
Q2 = Query("q2", "Frage zwei?", gold=("g2",), answers=("a",), own_hard_negatives=())
QUERIES = [Q1, Q2]


def hits(*pids):
    return [(pid, float(len(pids) - i)) for i, pid in enumerate(pids)]


BASE = Run("base", {"q1": hits("n1", "x", "g1"), "q2": hits("x", "y")})
BETTER = Run("better", {"q1": hits("g1", "n1", "x"), "q2": hits("x", "g2")})


def test_views_of_run_has_one_view_per_pool():
    views = views_of_run(BASE, QUERIES)
    assert set(views) == set(POOLS)
    assert [p for p, _ in views[DROPPED]["q1"]] == ["x", "g1"]
    assert [p for p, _ in views[SHIPPED]["q1"]] == ["n1", "x", "g1"]


def test_report_has_a_summary_for_each_pool():
    report = build_report("base", "toy baseline", views_of_run(BASE, QUERIES), QUERIES)
    assert report["system"] == "base"
    assert report["description"] == "toy baseline"
    assert report["pools"][DROPPED]["summary"]["recall@10"] == 0.5
    assert report["pools"][SHIPPED]["summary"]["mrr@10"] == pytest.approx((1 / 3 + 0) / 2)


def test_report_lists_the_gold_rank_of_every_query_per_pool():
    report = build_report("base", "toy baseline", views_of_run(BASE, QUERIES), QUERIES)
    assert report["pools"][DROPPED]["gold_rank"] == {"q1": 2, "q2": None}
    assert report["pools"][SHIPPED]["gold_rank"] == {"q1": 3, "q2": None}


def test_report_without_comparisons_has_none():
    report = build_report("base", "toy baseline", views_of_run(BASE, QUERIES), QUERIES)
    assert report["pools"][DROPPED]["versus"] == {}


def test_report_compares_against_each_named_system():
    report = build_report(
        "better",
        "toy improvement",
        views_of_run(BETTER, QUERIES),
        QUERIES,
        versus={"base": views_of_run(BASE, QUERIES)},
    )
    cmp = report["pools"][DROPPED]["versus"]["base"]["recall@10"]
    assert cmp["diff"] == pytest.approx(0.5)
    assert cmp["gained"] == 1 and cmp["lost"] == 0
