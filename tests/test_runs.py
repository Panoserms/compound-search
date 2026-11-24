import pytest

from compound_search.data import Query
from compound_search.runs import DROPPED, SHIPPED, Run, compare, load_run, pool_view, rrf, save_run, score_view, summarise

Q1 = Query("q1", "Frage eins?", gold=("g1",), answers=("a",), own_hard_negatives=("n1", "n2"))
Q2 = Query("q2", "Frage zwei?", gold=("g2",), answers=("a",), own_hard_negatives=())
Q3 = Query("q3", "Frage drei?", gold=("g1",), answers=("a",), own_hard_negatives=())


def hits(*pids):
    return [(pid, float(len(pids) - i)) for i, pid in enumerate(pids)]


RUN = Run("toy", {"q1": hits("n1", "x", "g1", "n2", "y"), "q2": hits("x", "y", "z"), "q3": hits("g1", "x")})


def test_shipped_pool_keeps_every_passage():
    view = pool_view(RUN, [Q1], SHIPPED, depth=10)
    assert [p for p, _ in view["q1"]] == ["n1", "x", "g1", "n2", "y"]


def test_dropped_pool_removes_only_the_questions_own_hard_negatives():
    view = pool_view(RUN, [Q1, Q2], DROPPED, depth=10)
    assert [p for p, _ in view["q1"]] == ["x", "g1", "y"]
    assert [p for p, _ in view["q2"]] == ["x", "y", "z"]


def test_view_is_cut_to_depth_after_dropping():
    view = pool_view(RUN, [Q1], DROPPED, depth=2)
    assert [p for p, _ in view["q1"]] == ["x", "g1"]


def test_unknown_pool_is_an_error():
    with pytest.raises(ValueError):
        pool_view(RUN, [Q1], "something", depth=10)


def test_rrf_scores_are_sums_of_reciprocal_ranks():
    fused = dict(rrf([hits("a", "b"), hits("b", "c")], k=60, depth=10))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_ranks_a_passage_found_by_both_lists_first():
    fused = rrf([hits("a", "b", "c"), hits("d", "b", "e")], k=60, depth=10)
    assert fused[0][0] == "b"


def test_rrf_breaks_ties_by_passage_id_and_cuts_to_depth():
    fused = rrf([hits("b", "x"), hits("a", "y")], k=60, depth=3)
    assert [p for p, _ in fused] == ["a", "b", "x"]


def test_score_view_gives_rank_recall_and_reciprocal_rank():
    scores = score_view(pool_view(RUN, [Q1, Q2], DROPPED, depth=10), [Q1, Q2])
    assert scores["q1"] == {"gold_rank": 2, "recall@10": 1.0, "rr@10": 0.5}
    assert scores["q2"] == {"gold_rank": None, "recall@10": 0.0, "rr@10": 0.0}


def test_summarise_reports_means_and_intervals():
    queries = [Q1, Q2, Q3]
    summary = summarise(score_view(pool_view(RUN, queries, DROPPED, depth=10), queries), queries)
    assert summary["n_queries"] == 3
    assert summary["n_gold_passages"] == 2
    assert summary["recall@10"] == pytest.approx(2 / 3)
    assert summary["recall@1"] == pytest.approx(1 / 3)
    assert summary["mrr@10"] == pytest.approx((0.5 + 0 + 1) / 3)
    lo, hi = summary["recall@10_ci95"]
    assert lo <= summary["recall@10"] <= hi


def test_compare_reports_a_minus_b():
    queries = [Q1, Q2, Q3]
    dropped = score_view(pool_view(RUN, queries, DROPPED, depth=1), queries)   # only q3 hits at rank 1
    deep = score_view(pool_view(RUN, queries, DROPPED, depth=10), queries)     # q1 and q3 hit
    result = compare(deep, dropped, queries)
    assert result["recall@10"]["diff"] == pytest.approx(1 / 3)
    assert set(result["recall@10"]) == {"diff", "ci95", "p", "gained", "lost"}
    assert result["recall@10"]["gained"] == 1 and result["recall@10"]["lost"] == 0
    assert 0 < result["recall@10"]["p"] <= 1


def test_run_survives_a_round_trip_to_disk(tmp_path):
    path = tmp_path / "runs" / "toy.json"
    save_run(RUN, path)
    assert load_run(path) == RUN
