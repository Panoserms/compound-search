import pytest

from compound_search.metrics import mean, recall_at_k, reciprocal_rank_at_k


def test_recall_hit_and_miss_with_one_gold():
    assert recall_at_k(["a", "b", "c"], ["c"], k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_with_two_gold_is_a_share():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5


def test_recall_ignores_repeated_gold_ids():
    assert recall_at_k(["a"], ["a", "a"], k=1) == 1.0


def test_reciprocal_rank_uses_first_gold_only():
    assert reciprocal_rank_at_k(["x", "a", "b"], ["a", "b"], k=10) == 0.5


def test_reciprocal_rank_is_zero_beyond_k():
    assert reciprocal_rank_at_k(["x", "y", "a"], ["a"], k=2) == 0.0


def test_empty_gold_is_an_error():
    with pytest.raises(ValueError):
        recall_at_k(["a"], [], k=1)
    with pytest.raises(ValueError):
        reciprocal_rank_at_k(["a"], [], k=1)


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    with pytest.raises(ValueError):
        mean([])
