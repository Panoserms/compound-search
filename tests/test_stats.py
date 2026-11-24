import numpy as np
import pytest

from compound_search.stats import cluster_bootstrap_ci, holm, paired_cluster_bootstrap_ci, sign_flip_p


def test_ci_of_constant_values_is_that_value():
    lo, hi = cluster_bootstrap_ci([0.5] * 20, clusters=list(range(20)))
    assert lo == hi == 0.5


def test_ci_contains_the_mean_and_is_reproducible():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2, size=300).astype(float).tolist()
    clusters = list(range(300))
    first = cluster_bootstrap_ci(values, clusters)
    assert first == cluster_bootstrap_ci(values, clusters)
    assert first[0] < np.mean(values) < first[1]


def test_ci_is_wider_when_questions_share_a_passage():
    # 40 passages with 10 questions each; all questions of a passage succeed or fail together.
    per_passage = [1.0] * 20 + [0.0] * 20
    values = [v for v in per_passage for _ in range(10)]
    shared = [c for c in range(40) for _ in range(10)]
    independent = list(range(400))
    lo_s, hi_s = cluster_bootstrap_ci(values, shared)
    lo_i, hi_i = cluster_bootstrap_ci(values, independent)
    assert (hi_s - lo_s) > 2 * (hi_i - lo_i)


def test_ci_weights_passages_by_their_number_of_questions():
    # One passage with 9 hits, one with a single miss: the mean is 0.9, not 0.5.
    values = [1.0] * 9 + [0.0]
    clusters = ["a"] * 9 + ["b"]
    lo, hi = cluster_bootstrap_ci(values, clusters)
    assert lo <= 0.9 <= hi
    assert hi == 1.0 and lo == 0.0


def test_paired_ci_of_identical_systems_is_zero():
    a = [1.0, 0.0, 1.0, 1.0]
    assert paired_cluster_bootstrap_ci(a, a, clusters=[0, 1, 2, 3]) == (0.0, 0.0, 0.0)


def test_paired_ci_reports_difference_a_minus_b():
    a = [1.0] * 60 + [0.0] * 40
    b = [1.0] * 40 + [0.0] * 60
    diff, lo, hi = paired_cluster_bootstrap_ci(a, b, clusters=list(range(100)))
    assert diff == pytest.approx(0.2)
    assert 0.0 < lo < 0.2 < hi


def test_sign_flip_p_is_one_for_identical_systems():
    a = [1.0, 0.0, 1.0]
    assert sign_flip_p(a, a, clusters=[0, 1, 2]) == 1.0


def test_sign_flip_p_is_small_for_a_consistent_gain():
    a = [1.0] * 30 + [0.0] * 70
    b = [0.0] * 100
    assert sign_flip_p(a, b, clusters=list(range(100))) < 0.001


def test_sign_flip_p_is_large_for_balanced_changes():
    a = [1.0, 0.0] * 50
    b = [0.0, 1.0] * 50
    assert sign_flip_p(a, b, clusters=list(range(100))) > 0.5


def test_sign_flip_p_treats_a_passage_as_one_unit():
    # 30 gains, but all on questions of the same passage: that is one observation, not 30.
    a = [1.0] * 30 + [0.0] * 70
    b = [0.0] * 100
    one_passage = [0] * 30 + list(range(1, 71))
    assert sign_flip_p(a, b, clusters=one_passage) > 0.4


def test_sign_flip_p_is_symmetric():
    a = [1.0] * 12 + [0.0] * 88
    b = [0.0] * 100
    clusters = list(range(100))
    assert sign_flip_p(a, b, clusters) == sign_flip_p(b, a, clusters)


def test_holm_matches_a_hand_computed_example():
    # Sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> kept monotone -> 0.06.
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_keeps_order_of_input():
    assert holm([0.9, 0.001]) == pytest.approx([0.9, 0.002])
    assert holm([0.6, 0.7]) == pytest.approx([1.0, 1.0])


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        cluster_bootstrap_ci([1.0, 0.0], clusters=[0])
    with pytest.raises(ValueError):
        sign_flip_p([1.0], [1.0, 0.0], clusters=[0, 1])
