"""Intervals and paired tests for per-query scores.

Several questions share one gold passage, so the unit that is resampled is the
passage ("cluster"), not the question. Treating questions as independent would
make the intervals too narrow.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

N_RESAMPLES = 10_000
SEED = 13
_CHUNK = 1_000


def _per_cluster(values: Sequence[float], clusters: Sequence[Hashable]) -> tuple[np.ndarray, np.ndarray]:
    """Sum of values and number of values for each cluster."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters differ in length")
    if len(values) == 0:
        raise ValueError("no values")
    index = {c: i for i, c in enumerate(dict.fromkeys(clusters))}
    sums = np.zeros(len(index))
    counts = np.zeros(len(index))
    for v, c in zip(values, clusters):
        sums[index[c]] += v
        counts[index[c]] += 1
    return sums, counts


def _bootstrap_means(sums: np.ndarray, counts: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(sums)
    means = []
    for start in range(0, n_resamples, _CHUNK):
        idx = rng.integers(0, n, size=(min(_CHUNK, n_resamples - start), n))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return np.concatenate(means)


def cluster_bootstrap_ci(
    values: Sequence[float], clusters: Sequence[Hashable], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> tuple[float, float]:
    """95 % percentile interval of the mean, resampling clusters with replacement."""
    sums, counts = _per_cluster(values, clusters)
    lo, hi = np.percentile(_bootstrap_means(sums, counts, n_resamples, seed), [2.5, 97.5])
    return float(lo), float(hi)


def paired_cluster_bootstrap_ci(
    a: Sequence[float], b: Sequence[float], clusters: Sequence[Hashable], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean of a - b, lower, upper): 95 % interval of the paired difference."""
    if len(a) != len(b):
        raise ValueError("a and b differ in length")
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sums, counts = _per_cluster(diffs.tolist(), clusters)
    lo, hi = np.percentile(_bootstrap_means(sums, counts, n_resamples, seed), [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def sign_flip_p(
    a: Sequence[float], b: Sequence[float], clusters: Sequence[Hashable], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> float:
    """Two-sided paired randomisation test.

    Under "no difference between a and b", swapping the two systems for all
    questions of one passage is equally likely, which flips the sign of that
    passage's summed difference. The p value is the share of random sign
    patterns whose total is at least as far from zero as the observed total.
    """
    if len(a) != len(b):
        raise ValueError("a and b differ in length")
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sums, _ = _per_cluster(diffs.tolist(), clusters)
    observed = abs(sums.sum())
    rng = np.random.default_rng(seed)
    at_least = 0
    for start in range(0, n_resamples, _CHUNK):
        signs = rng.choice([-1.0, 1.0], size=(min(_CHUNK, n_resamples - start), len(sums)))
        at_least += int((np.abs(signs @ sums) >= observed - 1e-12).sum())
    return (at_least + 1) / (n_resamples + 1)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-corrected p values, in the order of the input."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted
