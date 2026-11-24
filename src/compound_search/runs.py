"""Rankings, the two pool variants, rank fusion, and scoring.

A Run holds the raw ranking of one system for a set of queries. A view is that
ranking as seen through one pool variant, cut to DEPTH. Everything that is
scored or fused works on views.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from compound_search.data import Query
from compound_search.metrics import mean, recall_at_k, reciprocal_rank_at_k
from compound_search.stats import cluster_bootstrap_ci, paired_cluster_bootstrap_ci, sign_flip_p

DROPPED = "own_hard_negatives_dropped"  # headline pool
SHIPPED = "as_shipped"
POOLS = (DROPPED, SHIPPED)
DEPTH = 100
RAW_DEPTH = 110  # room for the hard negatives that the headline pool drops
RRF_K = 60
KS = (1, 5, 10, 20, 100)

Hits = list[tuple[str, float]]
View = dict[str, Hits]


@dataclass
class Run:
    system: str
    hits: dict[str, Hits]


def save_run(run: Run, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"system": run.system, "hits": run.hits}), encoding="utf-8")


def load_run(path: Path) -> Run:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Run(raw["system"], {qid: [(pid, score) for pid, score in hits] for qid, hits in raw["hits"].items()})


def pool_view(run: Run, queries: Sequence[Query], pool: str, depth: int = DEPTH) -> View:
    if pool not in POOLS:
        raise ValueError(f"unknown pool: {pool}")
    view = {}
    for q in queries:
        skip = set(q.own_hard_negatives) if pool == DROPPED else set()
        view[q.qid] = [(pid, score) for pid, score in run.hits[q.qid] if pid not in skip][:depth]
    return view


def rrf(rankings: Sequence[Hits], k: int = RRF_K, depth: int = DEPTH) -> Hits:
    """Reciprocal rank fusion: each list adds 1 / (k + rank) to a passage's score."""
    fused: dict[str, float] = {}
    for hits in rankings:
        for rank, (pid, _) in enumerate(hits, start=1):
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda hit: (-hit[1], hit[0]))[:depth]


def score_view(view: View, queries: Sequence[Query]) -> dict[str, dict]:
    scores = {}
    for q in queries:
        ranked = [pid for pid, _ in view[q.qid]]
        gold_rank = next((i for i, pid in enumerate(ranked, start=1) if pid in q.gold), None)
        scores[q.qid] = {
            "gold_rank": gold_rank,
            "recall@10": recall_at_k(ranked, q.gold, 10),
            "rr@10": reciprocal_rank_at_k(ranked, q.gold, 10),
        }
    return scores


def _clusters(queries: Sequence[Query]) -> list[str]:
    """The unit of resampling: the (first) gold passage of each question."""
    return [q.gold[0] for q in queries]


def summarise(scores: dict[str, dict], queries: Sequence[Query]) -> dict:
    clusters = _clusters(queries)
    recall10 = [scores[q.qid]["recall@10"] for q in queries]
    rr10 = [scores[q.qid]["rr@10"] for q in queries]
    ranks = [scores[q.qid]["gold_rank"] for q in queries]
    out: dict = {"n_queries": len(queries), "n_gold_passages": len(set(clusters))}
    for k in KS:
        # Rank-based for every k except 10, where the exact (possibly fractional) recall is used.
        out[f"recall@{k}"] = mean(recall10) if k == 10 else mean([float(r is not None and r <= k) for r in ranks])
    out["mrr@10"] = mean(rr10)
    out["recall@10_ci95"] = list(cluster_bootstrap_ci(recall10, clusters))
    out["mrr@10_ci95"] = list(cluster_bootstrap_ci(rr10, clusters))
    return out


def compare(a: dict[str, dict], b: dict[str, dict], queries: Sequence[Query]) -> dict:
    """Paired comparison a minus b: difference, interval, p value, and how many queries changed."""
    clusters = _clusters(queries)
    out = {}
    for metric in ("recall@10", "rr@10"):
        va = [a[q.qid][metric] for q in queries]
        vb = [b[q.qid][metric] for q in queries]
        diff, lo, hi = paired_cluster_bootstrap_ci(va, vb, clusters)
        out[metric] = {
            "diff": diff,
            "ci95": [lo, hi],
            "p": sign_flip_p(va, vb, clusters),
            "gained": sum(1 for x, y in zip(va, vb) if x > y),
            "lost": sum(1 for x, y in zip(va, vb) if x < y),
        }
    return out
