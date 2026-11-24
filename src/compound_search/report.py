"""Turn the views of one system into the JSON report that is saved under results/."""

from __future__ import annotations

from collections.abc import Sequence

from compound_search.data import Query
from compound_search.runs import POOLS, Run, View, compare, pool_view, score_view, summarise

Views = dict[str, View]  # pool name -> view


def views_of_run(run: Run, queries: Sequence[Query]) -> Views:
    return {pool: pool_view(run, queries, pool) for pool in POOLS}


def build_report(
    system: str,
    description: str,
    views: Views,
    queries: Sequence[Query],
    versus: dict[str, Views] | None = None,
) -> dict:
    """Summary, per-query gold ranks and paired comparisons, for each pool."""
    report: dict = {"system": system, "description": description, "pools": {}}
    for pool in POOLS:
        scores = score_view(views[pool], queries)
        report["pools"][pool] = {
            "summary": summarise(scores, queries),
            "versus": {
                other: compare(scores, score_view(other_views[pool], queries), queries)
                for other, other_views in (versus or {}).items()
            },
            "gold_rank": {q.qid: scores[q.qid]["gold_rank"] for q in queries},
        }
    return report
