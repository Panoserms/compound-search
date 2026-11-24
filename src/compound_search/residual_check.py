"""How many of the questions that every strong system misses may be label gaps?

    python -I -m compound_search.residual_check --raw data/raw --out results/residual_check.json

A dev question is "residual" if neither the full pipeline nor the dense model alone
has its gold passage in the top ten (headline pool). For each one, this looks at the
ten passages the full pipeline did return: if one of them contains the gold answer
string, the system may have found a right passage that GermanDPR did not label.

Dev set only. Gold answers are used here for diagnosis after the fact, never by a system.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from compound_search.data import Passage, Query
from compound_search.runs import DROPPED, View

FINAL_SYSTEM = "hybrid_parts_stem_dense_rewrite"
DENSE_SYSTEM = "dense"


def gold_in_top10(view: View, query: Query) -> bool:
    return any(pid in query.gold for pid, _ in view[query.qid][:10])


def answer_in_top10(view: View, query: Query, passages: Mapping[str, Passage]) -> bool:
    answers = [a.strip() for a in query.answers if a.strip()]
    return any(a in passages[pid].text for pid, _ in view[query.qid][:10] for a in answers)


def residual_check(
    queries: Sequence[Query], passages: Mapping[str, Passage], final_view: View, dense_view: View
) -> dict:
    residual = [q for q in queries if not gold_in_top10(final_view, q) and not gold_in_top10(dense_view, q)]
    with_answer = [q.qid for q in residual if answer_in_top10(final_view, q, passages)]
    return {
        "split": "dev",
        "pool": DROPPED,
        "final_system": FINAL_SYSTEM,
        "n_queries": len(queries),
        "n_residual": len(residual),
        "n_residual_with_answer_string_in_final_top10": len(with_answer),
        "residual_qids": [q.qid for q in residual],
        "qids_with_answer_string": with_answer,
    }


def main() -> None:
    from compound_search.dev import dev_workbench

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/residual_check.json"))
    args = parser.parse_args()

    bench = dev_workbench(args.raw)
    result = residual_check(
        bench.queries, bench.passages, bench.views(FINAL_SYSTEM)[DROPPED], bench.views(DENSE_SYSTEM)[DROPPED]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{result['n_residual']} residual dev questions; in {result['n_residual_with_answer_string_in_final_top10']} "
        "of them a returned passage contains the gold answer string"
    )


if __name__ == "__main__":
    main()
