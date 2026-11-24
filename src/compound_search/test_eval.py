"""Stage 6: Run the frozen configuration on the 1,025 GermanDPR test questions once.

Usage:
    python -I -m compound_search.test_eval --raw data/raw --out results/test --summary results/test_summary.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from compound_search.data import build_task, load_germandpr
from compound_search.dev import print_row
from compound_search.pipeline import Workbench
from compound_search.runs import DROPPED, POOLS, SHIPPED
from compound_search.stats import holm

FROZEN_SYSTEMS: tuple[str, ...] = (
    "bm25_plain",
    "bm25_stem",
    "bm25_parts_stem",
    "bm25_split_stem",
    "dense",
    "hybrid_plain_dense",
    "hybrid_stem_dense",
    "hybrid_parts_stem_dense",
    "hybrid_split_stem_dense",
    "hybrid_parts_stem_dense_rewrite",
)

# (id, claim, system, baseline_system)
HYPOTHESIS_TESTS: tuple[tuple[str, str, str, str], ...] = (
    ("H1", "Stems or lemmas help BM25", "bm25_stem", "bm25_plain"),
    ("H2", "Compound splitting helps on top", "bm25_parts_stem", "bm25_stem"),
    (
        "H3",
        "The preprocessing gain survives next to a dense model",
        "hybrid_parts_stem_dense",
        "hybrid_plain_dense",
    ),
    ("H4a", "The hybrid beats the dense model alone", "hybrid_parts_stem_dense", "dense"),
    ("H4b", "The hybrid beats the best lexical system alone", "hybrid_parts_stem_dense", "bm25_parts_stem"),
    (
        "H5",
        "LLM rewriting of unsure queries helps",
        "hybrid_parts_stem_dense_rewrite",
        "hybrid_parts_stem_dense",
    ),
)


def test_workbench(raw: Path, cache: Path | None = None) -> Workbench:
    passages, queries = build_task(load_germandpr(raw))
    cache_dir = cache if cache is not None else raw.parent / "cache"
    return Workbench(passages, queries["test"], cache_dir=cache_dir)


def evaluate_hypotheses(reports: dict[str, dict]) -> dict[str, list[dict]]:
    """Compute the 6 pre-registered hypothesis tests with Holm-corrected p-values for each pool."""
    by_pool: dict[str, list[dict]] = {}
    for pool in POOLS:
        raw_p_values = [
            reports[sys_a]["pools"][pool]["versus"][sys_b]["recall@10"]["p"]
            for _, _, sys_a, sys_b in HYPOTHESIS_TESTS
        ]
        corrected_p = holm(raw_p_values)
        rows: list[dict] = []
        for (hid, claim, sys_a, sys_b), p_holm in zip(HYPOTHESIS_TESTS, corrected_p):
            cmp = reports[sys_a]["pools"][pool]["versus"][sys_b]
            r10 = cmp["recall@10"]
            mrr10 = cmp["rr@10"]
            supported = bool(r10["diff"] > 0 and p_holm < 0.05)
            rows.append(
                {
                    "id": hid,
                    "claim": claim,
                    "system": sys_a,
                    "baseline": sys_b,
                    "system_recall@10": reports[sys_a]["pools"][pool]["summary"]["recall@10"],
                    "baseline_recall@10": reports[sys_b]["pools"][pool]["summary"]["recall@10"],
                    "recall@10_diff": r10["diff"],
                    "recall@10_ci95": r10["ci95"],
                    "recall@10_p_raw": r10["p"],
                    "recall@10_p_holm": p_holm,
                    "gained": r10["gained"],
                    "lost": r10["lost"],
                    "mrr@10_diff": mrr10["diff"],
                    "mrr@10_ci95": mrr10["ci95"],
                    "mrr@10_p_raw": mrr10["p"],
                    "supported": supported,
                }
            )
        by_pool[pool] = rows
    return by_pool


def build_test_summary(reports: dict[str, dict]) -> dict:
    """Assemble the complete Stage 6 test summary, hypothesis table, and headline sentences."""
    hypotheses = evaluate_hypotheses(reports)
    n_test = reports["bm25_plain"]["pools"][DROPPED]["summary"]["n_queries"]
    n_gold = reports["bm25_plain"]["pools"][DROPPED]["summary"]["n_gold_passages"]

    plain_r10 = reports["bm25_plain"]["pools"][DROPPED]["summary"]["recall@10"]
    rewrite_r10 = reports["hybrid_parts_stem_dense_rewrite"]["pools"][DROPPED]["summary"]["recall@10"]
    dense_r10 = reports["dense"]["pools"][DROPPED]["summary"]["recall@10"]
    h3_diff = reports["hybrid_parts_stem_dense"]["pools"][DROPPED]["versus"]["hybrid_plain_dense"]["recall@10"]["diff"]

    diff_rewrite_pts = round((rewrite_r10 - plain_r10) * 100, 1)
    diff_dense_pts = round((dense_r10 - plain_r10) * 100, 1)
    h3_pts = round(h3_diff * 100, 1)

    systems_table = {}
    for name, rep in reports.items():
        systems_table[name] = {
            "description": rep["description"],
            "seconds": rep.get("seconds"),
            "own_hard_negatives_dropped": rep["pools"][DROPPED]["summary"],
            "as_shipped": rep["pools"][SHIPPED]["summary"],
        }

    return {
        "split": "test",
        "n_queries": n_test,
        "n_gold_passages": n_gold,
        "headline_pool": DROPPED,
        "secondary_pool": SHIPPED,
        "systems": systems_table,
        "hypotheses": hypotheses,
        "headline": {
            "baseline_system": "bm25_plain",
            "baseline_recall@10": plain_r10,
            "full_pipeline_system": "hybrid_parts_stem_dense_rewrite",
            "full_pipeline_recall@10": rewrite_r10,
            "full_pipeline_gain_points": diff_rewrite_pts,
            "dense_system": "dense",
            "dense_recall@10": dense_r10,
            "dense_gain_points": diff_dense_pts,
            "h3_preprocessing_gain_in_hybrid_points": h3_pts,
            "sentence_pipeline": (
                f"{diff_rewrite_pts:+.1f} points Recall@10 over plain BM25 on {n_test:,} German questions "
                f"({rewrite_r10 * 100:.1f}% vs. {plain_r10 * 100:.1f}% on GermanDPR test, headline pool; "
                f"{h3_pts:+.1f} points inside the dense hybrid from German stemming and compound splitting)."
            ),
            "sentence_dense": (
                f"{diff_dense_pts:+.1f} points Recall@10 over plain BM25 on {n_test:,} German questions "
                f"({dense_r10 * 100:.1f}% vs. {plain_r10 * 100:.1f}% on GermanDPR test, headline pool; "
                f"linguistic preprocessing adds {h3_pts:+.1f} points inside the RRF hybrid)."
            ),
        },
    }


def run_test_evaluation(
    bench: Workbench,
    systems: Sequence[str] = FROZEN_SYSTEMS,
    out_dir: Path | None = None,
    summary_path: Path | None = None,
) -> dict:
    reports: dict[str, dict] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
    for system in systems:
        t0 = time.perf_counter()
        report = bench.report(system, versus=systems)
        report["split"] = "test"
        report["seconds"] = round(time.perf_counter() - t0, 1)
        reports[system] = report
        if out_dir is not None:
            (out_dir / f"{system}.json").write_text(
                json.dumps(report, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print_row(report)

    summary = build_test_summary(reports)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=Path("results/test_summary.json"))
    args = parser.parse_args()

    bench = test_workbench(args.raw)
    summary = run_test_evaluation(bench, out_dir=args.out, summary_path=args.summary)
    print("\n=== Pre-registered Hypothesis Tests (Headline Pool: own_hard_negatives_dropped) ===")
    for row in summary["hypotheses"][DROPPED]:
        verdict = "SUPPORTED" if row["supported"] else "NOT SUPPORTED"
        lo, hi = row["recall@10_ci95"]
        print(
            f"{row['id']:4s} ({row['system']} vs {row['baseline']}): "
            f"R@10 {row['recall@10_diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
            f"raw_p={row['recall@10_p_raw']:.4f} holm_p={row['recall@10_p_holm']:.4f} "
            f"(+{row['gained']}/-{row['lost']}) -> {verdict}"
        )
    print("\nHeadline (pipeline):", summary["headline"]["sentence_pipeline"])
    print("Headline (dense):   ", summary["headline"]["sentence_dense"])


if __name__ == "__main__":
    main()
