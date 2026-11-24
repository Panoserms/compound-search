"""Run systems on the DEV set and save one JSON report per system.

Usage:  python -I -m compound_search.dev --raw data/raw --out results/dev bm25_plain bm25_stem

The test questions are never searched here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from compound_search.data import DEV_SEED, DEV_SIZE, build_task, load_germandpr, split_dev
from compound_search.pipeline import BASELINE, Workbench
from compound_search.runs import DROPPED, SHIPPED


def dev_workbench(raw: Path, cache: Path | None = None) -> Workbench:
    passages, queries = build_task(load_germandpr(raw))
    cache_dir = cache if cache is not None else raw.parent / "cache"
    return Workbench(passages, split_dev(queries["train"]), cache_dir=cache_dir)


def print_row(report: dict) -> None:
    d = report["pools"][DROPPED]
    s = report["pools"][SHIPPED]["summary"]
    lo, hi = d["summary"]["recall@10_ci95"]
    line = (
        f"{report['system']:28s} R@10 {d['summary']['recall@10']:.3f} [{lo:.3f}, {hi:.3f}]"
        f"  MRR@10 {d['summary']['mrr@10']:.3f}  | as shipped: R@10 {s['recall@10']:.3f}  MRR@10 {s['mrr@10']:.3f}"
    )
    print(line)
    for other, cmp in d["versus"].items():
        r, m = cmp["recall@10"], cmp["rr@10"]
        print(
            f"{'':28s} vs {other}: R@10 {r['diff']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] p={r['p']:.4f}"
            f" (+{r['gained']}/-{r['lost']} queries)   MRR@10 {m['diff']:+.3f} p={m['p']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--versus", default=BASELINE, help="systems to compare each system with, comma-separated")
    parser.add_argument("systems", nargs="+")
    args = parser.parse_args()
    args.versus = [name for name in args.versus.split(",") if name]

    bench = dev_workbench(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        t0 = time.perf_counter()
        report = bench.report(system, versus=args.versus)
        report["split"] = "dev"
        report["dev"] = {"size": DEV_SIZE, "seed": DEV_SEED}
        report["seconds"] = round(time.perf_counter() - t0, 1)
        (args.out / f"{system}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print_row(report)


if __name__ == "__main__":
    main()
