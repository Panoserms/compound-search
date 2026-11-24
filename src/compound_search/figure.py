"""The results chart of the README, drawn from results/test_summary.json.

    python -m compound_search.figure --results results --out docs/recall_at_10.svg           # write the chart
    python -m compound_search.figure --results results --out docs/recall_at_10.svg --check   # exit 1 if it is out of date

A dot per system with its 95 % interval, on an axis from 60 % to 100 % (a dot plot,
not bars, so the axis need not start at zero; the chart says where it starts).
Plain SVG, no plotting library. The colours follow the reader's light or dark mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path

from compound_search.runs import DROPPED
from compound_search.tables import LABELS

GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Keyword search only (BM25)", ("bm25_plain", "bm25_stem", "bm25_split_stem", "bm25_parts_stem")),
    ("Dense model only", ("dense",)),
    (
        "Both, fused",
        (
            "hybrid_plain_dense",
            "hybrid_stem_dense",
            "hybrid_split_stem_dense",
            "hybrid_parts_stem_dense",
            "hybrid_parts_stem_dense_rewrite",
        ),
    ),
)

WIDTH = 860
LABEL_X = 16
PLOT_X0, PLOT_X1 = 372, 772
VALUE_X = 844
AXIS_LO, AXIS_HI = 0.60, 1.00
TICKS = (0.60, 0.70, 0.80, 0.90, 1.00)
ROW, GROUP_ROW, TOP, BOTTOM = 30, 30, 46, 40

STYLE = """
  text { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; fill: #1f2328; }
  .title { font-size: 14px; font-weight: 600; }
  .group { font-size: 12px; fill: #59636e; }
  .label { font-size: 13px; }
  .value { font-size: 13px; font-variant-numeric: tabular-nums; }
  .tick { font-size: 11px; fill: #59636e; }
  .grid { stroke: #d1d9e0; stroke-width: 1; }
  .ci { stroke: #59636e; stroke-width: 2; stroke-linecap: round; }
  .dot { fill: #1f2328; }
  @media (prefers-color-scheme: dark) {
    text { fill: #f0f6fc; }
    .group, .tick { fill: #9198a1; }
    .grid { stroke: #3d444d; }
    .ci { stroke: #9198a1; }
    .dot { fill: #f0f6fc; }
  }
""".rstrip()


def x_of(value: float) -> float:
    clipped = min(AXIS_HI, max(AXIS_LO, value))
    return PLOT_X0 + (clipped - AXIS_LO) / (AXIS_HI - AXIS_LO) * (PLOT_X1 - PLOT_X0)


def render_svg(summary: dict) -> str:
    systems = summary["systems"]
    rows: list[tuple[str, str | None]] = []  # ("group", title) or ("system", name)
    for title, names in GROUPS:
        rows.append(("group", title))
        rows.extend(("system", name) for name in names)
    height = TOP + sum(GROUP_ROW if kind == "group" else ROW for kind, _ in rows) + BOTTOM
    plot_bottom = height - BOTTOM

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" role="img" '
        f'aria-label="Recall at 10 of the ten tested systems on the {summary["n_queries"]:,} test questions">',
        f"<style>{STYLE}\n</style>",
        f'<text class="title" x="{LABEL_X}" y="24">Recall@10 on the {summary["n_queries"]:,} held-out test questions, with 95% intervals</text>',
    ]
    for tick in TICKS:
        x = x_of(tick)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{TOP}" x2="{x:.1f}" y2="{plot_bottom}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle">{tick * 100:.0f}%</text>')
    out.append(f'<text class="tick" x="{LABEL_X}" y="{plot_bottom + 18}">The axis starts at 60%.</text>')

    y = TOP
    for kind, name in rows:
        if kind == "group":
            out.append(f'<text class="group" x="{LABEL_X}" y="{y + 20}">{escape(name)}</text>')
            y += GROUP_ROW
            continue
        score = systems[name][DROPPED]
        lo, hi = score["recall@10_ci95"]
        mid = y + ROW / 2
        out.append(f'<text class="label" x="{LABEL_X}" y="{mid + 4.5:.1f}">{escape(LABELS[name])}</text>')
        out.append(f'<line class="ci" x1="{x_of(lo):.1f}" y1="{mid:.1f}" x2="{x_of(hi):.1f}" y2="{mid:.1f}"/>')
        out.append(f'<circle class="dot" cx="{x_of(score["recall@10"]):.1f}" cy="{mid:.1f}" r="5"/>')
        out.append(f'<text class="value" x="{VALUE_X}" y="{mid + 4.5:.1f}" text-anchor="end">{score["recall@10"] * 100:.1f}%</text>')
        y += ROW
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs/recall_at_10.svg"))
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if the chart is out of date")
    args = parser.parse_args()

    svg = render_svg(json.loads((args.results / "test_summary.json").read_text(encoding="utf-8")))
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != svg:
            print(f"{args.out} is out of date: run python -m compound_search.figure", file=sys.stderr)
            raise SystemExit(1)
        print(f"{args.out}: matches {args.results}/test_summary.json")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
