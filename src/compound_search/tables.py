"""The result tables of the README, written from the files under results/.

    python -m compound_search.tables --results results --readme README.md           # rewrite the tables
    python -m compound_search.tables --results results --readme README.md --check   # exit 1 if they are out of date

Each table sits in the README between two marker comments. Nothing between the
markers is typed by hand, so a number in a table cannot drift from the run that
produced it. `tests/test_tables.py` runs the check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from compound_search.labels import load_splitter_eval
from compound_search.runs import DROPPED, SHIPPED
from compound_search.stats import N_RESAMPLES

# Short names for readers who do not know the system ids.
LABELS: dict[str, str] = {
    "bm25_plain": "BM25 on plain words (baseline)",
    "bm25_stem": "BM25 on stems",
    "bm25_lemma": "BM25 on lemmas",
    "bm25_split": "BM25 on plain words + compound parts",
    "bm25_split_stem": "BM25 on stems + compound parts, whole word kept",
    "bm25_split_lemma": "BM25 on lemmas + compound parts, whole word kept",
    "bm25_parts_stem": "BM25 on stems, compounds replaced by their parts",
    "bm25_hanta_stem": "BM25 on stems + HanTa compound parts",
    "dense": "Dense model alone (multilingual-e5-small)",
    "hybrid_plain_dense": "Hybrid: plain BM25 + dense",
    "hybrid_stem_dense": "Hybrid: stemmed BM25 + dense",
    "hybrid_split_stem_dense": "Hybrid: stems + parts (whole kept) + dense",
    "hybrid_parts_stem_dense": "Hybrid: stems + parts (replaced) + dense",
    "bm25_parts": "BM25 on plain words, compounds replaced by their parts (added after the test run, dev only)",
    "hybrid_parts_dense": "Hybrid: plain words with parts (replaced) + dense (added after the test run, dev only)",
    "hybrid_parts_stem_dense_rewrite": "Hybrid + LLM rewriting of unsure queries",
    "hybrid_split_stem_dense_rewrite": "Hybrid (whole kept) + LLM rewriting of unsure queries",
}

TEST_ORDER = (
    "bm25_plain",
    "bm25_stem",
    "bm25_split_stem",
    "bm25_parts_stem",
    "dense",
    "hybrid_plain_dense",
    "hybrid_stem_dense",
    "hybrid_split_stem_dense",
    "hybrid_parts_stem_dense",
    "hybrid_parts_stem_dense_rewrite",
)

# The predictions of HYPOTHESES.md, Section 3, written before any of these systems was built.
# (text as written, lower and upper end in points; None = "a gain" or "about 0" without a range)
PREDICTIONS: dict[str, tuple[str, float | None, float | None]] = {
    "H1": ("+1 to +3", 1.0, 3.0),
    "H2": ("0 to +2", 0.0, 2.0),
    "H3": ("0 to +1 (shrinks)", 0.0, 1.0),
    "H4a": ("a gain", None, None),
    "H4b": ("a gain", None, None),
    "H5": ("about 0", None, None),
}

_P_FLOOR = 1.0 / (N_RESAMPLES + 1)  # the smallest p value a 10,000-draw randomisation test can return


def fmt_p(p: float) -> str:
    """Three decimals; anything below 0.001 is written as such, because the test cannot resolve it further."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def points(x: float) -> str:
    """A difference of two shares, in percentage points with a sign."""
    value = round(x * 100, 1)
    return f"{value + 0.0:+.1f}".replace("-", "−")


def prediction_outcome(hid: str, row: dict) -> str:
    """How the pre-written prediction compares with the test result."""
    _, lo, hi = PREDICTIONS[hid]
    diff = round(row["recall@10_diff"] * 100, 1)
    if lo is not None and hi is not None:
        if diff > hi:
            return "larger than predicted"
        if diff < lo:
            return "smaller than predicted"
        return "as predicted"
    if PREDICTIONS[hid][0] == "about 0":
        return "as predicted" if not row["supported"] else "larger than predicted"
    return "as predicted" if row["supported"] else "no gain found"


def _table(header: list[str], align: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _ci(pair: list[float], digits: int = 3) -> str:
    return f"[{pair[0]:.{digits}f}, {pair[1]:.{digits}f}]"


def systems_table(systems: dict[str, dict], order: list[str] | tuple[str, ...]) -> str:
    """One row per system: headline pool in full, the as-shipped pool beside it."""
    rows = []
    for name in order:
        d, s = systems[name][DROPPED], systems[name][SHIPPED]
        rows.append(
            [
                f"{LABELS.get(name, name)}<br>`{name}`",
                f"{d['recall@1']:.3f}",
                f"{d['recall@5']:.3f}",
                f"**{d['recall@10']:.3f}** {_ci(d['recall@10_ci95'])}",
                f"{d['recall@100']:.3f}",
                f"{d['mrr@10']:.3f}",
                f"{s['recall@10']:.3f}",
                f"{s['mrr@10']:.3f}",
            ]
        )
    header = ["System", "R@1", "R@5", "R@10 [95 % CI]", "R@100", "MRR@10", "as shipped: R@10", "as shipped: MRR@10"]
    return _table(header, ["---", "---:", "---:", "---", "---:", "---:", "---:", "---:"], rows)


def hypotheses_table(rows: list[dict], with_predictions: bool = True) -> str:
    out = []
    for r in rows:
        lo, hi = r["recall@10_ci95"]
        cells = [
            f"**{r['id']}**",
            r["claim"],
            f"`{r['system']}` vs. `{r['baseline']}`",
        ]
        if with_predictions:
            cells.append(PREDICTIONS[r["id"]][0])
        cells += [
            f"**{points(r['recall@10_diff'])}** [{points(lo)}, {points(hi)}]",
            fmt_p(r["recall@10_p_raw"]),
            fmt_p(r["recall@10_p_holm"]),
            f"{r['gained']} / {r['lost']}",
            "**supported**" if r["supported"] else "**not supported**",
        ]
        if with_predictions:
            cells.append(prediction_outcome(r["id"], r))
        out.append(cells)
    header = ["", "Claim", "Comparison"]
    align = ["---", "---", "---"]
    if with_predictions:
        header.append("Predicted (points)")
        align.append("---")
    header += ["Result in points [95 % CI]", "p", "Holm p", "Questions won / lost", "Verdict"]
    align += ["---", "---:", "---:", "---:", "---"]
    if with_predictions:
        header.append("Prediction")
        align.append("---")
    return _table(header, align, out)


def dev_table(reports: dict[str, dict]) -> str:
    order = sorted(reports, key=lambda n: reports[n]["pools"][DROPPED]["summary"]["recall@10"])
    systems = {n: {DROPPED: reports[n]["pools"][DROPPED]["summary"], SHIPPED: reports[n]["pools"][SHIPPED]["summary"]} for n in order}
    return systems_table(systems, order)


def splitter_table(evaluation: dict) -> str:
    names = {"lexicon": "Own splitter (`LexiconSplitter`)", "hanta": "HanTa (`HantaSplitter`)"}
    rows = []
    for key, scores in evaluation["splitters"].items():
        s = scores["all"]
        rows.append(
            [
                names.get(key, key),
                f"{s['precision']:.2f}",
                f"{s['recall']:.2f}",
                f"{s['f1']:.2f}",
                str(s["correct_split"]),
                str(s["wrong_cuts"]),
                str(s["missed"]),
                str(s["over_split"]),
                str(s["correct_whole"]),
            ]
        )
    header = ["Splitter", "Precision", "Recall", "F1", "Split correctly", "Wrong cuts", "Compound missed", "Split a non-compound", "Left whole correctly"]
    return _table(header, ["---"] + ["---:"] * 8, rows)


def build_tables(results: Path) -> dict[str, str]:
    """Every generated block, keyed by its marker name."""
    summary = json.loads((results / "test_summary.json").read_text(encoding="utf-8"))
    dev = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((results / "dev").glob("*.json"))}
    tables = {
        "test-systems": systems_table(summary["systems"], [n for n in TEST_ORDER if n in summary["systems"]]),
        "test-hypotheses": hypotheses_table(summary["hypotheses"][DROPPED]),
        "test-hypotheses-as-shipped": hypotheses_table(summary["hypotheses"][SHIPPED], with_predictions=False),
        "dev-systems": dev_table(dev),
    }
    evaluation = load_splitter_eval(results)  # a person's labels if there are any, else the AI labels
    if evaluation:
        tables["splitter"] = splitter_table(evaluation[0])
    return tables


def _block(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!-- BEGIN GENERATED: {re.escape(name)} -->\n)(?:.*?\n)?(<!-- END GENERATED: {re.escape(name)} -->)", flags=re.DOTALL
    )


def render(readme: str, tables: dict[str, str]) -> str:
    """The README with every marked block replaced. A table without its markers is an error."""
    for name, table in tables.items():
        pattern = _block(name)
        if not pattern.search(readme):
            raise ValueError(f"README has no markers for the table {name!r}")
        readme = pattern.sub(lambda m, t=table: m.group(1) + t + "\n" + m.group(2), readme)
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if the README is out of date")
    args = parser.parse_args()

    current = args.readme.read_text(encoding="utf-8")
    updated = render(current, build_tables(args.results))
    if args.check:
        if updated != current:
            print(f"{args.readme} is out of date: run python -m compound_search.tables", file=sys.stderr)
            raise SystemExit(1)
        print(f"{args.readme}: tables match {args.results}/")
        return
    args.readme.write_text(updated, encoding="utf-8")
    print(f"wrote {len(build_tables(args.results))} tables to {args.readme}")


if __name__ == "__main__":
    main()
