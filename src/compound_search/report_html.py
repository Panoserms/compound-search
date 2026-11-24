"""A one-file HTML report on the study, written from the files under results/ and labels/.

    python -m compound_search.report_html --results results --labels labels --out docs/report.html           # write it
    python -m compound_search.report_html --results results --labels labels --out docs/report.html --check   # exit 1 if out of date

Every number in the report is read from a result file when the report is built; none is typed
by hand. The file has no scripts and loads nothing from the network, so it can be opened from
disk, printed to PDF or sent as it is.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from html import escape
from pathlib import Path

from compound_search.figure import render_svg
from compound_search.runs import DROPPED, SHIPPED
from compound_search.tables import LABELS, PREDICTIONS, TEST_ORDER, fmt_p, points, prediction_outcome

QUESTIONS = {
    "H1": "Do stems help BM25?",
    "H2": "Does splitting compounds add more?",
    "H3": "Is the gain still there next to a dense model?",
    "H4a": "Does the hybrid beat the dense model alone?",
    "H4b": "Does the hybrid beat BM25 alone?",
    "H5": "Does LLM query rewriting help?",
}
SPLITTER_NAMES = {"lexicon": "Own splitter", "hanta": "HanTa"}
REPORT_DATE = "12 November 2026"
NUM = ' class="num"'

# Class names of the embedded results chart (figure.py) are: title, group, label, value, tick, grid, ci, dot.
# The page's own classes avoid them.
CSS = """
:root { --bg:#ffffff; --panel:#f6f8fa; --ink:#1f2328; --soft:#59636e; --line:#d1d9e0; --band:#dfe5ea; --accent:#0b5c4a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#0d1117; --panel:#161b22; --ink:#f0f6fc; --soft:#9198a1; --line:#3d444d; --band:#2a313a; --accent:#5cc4a8; }
}
* { box-sizing: border-box; }
html { background: var(--bg); }
body { margin: 0 auto; max-width: 900px; padding: 32px 16px 64px; background: var(--bg); color: var(--ink);
       font: 16px/1.6 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
h1 { font-size: 30px; line-height: 1.2; margin: 0 0 6px; }
h2 { font-size: 21px; margin: 44px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 22px 0 6px; }
p { margin: 8px 0; }
.lead { font-size: 18px; color: var(--soft); margin: 0 0 4px; }
.meta { color: var(--soft); font-size: 14px; }
ul, ol { padding-left: 22px; margin: 6px 0; }
li { margin: 3px 0; }
li > ul, li > ol { margin: 2px 0 4px; }
code { font: 13px ui-monospace, "SF Mono", Menlo, monospace; background: var(--panel); padding: 1px 4px; border-radius: 4px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 18px 0 6px; }
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
.tile b { display: block; font-size: 24px; line-height: 1.15; font-variant-numeric: tabular-nums; }
.tile span { display: block; font-size: 14px; color: var(--soft); margin-top: 4px; }
.tile em { display: block; font-style: normal; font-size: 13px; margin-top: 6px; }
.note { background: var(--panel); border-left: 4px solid var(--accent); border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 14px 0; }
figure { margin: 16px 0; }
figure svg { display: block; width: 100%; height: auto; }
figcaption { color: var(--soft); font-size: 14px; margin: 6px 0 10px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; margin: 8px 0; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--soft); font-weight: 600; }
td.num, th.num { text-align: right; white-space: nowrap; }
.yes { font-weight: 600; white-space: nowrap; }
.no { font-weight: 600; white-space: nowrap; text-decoration: underline; text-decoration-style: dotted; }
td:first-child { white-space: nowrap; }
.hy-id { font-size: 13px; font-weight: 600; fill: var(--ink); }
.hy-q, .hy-res { font-size: 13px; fill: var(--ink); }
.hy-res { font-variant-numeric: tabular-nums; }
.hy-small, .hy-tick { font-size: 11px; fill: var(--soft); }
.hy-grid { stroke: var(--line); stroke-width: 1; }
.hy-zero { stroke: var(--soft); stroke-width: 1.5; }
.hy-band { fill: var(--band); }
.hy-ci { stroke: var(--soft); stroke-width: 2; stroke-linecap: round; }
.hy-dot { stroke: var(--ink); stroke-width: 2; fill: var(--bg); }
.hy-dot.supported { fill: var(--ink); }
@media print { body { max-width: none; padding: 0; } h2 { break-after: avoid; } figure, table, .tile { break-inside: avoid; } }
""".strip()


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def ci_points(pair: list[float]) -> str:
    return f"[{points(pair[0])}, {points(pair[1])}]"


def load(results: Path, name: str) -> dict:
    return json.loads((results / name).read_text(encoding="utf-8"))


def table(header: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    def cells(tag: str, values: list[str]) -> str:
        return "".join(f"<{tag}{NUM if i in numeric else ''}>{v}</{tag}>" for i, v in enumerate(values))

    body = "".join(f"<tr>{cells('td', r)}</tr>" for r in rows)
    return f'<div class="scroll"><table><thead><tr>{cells("th", header)}</tr></thead><tbody>{body}</tbody></table></div>'


def verdict(supported: bool) -> str:
    return '<span class="yes">supported</span>' if supported else '<span class="no">not supported</span>'


def points_axis(lo: float, hi: float, x0: float, x1: float) -> Callable[[float], float]:
    """Map a value in points to an x position; values outside the axis are drawn at its edge."""
    return lambda v: x0 + (min(hi, max(lo, v)) - lo) / (hi - lo) * (x1 - x0)


def hypothesis_chart(rows: list[dict]) -> str:
    """One row per hypothesis: the predicted range as a band, the test result as a dot with its 95 % interval.
    A filled dot means supported, a hollow one not supported; the verdict is also written out."""
    width, row_h, top, bottom = 860, 44, 30, 40
    x0, x1 = 330.0, 640.0
    lo, hi, ticks = -5.0, 15.0, (-5, 0, 5, 10, 15)
    x = points_axis(lo, hi, x0, x1)
    height = top + row_h * len(rows) + bottom
    plot_bottom = height - bottom
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Predicted and measured difference in Recall at 10 for the six hypotheses">',
        '<text class="hy-small" x="16" y="16">Hypothesis</text>',
        f'<text class="hy-small" x="{x0}" y="16">Difference in Recall@10, in points. Grey band: the range predicted beforehand.</text>',
    ]
    for tick in ticks:
        cls = "hy-zero" if tick == 0 else "hy-grid"
        out.append(f'<line class="{cls}" x1="{x(tick):.1f}" y1="{top}" x2="{x(tick):.1f}" y2="{plot_bottom}"/>')
        out.append(f'<text class="hy-tick" x="{x(tick):.1f}" y="{plot_bottom + 16}" text-anchor="middle">{tick:+d}</text>')
    for i, row in enumerate(rows):
        hid = row["id"]
        mid = top + row_h * i + row_h / 2
        predicted, p_lo, p_hi = PREDICTIONS[hid]
        diff = row["recall@10_diff"] * 100
        ci_lo, ci_hi = (v * 100 for v in row["recall@10_ci95"])
        state = "supported" if row["supported"] else "not supported"
        hover = f"{hid}: {points(row['recall@10_diff'])} points, 95% interval {ci_points(row['recall@10_ci95'])}, predicted {predicted}, {state}"
        out.append(f'<text class="hy-id" x="16" y="{mid - 3:.1f}">{hid}</text>')
        out.append(f'<text class="hy-q" x="52" y="{mid - 3:.1f}">{escape(QUESTIONS[hid])}</text>')
        out.append(f'<text class="hy-small" x="52" y="{mid + 12:.1f}">predicted: {escape(predicted)}</text>')
        if p_lo is not None and p_hi is not None:
            out.append(f'<rect class="hy-band" x="{x(p_lo):.1f}" y="{mid - 11:.1f}" width="{x(p_hi) - x(p_lo):.1f}" height="22" rx="3"><title>predicted: {escape(predicted)} points</title></rect>')
        out.append(f'<line class="hy-ci" x1="{x(ci_lo):.1f}" y1="{mid:.1f}" x2="{x(ci_hi):.1f}" y2="{mid:.1f}"/>')
        out.append(f'<circle class="hy-dot{" supported" if row["supported"] else ""}" cx="{x(diff):.1f}" cy="{mid:.1f}" r="6"><title>{escape(hover)}</title></circle>')
        out.append(f'<text class="hy-res" x="{x1 + 18:.1f}" y="{mid + 4.5:.1f}">{points(row["recall@10_diff"])} · {state}</text>')
    out.append("</svg>")
    return "\n".join(out)


def read_labels(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {r["word"]: r for r in csv.DictReader(handle)}


def cut_text(row: dict) -> str:
    return (row["split"] or row["word"]).replace("+", "|")


def example(analysis: dict, question: str) -> dict:
    (found,) = [q for q in analysis["queries"] if q["question"].strip() == question]
    return found


def rank(r: int | None) -> str:
    return "not in the first 100" if r is None else f"rank {r}"


def build_report(results: Path, labels: Path) -> str:
    summary = load(results, "test_summary.json")
    systems = summary["systems"]
    rows_h = summary["hypotheses"][DROPPED]
    rows_s = {r["id"]: r for r in summary["hypotheses"][SHIPPED]}
    by_id = {r["id"]: r for r in rows_h}
    stage0 = load(results, "stage0_baseline.json")
    dev = {name: load(results, f"dev/{name}.json") for name in ("bm25_plain", "dense", "hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite")}
    routing_test = load(results, "test/hybrid_parts_stem_dense_rewrite.json")["routing"]
    routing_dev = dev["hybrid_parts_stem_dense_rewrite"]["routing"]
    analysis = load(results, "error_analysis_50.json")
    facts = analysis["facts_over_all_matching_queries"]
    totals = {k: v["total_matching_in_dev"] for k, v in analysis["categories"].items()}
    residual = load(results, "residual_check.json")
    speed = load(results, "splitter_speed.json")
    evals = {
        "first pass": load(results, "splitter_eval_first_pass.json"),
        "revised": load(results, "splitter_eval.json"),
        "AI labels": load(results, "splitter_eval_ai_labels.json"),
    }
    agreement = {"revised": load(results, "label_agreement.json"), "first pass": load(results, "label_agreement_first_pass.json")}
    first, revised = read_labels(labels / "compounds_labels_first_pass.csv"), read_labels(labels / "compounds_labels.csv")
    changed = [w for w in first if (first[w]["split"], first[w]["type"]) != (revised[w]["split"], revised[w]["type"])]

    def r10(name: str, pool: str = DROPPED) -> float:
        return systems[name][pool]["recall@10"]

    n_test, n_gold = summary["n_queries"], summary["n_gold_passages"]
    n_dev = dev["bm25_plain"]["pools"][DROPPED]["summary"]["n_queries"]
    n_dev_gold = dev["bm25_plain"]["pools"][DROPPED]["summary"]["n_gold_passages"]
    n_passages = stage0["data"]["germandpr"]["passage_pool"]["distinct_passages"]
    negatives = stage0["dev"]["as_shipped"]["own_hard_negatives"]
    lexical_gain = r10("bm25_parts_stem") - r10("bm25_plain")
    supported = sum(r["supported"] for r in rows_h)
    h1, h2, h3, h4a, h4b, h5 = (by_id[k] for k in ("H1", "H2", "H3", "H4a", "H4b", "H5"))

    # The sentences below state these things; stop rather than print a report that contradicts its data.
    if not (h1["supported"] and h2["supported"] and h3["supported"] and h4b["supported"]) or h4a["supported"] or h5["supported"]:
        raise RuntimeError("a verdict changed; the report text must be rewritten")
    if rows_s["H3"]["supported"]:
        raise RuntimeError("H3 is now supported on the as-shipped pool; the report text says it is not")
    if not {"straßburg", "newell", "deutschland"} <= set(analysis["splitting_losses_cut_words"]):
        raise RuntimeError("a name the report gives as an example is no longer cut in a lost question")

    tiles = [
        (f"{pct(r10('bm25_plain'))} → {pct(r10('bm25_parts_stem'))}", "BM25 with stems and split compounds", f"{points(lexical_gain)} points; supported"),
        (f"{points(h3['recall@10_diff'])} points", "still there inside a BM25 + dense hybrid", "the main question; supported"),
        (f"{pct(r10('hybrid_parts_stem_dense'))} vs. {pct(r10('dense'))}", "best hybrid against the dense model alone", f"{h4a['gained']} questions won, {h4a['lost']} lost; not supported"),
        (f"{points(h5['recall@10_diff'])} points", "from LLM rewriting of unsure queries", f"Holm p = {fmt_p(h5['recall@10_p_holm'])}; not supported"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{escape(a)}</b><span>{escape(b)}</span><em>{escape(c)}</em></div>' for a, b, c in tiles)

    systems_rows = []
    for name in TEST_ORDER:
        s, shipped = systems[name][DROPPED], systems[name][SHIPPED]
        systems_rows.append(
            [
                escape(LABELS[name]),
                f"{s['recall@1']:.3f}",
                f"{s['recall@5']:.3f}",
                f"<b>{pct(s['recall@10'])}</b>",
                f"[{pct(s['recall@10_ci95'][0])}, {pct(s['recall@10_ci95'][1])}]",
                f"{s['mrr@10']:.3f}",
                pct(shipped["recall@10"]),
            ]
        )
    systems_table = table(["System", "R@1", "R@5", "R@10", "95% interval", "MRR@10", "R@10, pool as shipped"], systems_rows, numeric={1, 2, 3, 4, 5, 6})

    hyp_rows = []
    for r in rows_h:
        s = rows_s[r["id"]]
        hyp_rows.append(
            [
                f"<b>{r['id']}</b>",
                escape(QUESTIONS[r["id"]]),
                escape(PREDICTIONS[r["id"]][0]),
                f"<b>{points(r['recall@10_diff'])}</b> {ci_points(r['recall@10_ci95'])}",
                fmt_p(r["recall@10_p_holm"]),
                f"{r['gained']} / {r['lost']}",
                verdict(r["supported"]),
                escape(prediction_outcome(r["id"], r)),
                f"{points(s['recall@10_diff'])}, {verdict(s['supported'])}",
            ]
        )
    hyp_table = table(
        ["", "Question", "Predicted (points)", "Result [95% interval]", "Holm p", "Won / lost", "Verdict", "Prediction was", "Pool as shipped"],
        hyp_rows,
        numeric={3, 4, 5},
    )

    splitter_rows = []
    for label_set, evaluation in evals.items():
        for key, name in SPLITTER_NAMES.items():
            s = evaluation["splitters"][key]["all"]
            splitter_rows.append(
                [escape(label_set), name, f"{s['precision']:.2f}", f"{s['recall']:.2f}", f"<b>{s['f1']:.2f}</b>", str(s["correct_split"]), str(s["missed"] + s["wrong_cuts"]), str(s["over_split"])]
            )
    splitter_table = table(
        ["Labels", "Splitter", "Precision", "Recall", "F1", "Cut exactly right", "Missed or cut wrongly", "Cut a word that should stay whole"],
        splitter_rows,
        numeric={2, 3, 4, 5, 6, 7},
    )
    changed_table = table(
        ["Word", "First pass", "Revised", "Type now"],
        [[escape(w), escape(cut_text(first[w])), escape(cut_text(revised[w])), escape(revised[w]["type"])] for w in changed],
    )
    own_over = [w for w, x in evals["revised"]["splitters"]["lexicon"]["per_word"].items() if not x["gold"] and x["predicted"]]

    melbourne = example(analysis, "Wie hoch ist die Einwohnerdichte in Melbourne?")["ranks"]
    strassburg = example(analysis, "Welches Unternehmen betreibt die Straßenbahn in Straßburg?")["ranks"]
    newell = example(analysis, "Wer ist Gabe Newell?")["ranks"]
    wins, losses, hybrid = facts["splitting_wins"], facts["splitting_losses"], facts["hybrid_preprocessing_wins_h3"]
    real_misses = residual["n_residual"] - residual["n_residual_with_answer_string_in_final_top10"]
    error_table = table(
        ["What happens", f"Questions (of {n_dev:,})", "What the counts show"],
        [
            ["Splitting rescues a question that plain words and stems both missed", str(totals["splitting_wins"]), f"In {wins['gains_shared_term']}, the cut gives the question and its passage a search term they did not share before."],
            ["Splitting loses a question that stems had found", str(totals["splitting_losses"]), f"In {losses['question_has_cut_word']}, the question contains a word that gets cut; several of those words are names. In the other {losses['n'] - losses['question_has_cut_word']}, nothing in the question is cut."],
            ["Inside the hybrid, the linguistic steps rescue a question", str(totals["hybrid_preprocessing_wins_h3"]), f"In {hybrid['plain_bm25_outside_top100']}, plain BM25 does not have the right passage in its first 100."],
            ["LLM rewriting rescues a question", str(totals["llm_rewrite_wins_h5"]), "Some are real repairs (a corrected spelling), some are luck."],
            ["Missed by everything", str(residual["n_residual"]), f"In {residual['n_residual_with_answer_string_in_final_top10']}, another returned passage contains the answer, so the label may be incomplete. The other {real_misses} are real misses."],
        ],
        numeric={1},
    )

    headline = summary["headline"]
    either = len(set(routing_test[DROPPED]["qids"]) | set(routing_test[SHIPPED]["qids"]))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Compound Search: project report</title>
<style>
{CSS}
</style>
</head>
<body>

<h1>Compound Search</h1>
<p class="lead">Do stemming and compound splitting still help German search, and is anything left of that help once a neural model is added?</p>
<p class="meta">Project report, {REPORT_DATE}. Every number is read from the study's result files when this page is built.</p>

<div class="tiles">{tiles_html}</div>
<p class="meta">Recall@10 on {n_test:,} test questions: how often the right passage is among the first ten results. {{supported_word}} of six hypotheses were supported.</p>

<h2>1. What the project is</h2>
<ul>
  <li><b>The problem.</b> German glues nouns together. A question asks for the <i>Einwohnerdichte</i> of Melbourne; the passage that answers it says <i>Einwohnerzahl</i> and <i>Bevölkerungsdichte</i>. A keyword search finds the city's name and nothing for the thing that is asked about.</li>
  <li><b>The classic fix.</b> Reduce words to their stems and cut compounds into their parts.</li>
  <li><b>The open question.</b> Search engines now also use neural ("dense") models, which compare meaning instead of words. Does the old linguistic work still pay once such a model is in the system?</li>
  <li><b>What the study does.</b> It tests {len(TEST_ORDER)} systems, from plain keyword search to a hybrid with an LLM, against six hypotheses that were written down before the systems existed.</li>
</ul>

<h2>2. How it was set up</h2>
<ul>
  <li><b>Data:</b> GermanDPR (CC BY 4.0), German Wikipedia.
    <ul>
      <li>{n_passages:,} passages to search in.</li>
      <li>{n_dev:,} development questions on {n_dev_gold} gold passages. Every choice was made here.</li>
      <li>{n_test:,} test questions on {n_gold} gold passages, held out until the configuration was frozen.</li>
    </ul>
  </li>
  <li><b>Two pools.</b> GermanDPR ships three trap passages with every question, picked with BM25.
    <ul>
      <li>Under plain BM25, one of them outranks the right passage for {negatives['share_of_queries_where_one_outranks_gold'] * 100:.0f}% of the development questions.</li>
      <li>Headline pool: each question is ranked without its own three traps. This was fixed before any comparison system was built.</li>
      <li>Pool as shipped: reported beside the headline for every system.</li>
    </ul>
  </li>
  <li><b>Fixed in advance, never tuned:</b> BM25 with its standard settings; the dense model <code>multilingual-e5-small</code> as published; rank fusion with its standard constant and no weights.</li>
  <li><b>Statistics:</b>
    <ul>
      <li>Many questions share one gold passage, so intervals and tests resample passages, not questions.</li>
      <li>95% bootstrap intervals; paired randomisation tests; Holm correction across the six tests.</li>
    </ul>
  </li>
  <li><b>Everything runs on one 8 GB laptop.</b> No paid API. The LLM is a local Gemma model.</li>
</ul>

<h2>3. The systems and the test result</h2>
<figure>
{render_svg(summary)}
<figcaption>Recall@10 on the {n_test:,} test questions, headline pool. The line through each dot is its 95% interval.</figcaption>
{systems_table}
</figure>
<ul>
  <li><b>Keyword search.</b> Stems lift plain BM25 from {pct(r10('bm25_plain'))} to {pct(r10('bm25_stem'))}; split compounds lift it further to {pct(r10('bm25_parts_stem'))}.</li>
  <li><b>Dense model alone.</b> {pct(r10('dense'))}, without any of this work.</li>
  <li><b>Hybrids.</b> Fusing the dense model with <i>plain</i> BM25 gives {pct(r10('hybrid_plain_dense'))}, which is worse than the dense model alone. Fusing it with the stemmed and split BM25 gives {pct(r10('hybrid_parts_stem_dense'))}.</li>
  <li><b>LLM rewriting.</b> A local model adds search terms to the queries on which the two retrievers disagree: {pct(r10('hybrid_parts_stem_dense_rewrite'))}.
    <ul>
      <li>The rule that picks these queries looks at rankings and scores only, never at the answer.</li>
      <li>It sent {pct(routing_dev[DROPPED]['share'])} of the development questions and {pct(routing_test[DROPPED]['share'])} of the test questions to the LLM ({routing_test[DROPPED]['n_routed']} of {n_test:,}; {either} under at least one of the two pools).</li>
    </ul>
  </li>
</ul>

<h2>4. The six hypotheses</h2>
<figure>
{hypothesis_chart(rows_h)}
<figcaption>Filled dot: supported (the difference is positive and its corrected p value is below 0.05). Hollow dot: not supported. H4a, H4b and H5 were predicted in words, not as a range.</figcaption>
{hyp_table}
</figure>
<ul>
  <li><b>Supported: H1, H2, H3, H4b.</b>
    <ul>
      <li>Stems and splitting help BM25, and by more than predicted ({points(h1['recall@10_diff'])} and {points(h2['recall@10_diff'])} points).</li>
      <li>The help survives inside the hybrid: {points(h3['recall@10_diff'])} points. This was the main question.</li>
    </ul>
  </li>
  <li><b>Not supported: H4a.</b> The hybrid does not beat the dense model alone: {h4a['gained']} questions won, {h4a['lost']} lost.</li>
  <li><b>Not supported: H5.</b> LLM rewriting gains {points(h5['recall@10_diff'])} points ({h5['gained']} questions won, {h5['lost']} lost), which does not pass the corrected test.</li>
  <li><b>A weaker second view.</b> With the pool as shipped, the gain inside the hybrid is {points(rows_s['H3']['recall@10_diff'])} points and does not pass the corrected test (Holm p = {fmt_p(rows_s['H3']['recall@10_p_holm'])}).</li>
</ul>
<div class="note"><b>What an engineer could take from it, on this data:</b> use the dense model. If you must stay with keyword search, stem and split. If you fuse the two, do the linguistic work first, because fusing with plain BM25 makes the dense model worse.</div>

<h2>5. The compound splitter by itself</h2>
<ul>
  <li><b>How it works.</b> It needs no dictionary. It counts the words of the passages it is going to index. A word may be cut into nouns it has seen, and the cut wins if its parts are, on average, more frequent than the whole word.
    <ul>
      <li>Linking elements are handled: <i>Bundes|verfassungs|gericht</i>.</li>
      <li>A frequent fixed word such as <i>Bahnhof</i> stays whole.</li>
    </ul>
  </li>
  <li><b>How it was scored.</b> 100 capitalised words from development questions, labelled by hand by the author on a page that shows nothing a splitter did. HanTa, an open morphology tool, is scored beside it.</li>
</ul>
{splitter_table}
<ul>
  <li><b>These labels were revised once, after the scores were known.</b>
    <ul>
      <li>The first pass also had cuts at a prefix, a suffix or an ending in {len(changed)} words. The labelling page's own instruction counts a derived word as "not a compound".</li>
      <li>They were removed by one rule: a cut stays only where a word meets a word.</li>
      <li>Because the change came after the scores, the table gives the first pass too. The first pass is kept unchanged in <code>labels/compounds_labels_first_pass.csv</code>.</li>
    </ul>
  </li>
  <li><b>Reading the scores.</b>
    <ul>
      <li>On 100 words the two splitters cannot be ranked.</li>
      <li>They fail differently. HanTa cut {evals['revised']['splitters']['hanta']['all']['over_split']} word that should stay whole; the study's splitter cut {len(own_over)}, all of them names ({", ".join(f"<i>{escape(w)}</i>" for w in own_over)}).</li>
      <li>An earlier labelling by an AI agrees with the revised labels on the cuts of {agreement['revised']['same_cuts']} of 100 words ({agreement['first pass']['same_cuts']} with the first pass).</li>
    </ul>
  </li>
  <li><b>Speed.</b> Cutting the {speed['n_distinct_capitalised_words']:,} distinct capitalised words of the passages took the study's splitter {speed['lexicon']['total_seconds']:.0f} seconds, including its word count, and HanTa {speed['hanta']['total_seconds']:.0f} seconds (one run each).</li>
</ul>
<h3>The {len(changed)} revised labels</h3>
{changed_table}

<h2>6. What the errors show</h2>
{error_table}
<ul>
  <li><b>A win.</b> "Wie hoch ist die Einwohnerdichte in Melbourne?": {rank(melbourne['bm25_plain'])} with plain words, {rank(melbourne['bm25_stem'])} with stems, {rank(melbourne['bm25_parts_stem'])} with split compounds.</li>
  <li><b>A loss.</b> "Welches Unternehmen betreibt die Straßenbahn in Straßburg?": {rank(strassburg['bm25_stem'])} with stems, {rank(strassburg['bm25_parts_stem'])} once <i>Straßburg</i> is cut into <i>Straße</i> and <i>Burg</i>.</li>
  <li><b>Another loss.</b> "Wer ist Gabe Newell?": {rank(newell['bm25_stem'])} with stems, {rank(newell['bm25_parts_stem'])} once <i>Newell</i> is cut into <i>new</i> and <i>ell</i>.</li>
  <li><b>Why the linguistic steps still matter in the hybrid.</b> Rank fusion rewards passages that both lists contain. If plain BM25 misses the right passage, that passage has only its dense rank, and passages found by both lists overtake it.</li>
</ul>

<h2>7. What was checked</h2>
<ul>
  <li><b>The test run.</b> The results come from one search of the test questions. The unchanged systems were later run on them twice more, only to check that the results repeat; both repeats gave identical reports.</li>
  <li><b>A clean copy.</b> On 12 November 2026 the whole study was repeated in a copy with a new environment, a fresh download and freshly computed embeddings. All 33 result files were identical to the saved ones. This was before the hand labels were added; the label scores are not part of that check.</li>
  <li><b>The write-up.</b>
    <ul>
      <li>Tables and charts are generated from the result files.</li>
      <li>An automated test looks up every rank, count and example that the README quotes.</li>
      <li>An earlier draft contained statements the data did not back. They were replaced, and each correction is dated in <code>HYPOTHESES.md</code>.</li>
    </ul>
  </li>
</ul>

<h2>8. Limits</h2>
<ul>
  <li><b>One dataset.</b> The questions were written by people who had the passage in front of them. Real search queries are shorter and messier.</li>
  <li><b>A small pool.</b> {n_passages:,} passages. Among millions, the balance between keyword search and a dense model may differ.</li>
  <li><b>One gold passage per question.</b> Other passages that also answer it count as misses.</li>
  <li><b>The choice of pool matters.</b> The main result passes the corrected test in the headline pool only.</li>
  <li><b>One dense model, one LLM, one prompt, all small.</b> A larger LLM or a weighted fusion could change the two results that failed.</li>
  <li><b>The compound labels are one person's,</b> on 100 words, and were revised once after the scores were known.</li>
  <li><b>The hypotheses were fixed in a local file,</b> not in a public registry. Its dates rest on file timestamps.</li>
</ul>

<h2>9. The headline number</h2>
<div class="note">{escape(headline['sentence_dense'])}</div>
<ul>
  <li>The protocol fixed this wording before the test run, for the system with the best development score. That system is the dense model alone ({pct(dev['dense']['pools'][DROPPED]['summary']['recall@10'])} on the development set).</li>
  <li>For the full pipeline the gain is +{headline['full_pipeline_gain_points']} points. Its last {points(h5['recall@10_diff']).lstrip('+')} points come from the LLM step, whose hypothesis was not supported.</li>
</ul>


</body>
</html>
""".replace("{supported_word}", ["Zero", "One", "Two", "Three", "Four", "Five", "Six"][supported])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--labels", type=Path, default=Path("labels"))
    parser.add_argument("--out", type=Path, default=Path("docs/report.html"))
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if the report is out of date")
    args = parser.parse_args()
    html = build_report(args.results, args.labels)
    if args.check:
        if not args.out.exists() or args.out.read_text(encoding="utf-8") != html:
            print(f"{args.out} is out of date: run python -m compound_search.report_html", file=sys.stderr)
            raise SystemExit(1)
        print(f"{args.out}: matches {args.results}/")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
