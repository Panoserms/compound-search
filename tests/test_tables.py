"""The README tables are written from results/. These tests keep the two in step."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compound_search.runs import DROPPED
from compound_search.tables import PREDICTIONS, build_readme_tables, build_tables, fmt_p, points, prediction_outcome, render, splitter_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
DETAILS = ROOT / "docs" / "RESULTS.md"
needs_results = pytest.mark.skipif(not (RESULTS / "test_summary.json").exists(), reason="results/ not present")


def test_small_p_values_are_not_given_false_precision():
    assert fmt_p(0.0001) == "< 0.001"
    assert fmt_p(0.0006) == "< 0.001"
    assert fmt_p(0.0054) == "0.005"
    assert fmt_p(1.0) == "1.000"


def test_points_have_a_sign_and_no_negative_zero():
    assert points(0.0634) == "+6.3"
    assert points(-0.0146) == "−1.5"
    assert points(-0.0001) == "+0.0"


def row(diff: float, supported: bool) -> dict:
    return {"recall@10_diff": diff, "supported": supported}


def test_prediction_outcome_compares_with_the_range_written_beforehand():
    assert prediction_outcome("H1", row(0.063, True)) == "larger than predicted"
    assert prediction_outcome("H1", row(0.02, True)) == "as predicted"
    assert prediction_outcome("H1", row(0.004, False)) == "smaller than predicted"
    assert prediction_outcome("H4a", row(0.0, False)) == "no gain found"
    assert prediction_outcome("H4b", row(0.11, True)) == "as predicted"
    assert prediction_outcome("H5", row(0.008, False)) == "as predicted"
    assert prediction_outcome("H5", row(0.03, True)) == "larger than predicted"


def test_render_replaces_only_what_is_between_the_markers():
    text = "before\n<!-- BEGIN GENERATED: t -->\nold\nlines\n<!-- END GENERATED: t -->\nafter\n"
    out = render(text, {"t": "| a |"})
    assert out == "before\n<!-- BEGIN GENERATED: t -->\n| a |\n<!-- END GENERATED: t -->\nafter\n"
    assert render(out, {"t": "| a |"}) == out  # a second pass changes nothing
    empty = "<!-- BEGIN GENERATED: t -->\n<!-- END GENERATED: t -->\n"
    assert "| a |" in render(empty, {"t": "| a |"})


def test_render_refuses_a_table_without_markers():
    with pytest.raises(ValueError):
        render("no markers here", {"t": "| a |"})


@needs_results
def test_readme_tables_match_the_saved_results():
    for path, tables in ((README, build_readme_tables(RESULTS)), (DETAILS, build_tables(RESULTS))):
        current = path.read_text(encoding="utf-8")
        assert render(current, tables) == current, f"{path.name}: run python -m compound_search.tables"


@needs_results
def test_every_hypothesis_has_a_prediction():
    summary = json.loads((RESULTS / "test_summary.json").read_text(encoding="utf-8"))
    assert [r["id"] for r in summary["hypotheses"][DROPPED]] == list(PREDICTIONS)


@needs_results
def test_readme_prose_numbers_match_the_saved_results():
    """The sentences around the tables are typed by hand. Their key numbers are checked here."""
    summary = json.loads((RESULTS / "test_summary.json").read_text(encoding="utf-8"))
    text = README.read_text(encoding="utf-8") + DETAILS.read_text(encoding="utf-8")
    systems = summary["systems"]

    def r10(name: str) -> str:
        return f"{systems[name][DROPPED]['recall@10'] * 100:.1f}%"

    for name in ("bm25_plain", "bm25_parts_stem", "dense", "hybrid_plain_dense", "hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"):
        assert r10(name) in text, name

    by_id = {r["id"]: r for r in summary["hypotheses"][DROPPED]}
    for hid in ("H1", "H2", "H3", "H5"):
        assert f"{points(by_id[hid]['recall@10_diff'])}" in text, hid
    lexical_gain = systems["bm25_parts_stem"][DROPPED]["recall@10"] - systems["bm25_plain"][DROPPED]["recall@10"]
    assert f"{points(lexical_gain)} points" in text
    assert f"{by_id['H4a']['gained']} questions are won and {by_id['H4a']['lost']} lost" in text
    assert f"{by_id['H5']['gained']} questions won, {by_id['H5']['lost']} lost" in text
    assert summary["headline"]["sentence_dense"] in text
    supported = sum(r["supported"] for r in by_id.values())
    assert {4: "Four", 3: "Three", 5: "Five"}[supported] + " of six hypotheses were supported" in text

    dev_rewrite = json.loads((RESULTS / "dev" / "hybrid_parts_stem_dense_rewrite.json").read_text(encoding="utf-8"))
    routing = dev_rewrite["routing"][DROPPED]
    assert f"{routing['share'] * 100:.1f}% of the development questions" in text
    assert f"({routing['n_routed']} of 1,000, headline pool)" in text

    stage0 = json.loads((RESULTS / "stage0_baseline.json").read_text(encoding="utf-8"))
    negs = stage0["dev"]["as_shipped"]["own_hard_negatives"]
    assert f"for {negs['share_of_queries_where_one_outranks_gold'] * 100:.0f}% of the questions" in text
    assert f"top hit for {negs['share_of_queries_where_top1_is_one'] * 100:.0f}%" in text


@needs_results
def test_the_splitter_table_reports_a_persons_labels_when_they_exist():
    person = RESULTS / "splitter_eval.json"
    if not person.exists():
        pytest.skip("no labels by a person")
    assert build_tables(RESULTS)["splitter"] == splitter_table(json.loads(person.read_text(encoding="utf-8")))
