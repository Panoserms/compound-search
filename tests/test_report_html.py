"""The HTML report is generated from results/. These tests keep it honest and self-contained."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from compound_search.report_html import build_report, hypothesis_chart, points_axis
from compound_search.runs import DROPPED, SHIPPED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
pytestmark = pytest.mark.skipif(not (RESULTS / "test_summary.json").exists(), reason="results/ not present")


@pytest.fixture(scope="module")
def html() -> str:
    return build_report(RESULTS, ROOT / "labels")


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((RESULTS / "test_summary.json").read_text(encoding="utf-8"))


def text_of(html: str) -> str:
    body = re.sub(r"<style.*?</style>", " ", html, flags=re.S)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))


def test_axis_maps_points_linearly():
    x = points_axis(lo=-5.0, hi=15.0, x0=100.0, x1=500.0)
    assert x(-5.0) == 100.0 and x(15.0) == 500.0 and x(5.0) == 300.0


def test_hypothesis_chart_has_one_row_per_test_and_marks_the_verdict(summary):
    svg = hypothesis_chart(summary["hypotheses"][DROPPED])
    assert svg.count('class="hy-dot') == 6
    assert svg.count('class="hy-dot supported"') == sum(r["supported"] for r in summary["hypotheses"][DROPPED])
    assert svg.count("<title>") >= 6  # a hover text on every mark


def test_report_is_one_self_contained_file(html):
    assert html.startswith("<!doctype html>")
    assert not re.search(r"""(src|href)=["']https?://""", html)
    assert "<script" not in html
    assert "{{" not in html and "}}" not in html


def test_every_chart_has_its_table(html):
    figures = re.findall(r"<figure.*?</figure>", html, flags=re.S)
    assert len(figures) >= 2
    for figure in figures:
        assert "<svg" in figure and "<table" in figure


def test_headline_numbers_come_from_the_results(html, summary):
    text = text_of(html)
    for name in ("bm25_plain", "bm25_stem", "bm25_parts_stem", "dense", "hybrid_plain_dense", "hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"):
        assert f"{summary['systems'][name][DROPPED]['recall@10'] * 100:.1f}%" in text, name
    assert f"{summary['n_queries']:,} test questions" in text
    assert f"{summary['n_gold_passages']} gold passages" in text


def test_failed_hypotheses_are_shown_as_failed(html, summary):
    text = text_of(html)
    rows = summary["hypotheses"][DROPPED]
    assert text.count("not supported") >= sum(not r["supported"] for r in rows)
    shipped_h3 = next(r for r in summary["hypotheses"][SHIPPED] if r["id"] == "H3")
    assert not shipped_h3["supported"]
    assert "does not pass" in text


def test_label_revision_is_disclosed_with_all_three_scores(html):
    text = text_of(html)
    assert "revised once, after the scores were known" in text
    for name in ("splitter_eval_first_pass.json", "splitter_eval.json", "splitter_eval_ai_labels.json"):
        scores = json.loads((RESULTS / name).read_text(encoding="utf-8"))["splitters"]
        for splitter in ("lexicon", "hanta"):
            assert f"{scores[splitter]['all']['f1']:.2f}" in text


def test_routing_share_and_reproduction_are_reported(html):
    routing = json.loads((RESULTS / "test" / "hybrid_parts_stem_dense_rewrite.json").read_text(encoding="utf-8"))["routing"][DROPPED]
    text = text_of(html)
    assert f"{routing['share'] * 100:.1f}% of the test questions" in text
    assert "identical" in text


def test_saved_report_is_current(html):
    saved = ROOT / "docs" / "report.html"
    if not saved.exists():
        pytest.skip("docs/report.html is built locally: python -m compound_search.report_html")
    assert saved.read_text(encoding="utf-8") == html, "run: python -m compound_search.report_html"
