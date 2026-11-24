"""Every example and count that README.md and docs/RESULTS.md quote is looked up in the saved results.

The first write-up of this study contained examples that were not in the data. These tests
fail if the README names a rank, a count or a word that the saved reports do not contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compound_search.runs import DROPPED, SHIPPED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
README = ROOT / "README.md"
DETAILS = ROOT / "docs" / "RESULTS.md"
pytestmark = pytest.mark.skipif(not (RESULTS / "error_analysis_50.json").exists(), reason="results/ not present")


@pytest.fixture(scope="module")
def text() -> str:
    return README.read_text(encoding="utf-8") + DETAILS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def analysis() -> dict:
    return json.loads((RESULTS / "error_analysis_50.json").read_text(encoding="utf-8"))


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def record(analysis: dict, question: str) -> dict:
    (found,) = [q for q in analysis["queries"] if q["question"].strip() == question]
    return found


def said(rank: int | None) -> str:
    return "not in the top 100" if rank is None else str(rank)


# (question, sentence in the README with the ranks left open, systems whose ranks fill it)
RANK_EXAMPLES = [
    (
        "Wie hoch ist die Einwohnerdichte in Melbourne?",
        "rank {} with plain words, {} with stems, {} with split compounds",
        ("bm25_plain", "bm25_stem", "bm25_parts_stem"),
    ),
    (
        "Wer hat den obersten Befehl über die Kanadische Armee?",
        "from {} to rank {}",
        ("bm25_stem", "bm25_parts_stem"),
    ),
    (
        "Welches Unternehmen betreibt die Straßenbahn in Straßburg?",
        'Straßenbahn in Straßburg?" falls from rank {} to {}',
        ("bm25_stem", "bm25_parts_stem"),
    ),
    ("Wer ist Gabe Newell?", '"Wer ist Gabe Newell?" falls from rank {} to {}', ("bm25_stem", "bm25_parts_stem")),
    (
        "Wer leitete das erste Verfassungskommitee im Osmanischen Reich?",
        "*Verfassungskomitee* (rank {} to {})",
        ("hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"),
    ),
    (
        "Wie groß ist North Carolina?",
        "gains the word *Fläche* ({} to {})",
        ("hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"),
    ),
    (
        "Welcher Fluss fließt durch Philadelphia?",
        "it adds *Delaware River* ({} to {})",
        ("hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"),
    ),
    (
        "Was versteht man unter NES?",
        "the rank still improves from {} to {}",
        ("hybrid_parts_stem_dense", "hybrid_parts_stem_dense_rewrite"),
    ),
]


@pytest.mark.parametrize("question, sentence, systems", RANK_EXAMPLES, ids=[e[0][:30] for e in RANK_EXAMPLES])
def test_quoted_ranks_are_the_saved_ranks(text, analysis, question, sentence, systems):
    ranks = record(analysis, question)["ranks"]
    assert sentence.format(*(said(ranks[s]) for s in systems)) in text


def test_quoted_ranks_agree_with_the_system_reports(analysis):
    """The error analysis copies ranks from results/dev; make sure the copy is current."""
    for item in analysis["queries"]:
        for system, rank in item["ranks"].items():
            assert load(f"dev/{system}.json")["pools"][DROPPED]["gold_rank"][item["qid"]] == rank


@pytest.mark.parametrize(
    "question, word, parts",
    [
        ("Wie hoch ist die Einwohnerdichte in Melbourne?", "einwohnerdichte", ["einwohner", "dichte"]),
        ("Welches Unternehmen betreibt die Straßenbahn in Straßburg?", "straßburg", ["straße", "burg"]),
        ("Wer ist Gabe Newell?", "newell", ["new", "ell"]),
    ],
)
def test_quoted_cuts_are_what_the_splitter_did(text, analysis, question, word, parts):
    cuts = {c["word"]: c["parts"] for c in record(analysis, question)["query_compound_splits"]}
    assert cuts[word] == parts
    assert word in text.lower()


@pytest.mark.parametrize(
    "question, added",
    [
        ("Wer leitete das erste Verfassungskommitee im Osmanischen Reich?", "Verfassungskomitee"),
        ("Wie groß ist North Carolina?", "Fläche"),
        ("Welcher Fluss fließt durch Philadelphia?", "Delaware River"),
        ("Was versteht man unter NES?", "Netzwerk-Energie-System"),
    ],
)
def test_quoted_llm_answers_are_the_stored_answers(text, analysis, question, added):
    assert added in record(analysis, question)["llm_expansion"]
    assert added in text


def test_category_counts_in_the_prose(text, analysis):
    total = {name: info["total_matching_in_dev"] for name, info in analysis["categories"].items()}
    facts = analysis["facts_over_all_matching_queries"]
    wins, losses, h3 = facts["splitting_wins"], facts["splitting_losses"], facts["hybrid_preprocessing_wins_h3"]
    for phrase in (
        f"**Splitting wins ({total['splitting_wins']} questions).**",
        f"In {wins['gains_shared_term']} of them, splitting gives the question and its gold passage an index term they did not share before",
        f"In {wins['question_has_cut_word']} the question itself contains a word that is cut",
        f"**Splitting losses ({total['splitting_losses']} questions).**",
        f"In {losses['question_has_cut_word']} of them the question contains a word that the splitter cuts",
        f"In the other {losses['n'] - losses['question_has_cut_word']} questions no word of the question is cut",
        f"**Why preprocessing still matters in the hybrid ({total['hybrid_preprocessing_wins_h3']} questions).**",
        f"In {h3['plain_bm25_outside_top100']} of these {h3['n']} questions plain BM25 does not have the gold passage in its first 100",
        f"BM25 ranks the gold passage higher in all {h3['best_lexical_ranks_higher_than_plain']}",
        f"**LLM rewriting wins ({total['llm_rewrite_wins_h5']} questions).**",
        f"**Missed by everything ({total['residual_hard_failures']} questions).**",
    ):
        assert phrase in text, phrase
    assert h3["best_lexical_ranks_higher_than_plain"] == h3["n"]


def test_names_listed_as_cut_are_cut_in_a_lost_question(text, analysis):
    cut = set(analysis["splitting_losses_cut_words"])
    assert {"straßburg", "newell", "deutschland", "estlands", "ostpreußen"} <= cut
    assert "*Deutschland*, *Estlands* and *Ostpreußen* are cut too" in text


def test_residual_counts(text):
    residual = load("residual_check.json")
    rest = residual["n_residual"] - residual["n_residual_with_answer_string_in_final_top10"]
    assert f"In {residual['n_residual_with_answer_string_in_final_top10']} of them another passage in the top ten contains the gold answer string" in text
    assert f"The other {rest} are real misses" in text


def test_routing_shares(text):
    routing = load("test/hybrid_parts_stem_dense_rewrite.json")["routing"]
    head, shipped = routing[DROPPED], routing[SHIPPED]
    either = len(set(head["qids"]) | set(shipped["qids"]))
    assert (
        f"{head['share'] * 100:.1f}% of the test questions ({head['n_routed']} of 1,025, headline pool; "
        f"{shipped['n_routed']} with the pool as shipped; {either} under at least one of the two)"
    ) in text


def test_splitter_speed(text):
    speed = load("splitter_speed.json")
    assert (
        f"splitting the pool's {speed['n_distinct_capitalised_words']:,} distinct capitalised words took my splitter "
        f"{speed['lexicon']['total_seconds']:.0f} seconds, including building its word list, and HanTa {speed['hanta']['total_seconds']:.0f} seconds"
    ) in text


def cut(word: str, offsets: list[int]) -> str:
    pieces, prev = [], 0
    for c in offsets:
        pieces.append(word[prev:c])
        prev = c
    return "|".join([*pieces, word[prev:]])


def test_splitter_score_sentences(text):
    """The splitter section reports the labels of a person; the AI labels are a comparison."""
    scores = load("splitter_eval.json")["splitters"]
    own, hanta = scores["lexicon"], scores["hanta"]
    assert f"F1 {own['all']['f1']:.2f} against {hanta['all']['f1']:.2f}" in text
    cut_words = {name: s["all"]["correct_split"] + s["all"]["wrong_cuts"] + s["all"]["missed"] for name, s in scores.items()}
    assert cut_words["lexicon"] == cut_words["hanta"]
    assert own["all"]["correct_split"] == hanta["all"]["correct_split"]
    assert f"Both get {own['all']['correct_split']} of the {cut_words['lexicon']} words that I cut exactly right" in text

    def over(s: dict) -> list[str]:
        return [cut(w, x["predicted"]) for w, x in s["per_word"].items() if not x["gold"] and x["predicted"]]

    def missed(s: dict) -> set[str]:
        return {cut(w, x["gold"]) for w, x in s["per_word"].items() if x["gold"] and not x["predicted"]}

    assert over(hanta) == ["Sound|track"]
    assert "HanTa split one word that I left whole (*Sound|track*)" in text
    assert all(x["type"] == "name" for x in own["per_word"].values() if not x["gold"] and x["predicted"])
    assert "my splitter split five (" + ", ".join(f"*{w}*" for w in over(own)) + ")" in text
    assert "four of them names" in text  # Soundtrack carries the label "name" but is a loanword
    assert missed(own) and missed(hanta)


def test_the_label_revision_is_stated_with_both_scores(text):
    """The labels were revised once after the scores were known. The README must say so and give both scores."""
    import csv

    def rows(name: str) -> dict[str, dict]:
        with (ROOT / "labels" / name).open(encoding="utf-8-sig", newline="") as handle:
            return {r["word"]: r for r in csv.DictReader(handle)}

    first, revised = rows("compounds_labels_first_pass.csv"), rows("compounds_labels.csv")
    changed = [w for w in first if (first[w]["split"], first[w]["type"]) != (revised[w]["split"], revised[w]["type"])]
    assert len(changed) == 8
    assert all(revised[w]["note"].startswith("revised 2026-11-12 after the scores were known") for w in changed)
    assert all(not revised[w]["note"] for w in revised if w not in changed)
    assert "in eight words" in text and "after the scores were known" in text
    for word, cut_before in (("Verschiebung", "Ver+schiebung"), ("Judentum", "Juden+tum"), ("Wissenschaften", "Wissen+schaften"), ("Preußische", "Preußisch+e")):
        assert first[word]["split"] == cut_before and "+" not in revised[word]["split"]
        assert f"*{cut_before.replace('+', '|')}*" in text
    before = load("splitter_eval_first_pass.json")["splitters"]
    assert f"on it the scores are F1 {before['lexicon']['all']['f1']:.2f} and {before['hanta']['all']['f1']:.2f}" in text


def test_comparison_with_the_ai_labels(text):
    agreement, before = load("label_agreement.json"), load("label_agreement_first_pass.json")
    assert (
        f"agrees with the revised labels on the cuts of {agreement['same_cuts']} of the {agreement['n']} words "
        f"({before['same_cuts']} with the first pass)"
    ) in text
    ai = load("splitter_eval_ai_labels.json")["splitters"]
    assert {f"{s['all']['f1']:.2f}" for s in ai.values()} == {"0.86"}
    assert "Scored against the AI labels, both splitters reach F1 0.86" in text


def test_secondary_metric_sentence(text):
    hybrid = load("test/hybrid_parts_stem_dense.json")["pools"][SHIPPED]
    dense = load("test/dense.json")["pools"][SHIPPED]["summary"]["mrr@10"]
    p = hybrid["versus"]["dense"]["rr@10"]["p"]
    assert f"({dense:.3f} vs. {hybrid['summary']['mrr@10']:.3f}, p = {p:.3f})" in text


def test_the_chart_is_in_the_readme_and_current(text):
    from compound_search.figure import render_svg

    assert "](docs/recall_at_10.svg)" in text
    assert (ROOT / "docs" / "recall_at_10.svg").read_text(encoding="utf-8") == render_svg(load("test_summary.json"))


def test_every_quoted_rank_in_the_short_readme_is_also_in_the_details():
    """The README repeats three examples of docs/RESULTS.md; a rank may not differ between the two."""
    import re

    short, details = README.read_text(encoding="utf-8"), DETAILS.read_text(encoding="utf-8")
    quoted = re.findall(r"rank \d+ (?:to|with)[^.;]*?\d+", short)
    assert quoted
    for ranks in quoted:
        numbers = re.findall(r"\d+", ranks)
        assert any(re.findall(r"\d+", d)[: len(numbers)] == numbers for d in re.findall(r"rank \d+ (?:to|with)[^.;]*", details)), ranks
