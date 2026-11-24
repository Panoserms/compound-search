"""The website export: its word-by-word terms must be what the real tokenisers produce."""

from __future__ import annotations

import json

import pytest

from compound_search.data import Passage
from compound_search.export_site import (
    CONFIGS,
    MODES,
    annotate,
    body_of,
    comparison,
    cuts_of,
    lucene_idf,
    pick_excerpt,
    real_tokenizer,
    segments,
    word_terms,
)
from compound_search.pipeline import HYBRIDS, LEXICAL
from compound_search.splitter import Lexicon, LexiconSplitter

SPLITTER = LexiconSplitter(
    Lexicon(
        freq={"tabak": 50, "konzern": 50, "konzerne": 20, "industrie": 80, "bund": 60, "gericht": 70},
        nouns={"tabak", "konzern", "konzerne", "industrie", "bund", "gericht"},
    )
)
TEXT = "Die Konzerne der Tabakindustrie wachsen; das Bundesgericht prüft den Tabakkonzern_Fall 2020."


@pytest.mark.parametrize("mode", MODES)
def test_word_terms_equal_the_real_tokeniser(mode):
    mine = [t for token, _ in segments(TEXT) for t in word_terms(token, mode, SPLITTER)]
    assert mine == real_tokenizer(mode, SPLITTER)(TEXT)


def test_every_switch_position_is_a_system_of_the_study():
    assert set(CONFIGS) == {f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"}
    for key, system in CONFIGS.items():
        assert system in LEXICAL or system in HYBRIDS
        assert (key[2] == "1") == system.startswith("hybrid_")  # third switch = dense model


def test_segments_keep_the_text_between_words():
    assert segments("Wo ist  Bonn?") == [("Wo", " "), ("ist", "  "), ("Bonn", "?")]
    assert "".join(w + gap for w, gap in segments(TEXT)) == TEXT


def test_cuts_only_for_capitalised_words():
    assert cuts_of("Tabakindustrie", SPLITTER) == [5]
    assert cuts_of("Bundesgericht", SPLITTER) == [6]  # the linking element stays with the left part
    assert cuts_of("tabakindustrie", SPLITTER) == []


def test_annotate_marks_the_parts_the_other_text_shares():
    idf = {mode: {} for mode in MODES}
    idf["parts_stem"] = {"tabak": 4.0}
    other = {mode: set() for mode in MODES}
    other["parts_stem"] = {"tabak"}
    words = {w["t"]: w for w in annotate(TEXT, other, idf, SPLITTER)}
    assert words["Tabakindustrie"]["m"]["parts_stem"] == [4.0, 0]  # "tabak" matches, "industri" does not
    assert words["Tabakindustrie"]["m"]["plain"] == [0]
    assert words["Tabakindustrie"]["m"]["parts"] == [0, 0]  # the two parts, neither shared in this setting
    assert words["wachsen"]["cuts"] == []


def test_body_drops_the_title_and_rewrites_headings():
    passage = Passage("p", "Kanada", "Kanada\n\n=== Befehlskette ===\nDer Oberbefehlshaber ist die Königin.")
    assert body_of(passage) == "Befehlskette: Der Oberbefehlshaber ist die Königin."


def test_excerpt_takes_the_sentences_with_the_most_valuable_matches():
    body = "Erster Satz ohne Treffer. Der Tabak ist teuer. Noch ein Satz. Das Gericht tagt."
    excerpt, shortened = pick_excerpt(body, {"tabak", "gericht"}, {"tabak": 5.0, "gericht": 4.0}, SPLITTER)
    assert excerpt == "Der Tabak ist teuer. … Das Gericht tagt."
    assert shortened


def test_lucene_idf_falls_with_document_frequency():
    assert lucene_idf(1, 1000) > lucene_idf(100, 1000) > lucene_idf(900, 1000) > 0


def test_comparison_can_be_read_in_both_directions():
    reports = {
        "a": {"pools": {"own_hard_negatives_dropped": {"versus": {"b": {
            "recall@10": {"diff": 0.02, "ci95": [0.01, 0.03], "p": 0.04, "gained": 5, "lost": 2},
            "rr@10": {"diff": 0.01, "p": 0.2},
        }}}}},
        "b": {"pools": {"own_hard_negatives_dropped": {"versus": {}}}},
    }
    forward = comparison(reports, "a", "b")
    backward = comparison(reports, "b", "a")
    assert forward["diff"] == 0.02 and (forward["gained"], forward["lost"]) == (5, 2)
    assert backward["diff"] == -0.02 and backward["ci"] == [-0.03, -0.01]
    assert (backward["gained"], backward["lost"]) == (2, 5)


# --- the splitter score on the page: whose labels, and the comparison with the AI labels -------

from compound_search.export_site import ai_comparison, eval_block  # noqa: E402

EVALUATION = {
    "n_labelled": 2,
    "types": {"compound": 1, "name": 1},
    "splitters": {
        "lexicon": {
            "all": {"n": 2, "f1": 0.5, "over_split": 1},
            "per_word": {
                "Schulbuch": {"gold": [5], "predicted": [5], "type": "compound"},
                "Newell": {"gold": [], "predicted": [3], "type": "name"},
            },
        }
    },
}


def test_eval_block_says_who_labelled():
    assert eval_block(EVALUATION, "person")["labelled_by"] == "person"
    assert eval_block(EVALUATION, "ai")["labelled_by"] == "ai"


def test_eval_block_lists_only_the_words_a_splitter_got_wrong():
    block = eval_block(EVALUATION, "person")
    assert block["n"] == 2
    assert block["splitters"]["lexicon"]["f1"] == 0.5
    assert block["splitters"]["lexicon"]["errors"] == [{"word": "Newell", "gold": "Newell", "predicted": "New|ell", "type": "name"}]


def test_ai_comparison_reports_agreement_and_the_score_on_the_ai_labels(tmp_path):
    assert ai_comparison(tmp_path) is None
    (tmp_path / "label_agreement.json").write_text(
        json.dumps({"n": 100, "same_cuts": 86, "same_cuts_and_type": 61, "only_first_cuts": ["a"] * 10, "only_second_cuts": ["b", "c"], "both_cut_differently": ["d", "e"]})
    )
    (tmp_path / "splitter_eval_ai_labels.json").write_text(json.dumps({"splitters": {"lexicon": {"all": {"f1": 0.86}}, "hanta": {"all": {"f1": 0.86}}}}))
    assert ai_comparison(tmp_path) == {
        "n": 100,
        "same_cuts": 86,
        "cut_only_by_the_person": 10,
        "cut_only_by_the_ai": 2,
        "cut_differently": 2,
        "f1_on_ai_labels": {"lexicon": 0.86, "hanta": 0.86},
    }


from compound_search.export_site import first_pass  # noqa: E402


def test_first_pass_block_gives_the_scores_before_the_label_revision(tmp_path):
    assert first_pass(tmp_path) is None
    (tmp_path / "splitter_eval_first_pass.json").write_text(json.dumps({"splitters": {"lexicon": {"all": {"f1": 0.73}}, "hanta": {"all": {"f1": 0.77}}}}))
    (tmp_path / "label_agreement_first_pass.json").write_text(json.dumps({"n": 100, "same_cuts": 86}))
    assert first_pass(tmp_path) == {"f1": {"lexicon": 0.73, "hanta": 0.77}, "same_cuts_with_ai": 86}
