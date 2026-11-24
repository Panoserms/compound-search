"""Unit tests for Stage 7 error analysis query sampler and formatter."""

from __future__ import annotations

from compound_search.data import Passage, Query, passage_id
from compound_search.error_analysis import (
    TRACKED_SYSTEMS,
    category_facts,
    compound_splits_in_text,
    format_markdown,
    select_50_queries,
)
from compound_search.splitter import Lexicon, LexiconSplitter

TEXTS = [
    "Industrie\n\nDie Konzerne der Tabakindustrie wachsen seit Jahren.",
    "Berlin\n\nDie Hauptstadt von Deutschland ist Berlin.",
]
PASSAGES = {passage_id(t): Passage(passage_id(t), t.split("\n")[0], t) for t in TEXTS}
QUERIES = [
    Query("q1", "Seit wann wächst der Tabakkonzern?", (passage_id(TEXTS[0]),), ("1990",), ()),
    Query("q2", "Was ist die Hauptstadt von Deutschland?", (passage_id(TEXTS[1]),), ("Berlin",), ()),
]
TOY_SPLITTER = LexiconSplitter(
    Lexicon(
        freq={"tabak": 50, "konzern": 50, "konzerne": 20, "industrie": 80},
        nouns={"tabak", "konzern", "konzerne", "industrie"},
    )
)


def test_compound_splits_in_text_finds_compound_parts():
    splits = compound_splits_in_text("Seit wann wächst der Tabakkonzern?", TOY_SPLITTER)
    assert len(splits) == 1
    assert splits[0]["word"] == "tabakkonzern"
    assert splits[0]["parts"] == ["tabak", "konzern"]


def test_select_50_queries_categorises_and_formats_markdown():
    ranks = {sys_name: {"q1": 1, "q2": 1} for sys_name in TRACKED_SYSTEMS}
    # Make q1 a splitting_win (plain > 10, stem > 10, parts_stem <= 10)
    ranks["bm25_plain"]["q1"] = 40
    ranks["bm25_stem"]["q1"] = 25
    ranks["bm25_parts_stem"]["q1"] = 2
    # Make q2 an llm_rewrite_wins_h5 (hybrid_parts_stem_dense > 10, rewrite <= 10)
    ranks["hybrid_parts_stem_dense"]["q2"] = 18
    ranks["hybrid_parts_stem_dense_rewrite"]["q2"] = 4

    analysis = select_50_queries(
        QUERIES,
        PASSAGES,
        ranks,
        splitter=TOY_SPLITTER,
        rewrites_cache={"Was ist die Hauptstadt von Deutschland?": "Berlin Regierungssitz"},
    )
    assert analysis["total_sampled"] == 2
    assert analysis["categories"]["splitting_wins"]["sampled"] == 1
    assert analysis["categories"]["llm_rewrite_wins_h5"]["sampled"] == 1
    q1_rec = next(q for q in analysis["queries"] if q["qid"] == "q1")
    assert "tabak" in q1_rec["shared_stems_gained_by_splitting"]
    md = format_markdown(analysis)
    assert "Stage 7: Error Analysis" in md
    assert "Berlin Regierungssitz" in md


def test_category_facts_count_over_all_matching_queries():
    ranks = {sys_name: {"q1": 1, "q2": 1} for sys_name in TRACKED_SYSTEMS}
    ranks["bm25_plain"]["q1"] = None  # not in the first 100
    ranks["bm25_stem"]["q1"] = 25
    ranks["bm25_parts_stem"]["q1"] = 2
    facts = category_facts(QUERIES, PASSAGES, ranks, TOY_SPLITTER)
    wins = facts["splitting_wins"]
    assert wins["n"] == 1
    assert wins["question_has_cut_word"] == 1  # Tabakkonzern
    assert wins["gains_shared_term"] == 1  # "tabak" is shared only after splitting
    assert wins["plain_bm25_outside_top100"] == 1
    assert wins["best_lexical_ranks_higher_than_plain"] == 1
    assert wins["dense_in_top10"] == 1
    assert facts["splitting_losses"]["n"] == 0
    assert set(facts) == {"splitting_wins", "splitting_losses", "hybrid_preprocessing_wins_h3", "llm_rewrite_wins_h5", "residual_hard_failures"}
