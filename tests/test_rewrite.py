"""Unit tests for Stage 5 (uncertainty routing rule and LLM query expansion)."""

from __future__ import annotations

from compound_search.rewrite import (
    DEFAULT_MAX_RRF_TOP1,
    OllamaRewriter,
    clean_llm_expansion,
    expand_query_text,
    is_uncertain_ranking,
    select_uncertain_qids,
)
from compound_search.runs import RRF_K


def test_clean_llm_expansion_strips_think_tags_and_labels():
    raw = "<think>Ich überlege kurz.</think>\nSuchbegriffe: Bundesverfassungsgericht, Karlsruhe, Gericht; Recht!"
    assert clean_llm_expansion(raw) == "Bundesverfassungsgericht Karlsruhe Gericht Recht"


def test_expand_query_text_appends_cleaned_terms():
    q = "Wo sitzt das Bundesverfassungsgericht?"
    out = expand_query_text(q, "Karlsruhe Verfassungsgericht")
    assert out == "Wo sitzt das Bundesverfassungsgericht? Karlsruhe Verfassungsgericht"
    assert expand_query_text(q, "   ") == q


def test_confident_agreement_at_rank_1_is_not_uncertain():
    # Both lexical and dense rank p1 first -> RRF score is 2 / (60 + 1)
    top_rrf = 2.0 / (RRF_K + 1)
    hybrid = [("p1", top_rrf), ("p2", 1.0 / (RRF_K + 2))]
    lex = [("p1", 12.0), ("p2", 8.0)]
    dense = [("p1", 0.90), ("p2", 0.80)]
    assert not is_uncertain_ranking(hybrid, lex, dense)


def test_disagreement_with_low_top1_rrf_is_uncertain():
    # Lexical ranks p1 first (absent in dense), dense ranks p2 first (absent in lexical)
    low_rrf = 1.0 / (RRF_K + 1)
    assert low_rrf < DEFAULT_MAX_RRF_TOP1
    hybrid = [("p1", low_rrf), ("p2", low_rrf)]
    lex = [("p1", 9.5), ("p3", 8.9)]
    dense = [("p2", 0.84), ("p4", 0.83)]
    assert is_uncertain_ranking(hybrid, lex, dense)


def test_disagreement_with_strong_shared_runner_up_is_not_uncertain():
    # Lexical has p1 rank 1, p2 rank 2; Dense has p2 rank 1 -> p2 gets 1/61 + 1/62 > DEFAULT_MAX_RRF_TOP1
    strong_rrf = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 2))
    assert strong_rrf > DEFAULT_MAX_RRF_TOP1
    hybrid = [("p2", strong_rrf), ("p1", 1.0 / (RRF_K + 1))]
    lex = [("p1", 10.0), ("p2", 9.8)]
    dense = [("p2", 0.91), ("p3", 0.80)]
    assert not is_uncertain_ranking(hybrid, lex, dense)


def test_select_uncertain_qids_picks_only_uncertain_queries():
    top_rrf = 2.0 / (RRF_K + 1)
    low_rrf = 1.0 / (RRF_K + 1)
    hybrid_view = {
        "q_sure": [("p1", top_rrf)],
        "q_unsure": [("p2", low_rrf)],
    }
    lex_view = {
        "q_sure": [("p1", 14.0)],
        "q_unsure": [("p2", 8.0)],
    }
    dense_view = {
        "q_sure": [("p1", 0.92)],
        "q_unsure": [("p3", 0.81)],
    }
    assert select_uncertain_qids(hybrid_view, lex_view, dense_view) == {"q_unsure"}


def test_ollama_rewriter_reads_from_disk_cache_without_network(tmp_path):
    rewriter = OllamaRewriter(cache_dir=tmp_path)
    q = "Seit wann wächst der Tabakkonzern?"
    rewriter._cache[q] = "Tabak Industrie Konzern Unternehmen"
    rewriter._save_cache()

    rewriter2 = OllamaRewriter(cache_dir=tmp_path)
    assert rewriter2.rewrite(q) == "Seit wann wächst der Tabakkonzern? Tabak Industrie Konzern Unternehmen"
