import unicodedata

from compound_search.text import plain_tokens


def test_lowercases_and_keeps_umlauts_and_eszett():
    assert plain_tokens("Die Straße über Köln") == ["die", "straße", "über", "köln"]


def test_punctuation_and_hyphens_separate_tokens():
    assert plain_tokens("Baden-Württemberg, (1952)!") == ["baden", "württemberg", "1952"]


def test_underscore_separates_tokens():
    # GermanDPR titles look like this.
    assert plain_tokens("Recht_der_Vereinigten_Staaten") == ["recht", "der", "vereinigten", "staaten"]


def test_compounds_stay_whole():
    # The baseline must not split compounds; that is what the study measures.
    assert plain_tokens("Bundesverfassungsgericht") == ["bundesverfassungsgericht"]


def test_decomposed_umlaut_equals_composed():
    decomposed = unicodedata.normalize("NFD", "Müller")
    assert decomposed != "Müller"
    assert plain_tokens(decomposed) == plain_tokens("Müller") == ["müller"]


def test_empty_and_symbol_only_text():
    assert plain_tokens("") == []
    assert plain_tokens("=== ===\n") == []


# --- stems and lemmas -------------------------------------------------------

from compound_search.text import lemma_tokens, raw_tokens, stem_tokens  # noqa: E402


def test_raw_tokens_keep_case():
    assert raw_tokens("Die Straße_über Köln") == ["Die", "Straße", "über", "Köln"]


def test_stems_join_inflected_forms():
    assert stem_tokens("Haus Häuser Häusern") == ["haus", "haus", "haus"]
    assert stem_tokens("Universität Universitäten") == ["universitat", "universitat"]


def test_stems_do_not_split_compounds():
    assert stem_tokens("Bundesverfassungsgerichts") == ["bundesverfassungsgericht"]


def test_lemmas_join_irregular_forms_that_stems_miss():
    assert lemma_tokens("ging gegangen gehen") == ["gehen", "gehen", "gehen"]
    assert stem_tokens("ging gegangen gehen") != ["gehen", "gehen", "gehen"]


def test_lemmas_are_lowercased():
    assert lemma_tokens("Den Häusern") == ["der", "haus"]


def test_lemmas_handle_a_capitalised_verb_at_sentence_start():
    assert lemma_tokens("Ging er?")[0] == "gehen"


def test_lemmas_do_not_split_compounds():
    assert lemma_tokens("Tabakkonzerne") == ["tabakkonzern"]
