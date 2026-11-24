"""The splitter is tested with a small hand-made lexicon, so every case is under control."""

import pytest

from compound_search.splitter import Lexicon, LexiconSplitter, Part

# form -> frequency. Everything listed under NOUNS counts as noun-like.
NOUNS = {
    "tabak": 80, "konzern": 120, "konzerne": 30, "universität": 900, "bibliothek": 300,
    "bund": 400, "gericht": 500, "verfassung": 350, "sonne": 600, "schein": 90,
    "kind": 700, "arbeit": 1500, "schule": 1200, "buch": 2000, "bahn": 100, "hof": 50,
    "bahnhof": 500, "stadt": 5000, "tor": 150, "karlsruhe": 300, "grund": 800,
    "wirt": 40, "schaft": 12, "bruder": 500, "herz": 300, "wunsch": 200,
    "mini": 20, "liebe": 400, "brief": 350, "rat": 600, "haus": 3000, "rathaus": 40,
    "jahr": 9000, "zahl": 800, "einwohner": 2500, "ster": 2,
}
OTHER = {"der": 90000, "vor": 8000, "vorder": 15, "weiter": 900, "hin": 300, "ab": 4000}


@pytest.fixture(scope="module")
def splitter() -> LexiconSplitter:
    return LexiconSplitter(Lexicon(freq={**NOUNS, **OTHER}, nouns=set(NOUNS)))


def bases(splitter, word):
    return [p.base for p in splitter.analyse(word)]


def test_two_nouns(splitter):
    assert bases(splitter, "Tabakkonzern") == ["tabak", "konzern"]


def test_inflected_head_is_kept_as_it_is_written(splitter):
    assert bases(splitter, "Tabakkonzerne") == ["tabak", "konzerne"]


def test_linking_s(splitter):
    parts = splitter.analyse("Universitätsbibliothek")
    assert parts == [Part("universitäts", "universität", "s"), Part("bibliothek", "bibliothek", "")]


def test_linking_es_and_three_parts(splitter):
    parts = splitter.analyse("Bundesverfassungsgericht")
    assert [(p.base, p.link) for p in parts] == [("bund", "es"), ("verfassung", "s"), ("gericht", "")]


def test_linking_n(splitter):
    assert [(p.base, p.link) for p in splitter.analyse("Sonnenschein")] == [("sonne", "n"), ("schein", "")]


def test_linking_er(splitter):
    assert [(p.base, p.link) for p in splitter.analyse("Kinderarbeit")] == [("kind", "er"), ("arbeit", "")]


def test_linking_ens(splitter):
    assert [(p.base, p.link) for p in splitter.analyse("Herzenswunsch")] == [("herz", "ens"), ("wunsch", "")]


def test_dropped_e_is_restored(splitter):
    parts = splitter.analyse("Schulbuch")
    assert parts == [Part("schul", "schule", "-e"), Part("buch", "buch", "")]


def test_surfaces_spell_the_word(splitter):
    for word in ("Bundesverfassungsgericht", "Schulbuch", "Karlsruhe", "Kinderarbeit"):
        assert "".join(p.surface for p in splitter.analyse(word)) == word.lower()


def test_input_case_does_not_matter(splitter):
    assert bases(splitter, "TABAKKONZERN") == bases(splitter, "tabakkonzern") == ["tabak", "konzern"]


def test_word_without_known_parts_stays_whole(splitter):
    assert splitter.analyse("Karlsruhe") == [Part("karlsruhe", "karlsruhe", "")]
    assert splitter.analyse("Xylophon") == [Part("xylophon", "xylophon", "")]


def test_frequent_whole_word_beats_rarer_parts(splitter):
    # Koehn & Knight: "Bahnhof" (500) is more frequent than the geometric mean of Bahn (100) and Hof (50).
    assert bases(splitter, "Bahnhof") == ["bahnhof"]


def test_rare_whole_word_is_split_into_frequent_parts(splitter):
    # "Rathaus" (40) loses against Rat (600) and Haus (3000).
    assert bases(splitter, "Rathaus") == ["rat", "haus"]


def test_unseen_compound_is_split(splitter):
    assert bases(splitter, "Stadttor") == ["stadt", "tor"]


def test_parts_must_be_noun_like(splitter):
    # "vor" + "der" + "grund" and "weiter" + "hin" use function words, so no split.
    assert bases(splitter, "Vordergrund") == ["vordergrund"]
    assert bases(splitter, "weiterhin") == ["weiterhin"]


def test_parts_need_a_minimum_length(splitter):
    assert bases(splitter, "Abbild") == ["abbild"]


def test_parts_need_a_minimum_frequency(splitter):
    # "Ster" is a noun in the lexicon, but it occurs only twice.
    assert bases(splitter, "Minister") == ["minister"]


def test_noun_like_suffixes_are_not_parts(splitter):
    # "Schaft" is a noun, but in "Bruderschaft" it is a suffix.
    assert bases(splitter, "Bruderschaft") == ["bruderschaft"]


def test_short_words_are_never_split(splitter):
    assert bases(splitter, "Hof") == ["hof"]
    assert bases(splitter, "Rates") == ["rates"]


def test_terms_gives_whole_word_first_then_parts(splitter):
    assert splitter.terms("Tabakkonzern") == ["tabakkonzern", "tabak", "konzern"]
    assert splitter.terms("Karlsruhe") == ["karlsruhe"]


def test_boundaries_are_character_offsets(splitter):
    assert splitter.boundaries("Bundesverfassungsgericht") == (6, 17)
    assert splitter.boundaries("Karlsruhe") == ()


# --- building the lexicon from text ------------------------------------------

TEXT = (
    "Karlsruhe\n\n=== Die Gerichte ===\n"
    "Das Gericht tagt in der Stadt. Die Stadt hat ein Gericht. Schöne Häuser stehen dort.\n"
    "Er sah die schöne Stadt und das schöne Gericht. Schöne Aussicht!"
)


def test_lexicon_counts_lowercased_forms():
    lex = Lexicon.from_texts([TEXT])
    assert lex.freq["stadt"] == 3
    assert lex.freq["gericht"] == 3
    assert lex.freq["schöne"] == 4


def test_noun_like_means_capitalised_inside_sentences():
    lex = Lexicon.from_texts([TEXT], min_noun_evidence=2)
    assert "stadt" in lex.nouns and "gericht" in lex.nouns
    # "Schöne" is capitalised only at sentence starts, which is no evidence.
    assert "schöne" not in lex.nouns
    assert "der" not in lex.nouns


def test_title_and_heading_lines_are_no_evidence_for_nouns():
    lex = Lexicon.from_texts(["Die Schöne\n\n=== Die Schöne ===\nSie ist schöne Dinge gewohnt, schöne Dinge."], min_noun_evidence=1)
    assert "schöne" not in lex.nouns
    assert "dinge" in lex.nouns


# --- fixes after the first run on real dev words ------------------------------


def lexicon_splitter(nouns: dict[str, int], other: dict[str, int] | None = None) -> LexiconSplitter:
    return LexiconSplitter(Lexicon(freq={**nouns, **(other or {})}, nouns=set(nouns)))


def test_rare_inflected_head_counts_with_its_stem():
    # "Unfällen" itself is rare, but it is a form of the frequent noun "Unfall".
    sp = lexicon_splitter({"verkehr": 1500, "unfall": 300, "sun": 30, "fall": 2000}, {"unfällen": 1, "fällen": 200})
    assert [(p.base, p.link) for p in sp.analyse("Verkehrsunfällen")] == [("verkehr", "s"), ("unfällen", "")]


def test_whole_word_frequency_pools_its_inflected_forms():
    # Bahnhof 20 + Bahnhöfe 30 + Bahnhofs 40 = 90, more than the geometric mean of Bahn (100) and Hof (50).
    sp = lexicon_splitter({"bahn": 100, "hof": 50, "bahnhof": 20, "bahnhöfe": 30, "bahnhofs": 40})
    assert [p.base for p in sp.analyse("Bahnhof")] == ["bahnhof"]


def test_linking_er_needs_a_consonant_before_it():
    # Not Energie-er + Haltung, although "Haltung" is more frequent than "Erhaltung".
    sp = lexicon_splitter({"energie": 500, "erhaltung": 60, "haltung": 300, "satz": 400})
    assert [(p.base, p.link) for p in sp.analyse("Energieerhaltungssatz")] == [
        ("energie", ""),
        ("erhaltung", "s"),
        ("satz", ""),
    ]


def test_linking_n_only_follows_e_el_or_er():
    # Not Pla-n + Stelle.
    sp = lexicon_splitter({"pla": 500, "plan": 100, "stelle": 300})
    assert [(p.base, p.link) for p in sp.analyse("Planstelle")] == [("plan", ""), ("stelle", "")]


def test_linking_s_does_not_follow_s():
    # Not Hau-s... and not Haus-s: "Hausschuh" is Haus + Schuh, with no linking element.
    sp = lexicon_splitter({"haus": 3000, "schuh": 80, "chuh": 900})
    assert [(p.base, p.link) for p in sp.analyse("Hausschuh")] == [("haus", ""), ("schuh", "")]


def test_dropped_e_is_only_assumed_when_nothing_else_fits():
    # "Atom" is a word, so "Atomspaltung" is not Atom(e) + Spaltung, even if "Atome" is more frequent.
    sp = lexicon_splitter({"atom": 100, "atome": 300, "spaltung": 80})
    assert [(p.base, p.link) for p in sp.analyse("Atomspaltung")] == [("atom", ""), ("spaltung", "")]


def test_modifier_must_itself_be_a_noun_form():
    # "weiter" shares its stem with the noun "Weite", but it is not a noun.
    sp = lexicon_splitter({"weite": 50, "bildung": 900}, {"weiter": 900})
    assert [p.base for p in sp.analyse("Weiterbildung")] == ["weiterbildung"]


def test_dropped_e_can_be_followed_by_linking_s():
    # Geschichte -> Geschichts|buch.
    sp = lexicon_splitter({"geschichte": 2500, "buch": 2000})
    assert sp.analyse("Geschichtsbuch") == [Part("geschichts", "geschichte", "-e+s"), Part("buch", "buch", "")]


def test_linking_s_may_follow_a_vowel():
    sp = lexicon_splitter({"liebe": 400, "brief": 350})
    assert [(p.base, p.link) for p in sp.analyse("Liebesbrief")] == [("liebe", "s"), ("brief", "")]


def test_linking_en_may_follow_a_diphthong():
    sp = lexicon_splitter({"frau": 900, "kirche": 700})
    assert [(p.base, p.link) for p in sp.analyse("Frauenkirche")] == [("frau", "en"), ("kirche", "")]


# --- from splitter to index terms ---------------------------------------------

from compound_search.splitter import make_split_tokenizer  # noqa: E402
from compound_search.text import lemma, stem  # noqa: E402


def test_tokenizer_adds_parts_after_the_whole_word(splitter):
    tokenize = make_split_tokenizer(splitter, str.lower)
    assert tokenize("Der Tabakkonzern wächst") == ["der", "tabakkonzern", "tabak", "konzern", "wächst"]


def test_tokenizer_only_splits_capitalised_words(splitter):
    # German nouns are capitalised. A lowercase word is an adjective or verb and is left alone.
    tokenize = make_split_tokenizer(splitter, str.lower)
    assert tokenize("ein tabakkonzern") == ["ein", "tabakkonzern"]


def test_tokenizer_can_replace_the_whole_word_by_its_parts(splitter):
    tokenize = make_split_tokenizer(splitter, str.lower, keep_whole=False)
    assert tokenize("Der Tabakkonzern in Karlsruhe") == ["der", "tabak", "konzern", "in", "karlsruhe"]


def test_tokenizer_stems_whole_word_and_parts(splitter):
    tokenize = make_split_tokenizer(splitter, stem)
    assert tokenize("Die Schulbücher") == ["die", "schulbuch", "schul", "buch"]


def test_tokenizer_lemmatises_parts_as_nouns(splitter):
    tokenize = make_split_tokenizer(splitter, lemma)
    assert tokenize("Die Kinderarbeit") == ["der", "kinderarbeit", "kind", "arbeit"]


# --- HanTa as a second splitter -------------------------------------------------

from compound_search.splitter import HantaSplitter  # noqa: E402


@pytest.fixture(scope="module")
def hanta() -> HantaSplitter:
    return HantaSplitter()


def test_hanta_leaves_out_linking_elements_and_endings(hanta):
    assert hanta.terms("Universitätsbibliothek") == ["universitätsbibliothek", "universität", "bibliothek"]
    assert hanta.terms("Tabakkonzerne") == ["tabakkonzerne", "tabak", "konzern"]


def test_hanta_boundary_comes_after_the_linking_element(hanta):
    assert hanta.boundaries("Kinderarbeit") == (6,)


def test_hanta_does_not_split_a_derived_word(hanta):
    assert hanta.terms("Entscheidungen") == ["entscheidungen"]
    assert hanta.boundaries("Entscheidungen") == ()


def test_hanta_does_not_treat_a_verb_particle_as_a_part(hanta):
    assert hanta.terms("Abkühlen") == ["abkühlen"]
