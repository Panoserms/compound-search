import pytest

from compound_search.data import Query
from compound_search.labels import parse_split, sample_words, score_splits


def q(i, text):
    return Query(f"q{i}", text, gold=("g",), answers=("a",))


QUERIES = [
    q(1, "Welche Universitätsbibliothek steht in der Hansestadt Hamburg?"),
    q(2, "Wann wurde das Bundesverfassungsgericht in Karlsruhe gegründet?"),
    q(3, "Wie heißt der Bürgermeister von München und Stuttgart?"),
    q(4, "Warum ist die Staatsangehörigkeit im Grundgesetz wichtig?"),
]


def test_sample_takes_capitalised_words_that_are_not_the_first_word():
    words = {w.word for w in sample_words(QUERIES, n_long=10, n_short=10, seed=1)}
    assert "Universitätsbibliothek" in words and "Karlsruhe" in words
    assert "Welche" not in words and "Warum" not in words  # first words of a question
    assert "wurde" not in words and "wichtig" not in words  # lowercase
    assert "Wann" not in words  # shorter than six letters, and first


def test_sample_is_stratified_by_length():
    words = sample_words(QUERIES, n_long=2, n_short=3, seed=1)
    assert sum(1 for w in words if w.stratum == "long") == 2
    assert sum(1 for w in words if w.stratum == "short") == 3
    assert all(len(w.word) >= 12 for w in words if w.stratum == "long")
    assert all(6 <= len(w.word) < 12 for w in words if w.stratum == "short")


def test_sample_is_reproducible_and_has_context():
    first = sample_words(QUERIES, n_long=2, n_short=2, seed=7)
    assert first == sample_words(QUERIES, n_long=2, n_short=2, seed=7)
    assert all(w.word in w.context for w in first)


def test_sample_has_no_repeated_words():
    doubled = QUERIES + [q(9, "Wo liegt Karlsruhe und wo liegt Hamburg?")]
    words = [w.word for w in sample_words(doubled, n_long=10, n_short=10, seed=1)]
    assert len(words) == len(set(words))


def test_parse_split_turns_plus_signs_into_offsets():
    assert parse_split("Bundesverfassungsgericht", "Bundes+verfassungs+gericht") == (6, 17)
    assert parse_split("Karlsruhe", "Karlsruhe") == ()


def test_parse_split_rejects_a_label_that_does_not_spell_the_word():
    with pytest.raises(ValueError):
        parse_split("Karlsruhe", "Karls+ruh")


GOLD = {"Schulbuch": (5,), "Bundesverfassungsgericht": (6, 17), "Karlsruhe": (), "Bahnhof": (4,), "Minister": ()}


def test_score_counts_each_outcome():
    predicted = {
        "Schulbuch": (5,),                    # correct split
        "Bundesverfassungsgericht": (6,),     # split, but not the gold cuts
        "Karlsruhe": (),                      # correctly left whole
        "Bahnhof": (),                        # missed compound
        "Minister": (4,),                     # split a word that is not a compound
    }
    s = score_splits(GOLD, predicted)
    assert s["n"] == 5
    assert s["correct_split"] == 1
    assert s["wrong_cuts"] == 1
    assert s["correct_whole"] == 1
    assert s["missed"] == 1
    assert s["over_split"] == 1


def test_score_precision_recall_accuracy():
    predicted = {"Schulbuch": (5,), "Bundesverfassungsgericht": (6,), "Karlsruhe": (), "Bahnhof": (), "Minister": (4,)}
    s = score_splits(GOLD, predicted)
    assert s["accuracy"] == pytest.approx(2 / 5)      # correct split + correctly whole
    assert s["precision"] == pytest.approx(1 / 3)     # of 3 words the splitter cut, 1 is right
    assert s["recall"] == pytest.approx(1 / 3)        # of 3 gold compounds, 1 is right
    assert s["f1"] == pytest.approx(1 / 3)


def test_score_cut_level_counts_partial_credit():
    predicted = {"Schulbuch": (5,), "Bundesverfassungsgericht": (6,), "Karlsruhe": (), "Bahnhof": (), "Minister": (4,)}
    s = score_splits(GOLD, predicted)
    # gold cuts: 1 + 2 + 1 = 4; predicted cuts: 1 + 1 + 1 = 3; matching: Schulbuch 5, Bundes... 6 = 2
    assert s["cut_precision"] == pytest.approx(2 / 3)
    assert s["cut_recall"] == pytest.approx(2 / 4)


def test_score_needs_a_prediction_for_every_gold_word():
    with pytest.raises(KeyError):
        score_splits(GOLD, {"Schulbuch": (5,)})


# --- the labelling page ---------------------------------------------------------

from compound_search.label_page import render_page  # noqa: E402
from compound_search.labels import SheetWord  # noqa: E402


def test_page_contains_every_word_and_its_context():
    words = [SheetWord("Schulbuch", "Wer schrieb das Schulbuch?", "short"), SheetWord("Bundesverfassungsgericht", "Wo sitzt das Bundesverfassungsgericht?", "long")]
    page = render_page(words)
    assert page.startswith("<!doctype html>")
    assert '"word": "Schulbuch"' in page and '"word": "Bundesverfassungsgericht"' in page
    assert "Wo sitzt das Bundesverfassungsgericht?" in page
    assert "compounds_labels.csv" in page


def test_page_cannot_be_broken_by_text_in_a_question():
    page = render_page([SheetWord("Schulbuch", "Was ist </script><b>das</b> Schulbuch?", "short")])
    assert "</script><b>" not in page


# --- which labels count, and how two label sets compare -------------------------

import json  # noqa: E402

from compound_search.labels import label_agreement, load_splitter_eval  # noqa: E402


def test_a_persons_labels_are_preferred_over_the_ai_labels(tmp_path):
    (tmp_path / "splitter_eval_ai_labels.json").write_text(json.dumps({"n_labelled": 1}))
    assert load_splitter_eval(tmp_path) == ({"n_labelled": 1}, "ai")
    (tmp_path / "splitter_eval.json").write_text(json.dumps({"n_labelled": 2}))
    assert load_splitter_eval(tmp_path) == ({"n_labelled": 2}, "person")


def test_no_score_file_gives_none(tmp_path):
    assert load_splitter_eval(tmp_path) is None


def label(word, split, kind):
    return {"word": word, "split": split, "type": kind, "cuts": parse_split(word, split or word)}


def test_agreement_counts_cuts_and_types():
    first = [
        label("Schulbuch", "Schul+buch", "compound"),
        label("Bahnhof", "Bahn+hof", "lexicalised compound"),
        label("Verschiebung", "Ver+schiebung", "lexicalised compound"),
        label("Griechenland", "", "name"),
        label("Regierung", "", "not a compound"),
    ]
    second = [
        label("Schulbuch", "Schul+buch", "compound"),
        label("Bahnhof", "Bahn+hof", "compound"),
        label("Verschiebung", "", "not a compound"),
        label("Griechenland", "Griechen+land", "name"),
        label("Regierung", "", "not a compound"),
    ]
    result = label_agreement(first, second)
    assert result["n"] == 5
    assert result["same_cuts"] == 3
    assert result["same_cuts_and_type"] == 2
    assert result["only_first_cuts"] == ["Verschiebung"]
    assert result["only_second_cuts"] == ["Griechenland"]
    assert result["both_cut_differently"] == []


def test_agreement_needs_the_same_words():
    with pytest.raises(ValueError):
        label_agreement([label("Schulbuch", "", "name")], [label("Bahnhof", "", "name")])
