from compound_search.data import build_task, passage_id, split_dev


def ctxs(*texts):
    return {"title": [t.split()[0] for t in texts], "text": list(texts), "passage_id": [""] * len(texts)}


def row(question, positive, negatives, answer="x"):
    return {
        "question": question,
        "answers": [answer],
        "positive_ctxs": ctxs(positive),
        "negative_ctxs": ctxs(),
        "hard_negative_ctxs": ctxs(*negatives),
    }


ROWS = {
    "train": [
        row("Frage eins?", "Passage A", ["Passage B", "Passage C"]),
        row("Frage zwei?", "Passage B", ["Passage A", "Passage D"]),
        # Same question again with another positive: one query, two gold passages.
        row("Frage eins?", "Passage E", ["Passage C"], answer="y"),
    ],
    "test": [row("Testfrage?", "Passage F", ["Passage A"])],
}


def test_passages_are_deduplicated_across_roles_and_splits():
    passages, _ = build_task(ROWS)
    assert sorted(p.text for p in passages.values()) == [f"Passage {c}" for c in "ABCDEF"]
    assert all(pid == passage_id(p.text) for pid, p in passages.items())


def test_passages_that_differ_only_in_whitespace_are_one_passage():
    assert passage_id("Titel\n\nEin  Satz.") == passage_id("Titel\nEin Satz. ")
    assert passage_id("Ein Satz.") != passage_id("Ein Satz!")
    rows = {"train": [row("Frage?", "Passage A", ["Passage  A", "Passage B"])], "test": []}
    passages, queries = build_task(rows)
    assert len(passages) == 2
    # The whitespace twin of the gold passage is the gold passage, not a negative.
    assert queries["train"][0].own_hard_negatives == (passage_id("Passage B"),)


def test_duplicate_questions_are_merged_with_joined_gold():
    _, queries = build_task(ROWS)
    train = {q.text: q for q in queries["train"]}
    assert len(queries["train"]) == 2
    assert train["Frage eins?"].gold == (passage_id("Passage A"), passage_id("Passage E"))
    assert train["Frage eins?"].answers == ("x", "y")
    assert train["Frage zwei?"].gold == (passage_id("Passage B"),)


def test_own_hard_negatives_are_kept_per_question():
    _, queries = build_task(ROWS)
    train = {q.text: q for q in queries["train"]}
    assert train["Frage eins?"].own_hard_negatives == (passage_id("Passage B"), passage_id("Passage C"))
    assert train["Frage zwei?"].own_hard_negatives == (passage_id("Passage A"), passage_id("Passage D"))


def test_a_gold_passage_is_never_its_own_hard_negative():
    rows = {
        "train": [
            row("Frage?", "Passage A", ["Passage B"]),
            row("Frage?", "Passage B", ["Passage C"]),
        ],
        "test": [],
    }
    _, queries = build_task(rows)
    (q,) = queries["train"]
    assert q.gold == (passage_id("Passage A"), passage_id("Passage B"))
    assert q.own_hard_negatives == (passage_id("Passage C"),)


def test_test_queries_keep_their_own_split():
    _, queries = build_task(ROWS)
    assert [q.text for q in queries["test"]] == ["Testfrage?"]
    assert queries["test"][0].qid.startswith("test-")


def test_dev_split_is_fixed_and_drawn_from_train():
    _, queries = build_task(ROWS)
    first = split_dev(queries["train"], size=1, seed=13)
    again = split_dev(queries["train"], size=1, seed=13)
    assert first == again
    assert first[0] in queries["train"]
