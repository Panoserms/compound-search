"""The label-gap check on questions that every strong system misses."""

from __future__ import annotations

from compound_search.data import Passage, Query
from compound_search.residual_check import residual_check

PASSAGES = {
    "g1": Passage("g1", "A", "A\nDie Antwort ist Karlsruhe."),
    "g2": Passage("g2", "B", "B\nDie Antwort ist Berlin."),
    "g3": Passage("g3", "C", "C\nDie Antwort ist Bonn."),
    "other": Passage("other", "D", "D\nAuch hier steht Berlin als Antwort."),
    "noise": Passage("noise", "E", "E\nNichts davon."),
}
QUERIES = [
    Query("q1", "eins?", ("g1",), ("Karlsruhe",)),
    Query("q2", "zwei?", ("g2",), ("Berlin",)),
    Query("q3", "drei?", ("g3",), ("Bonn",)),
]


def view(**ranked: list[str]) -> dict:
    return {qid: [(pid, 1.0) for pid in pids] for qid, pids in ranked.items()}


def test_residual_questions_are_missed_by_both_systems():
    final = view(q1=["g1"], q2=["other", "noise"], q3=["noise"])
    dense = view(q1=["noise"], q2=["noise"], q3=["noise"])
    out = residual_check(QUERIES, PASSAGES, final, dense)
    assert out["residual_qids"] == ["q2", "q3"]  # q1 is found by the final system
    assert out["n_residual"] == 2


def test_a_returned_passage_with_the_answer_string_is_counted():
    final = view(q1=["noise"], q2=["other", "noise"], q3=["noise"])
    dense = view(q1=["noise"], q2=["noise"], q3=["g3"])
    out = residual_check(QUERIES, PASSAGES, final, dense)
    assert out["residual_qids"] == ["q1", "q2"]  # q3 is found by the dense model
    assert out["qids_with_answer_string"] == ["q2"]
    assert out["n_residual_with_answer_string_in_final_top10"] == 1
