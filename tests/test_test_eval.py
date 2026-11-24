"""Unit tests for Stage 6 test evaluation and Holm-corrected hypothesis table."""

from __future__ import annotations

import numpy as np

from compound_search.data import Passage, Query, passage_id
from compound_search.pipeline import Workbench
from compound_search.runs import DROPPED, SHIPPED
from compound_search.splitter import Lexicon
from compound_search.test_eval import FROZEN_SYSTEMS, HYPOTHESIS_TESTS, run_test_evaluation

TEXTS = [
    "Industrie\n\nDie Konzerne der Tabakindustrie wachsen seit Jahren in Karlsruhe.",
    "Berlin\n\nDie Hauptstadt von Deutschland ist Berlin, seit Jahren.",
    "Bodensee\n\nIm Bodensee leben viele Fische. Die Häuser am Ufer sind alt.",
]
PASSAGES = {passage_id(t): Passage(passage_id(t), t.split("\n")[0], t) for t in TEXTS}
QUERIES = [
    Query("q1", "Seit wann wächst der Tabakkonzern?", (passage_id(TEXTS[0]),), ("x",), (passage_id(TEXTS[1]),)),
    Query("q2", "In welchem Haus am Bodensee?", (passage_id(TEXTS[2]),), ("x",), ()),
]
TOY_LEXICON = Lexicon(
    freq={"tabak": 50, "konzern": 50, "konzerne": 20, "industrie": 80},
    nouns={"tabak", "konzern", "konzerne", "industrie"},
)


class ToyEncoder:
    name = "toy-dense"

    def encode(self, texts, prefix: str) -> np.ndarray:
        out = []
        for t in texts:
            low = t.lower()
            v = np.array(
                [
                    1.0 if "tabak" in low or "karlsruhe" in low else 0.0,
                    1.0 if "bodensee" in low or "haus" in low or "häuser" in low else 0.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            v /= np.linalg.norm(v)
            out.append(v)
        return np.vstack(out)


class ToyRewriter:
    name = "toy-rewriter"

    def rewrite(self, query_text: str) -> str:
        return f"{query_text} Tabakindustrie Bodensee"


def test_run_test_evaluation_produces_all_six_hypotheses_and_the_headline(tmp_path):
    bench = Workbench(
        PASSAGES,
        QUERIES,
        lexicon=TOY_LEXICON,
        dense_encoder=ToyEncoder(),
        rewriter=ToyRewriter(),
        cache_dir=tmp_path,
    )
    out_dir = tmp_path / "test"
    summary_path = tmp_path / "test_summary.json"
    summary = run_test_evaluation(bench, out_dir=out_dir, summary_path=summary_path)

    assert summary["split"] == "test"
    assert summary["n_queries"] == len(QUERIES)
    assert set(summary["systems"]) == set(FROZEN_SYSTEMS)
    assert summary_path.exists()
    for sys_name in FROZEN_SYSTEMS:
        assert (out_dir / f"{sys_name}.json").exists()

    for pool in (DROPPED, SHIPPED):
        rows = summary["hypotheses"][pool]
        assert [r["id"] for r in rows] == [h[0] for h in HYPOTHESIS_TESTS]
        for r in rows:
            assert 0.0 <= r["recall@10_p_raw"] <= 1.0
            assert r["recall@10_p_raw"] <= r["recall@10_p_holm"] <= 1.0
            assert isinstance(r["supported"], bool)

    assert "points Recall@10 over plain BM25" in summary["headline"]["sentence_pipeline"]
