"""The workbench: builds any named system for one query set and reports on it.

Lexical systems differ only in their tokeniser. They are rebuilt on every call
(a few seconds), so a cached ranking can never go stale after a code change.
Dense embeddings of the passage pool depend only on the passage texts and may be
cached on disk under data/cache/. Hybrids fuse the pool views of their members
with reciprocal rank fusion (k = 60, top 100).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from compound_search.bm25 import BM25Index
from compound_search.data import Passage, Query
from compound_search.dense import DenseIndex, Encoder
from compound_search.report import Views, build_report, views_of_run
from compound_search.rewrite import OllamaRewriter, Rewriter, select_uncertain_qids
from compound_search.runs import DEPTH, POOLS, RAW_DEPTH, RRF_K, Run, rrf
from compound_search.splitter import HantaSplitter, Lexicon, LexiconSplitter, make_split_tokenizer
from compound_search.text import lemma, lemma_tokens, plain_tokens, stem, stem_tokens

BASELINE = "bm25_plain"
DENSE_SYSTEM = "dense"
DENSE_DESCRIPTION = "Dense retrieval with multilingual-e5-small (used as published, no fine-tuning)"

Tokenizer = Callable[[str], list[str]]

# name -> (description, function that builds the tokeniser for a workbench)
LEXICAL: dict[str, tuple[str, Callable[["Workbench"], Tokenizer]]] = {
    "bm25_plain": ("BM25 on plain lowercase tokens (the baseline)", lambda b: plain_tokens),
    "bm25_stem": ("BM25 on Snowball German stems", lambda b: stem_tokens),
    "bm25_lemma": ("BM25 on simplemma lemmas", lambda b: lemma_tokens),
    "bm25_split": (
        "BM25 on plain tokens; noun compounds also indexed by their parts (own lexicon splitter)",
        lambda b: make_split_tokenizer(b.lexicon_splitter, str.lower),
    ),
    "bm25_split_stem": (
        "BM25 on stems; noun compounds also indexed by their parts (own lexicon splitter)",
        lambda b: make_split_tokenizer(b.lexicon_splitter, stem),
    ),
    "bm25_split_lemma": (
        "BM25 on lemmas; noun compounds also indexed by their parts (own lexicon splitter)",
        lambda b: make_split_tokenizer(b.lexicon_splitter, lemma),
    ),
    "bm25_parts_stem": (
        "BM25 on stems; noun compounds replaced by their parts (own lexicon splitter)",
        lambda b: make_split_tokenizer(b.lexicon_splitter, stem, keep_whole=False),
    ),
    "bm25_hanta_stem": (
        "BM25 on stems; noun compounds also indexed by their parts (HanTa morphology)",
        lambda b: make_split_tokenizer(b.hanta_splitter, stem),
    ),
    # Added on 2026-11-12, after the test run, for the website demo: "split compounds" without stems, with the
    # same meaning as in the best lexical system (parts replace the whole word). Dev only, never tested.
    "bm25_parts": (
        "BM25 on plain tokens; noun compounds replaced by their parts (own lexicon splitter) (dev only)",
        lambda b: make_split_tokenizer(b.lexicon_splitter, str.lower, keep_whole=False),
    ),
}

# name -> (description, member system names fused with RRF over top 100)
HYBRIDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "hybrid_plain_dense": (
        "Reciprocal rank fusion (k=60) of plain BM25 and multilingual-e5-small",
        ("bm25_plain", "dense"),
    ),
    "hybrid_stem_dense": (
        "Reciprocal rank fusion (k=60) of BM25 on stems and multilingual-e5-small",
        ("bm25_stem", "dense"),
    ),
    "hybrid_split_stem_dense": (
        "Reciprocal rank fusion (k=60) of BM25 on stems + compound parts and multilingual-e5-small",
        ("bm25_split_stem", "dense"),
    ),
    "hybrid_parts_stem_dense": (
        "Reciprocal rank fusion (k=60) of BM25 on stems with compounds replaced by parts (best lexical) and multilingual-e5-small",
        ("bm25_parts_stem", "dense"),
    ),
    # Added on 2026-11-12, after the test run, for the website demo (see bm25_parts). Dev only, never tested.
    "hybrid_parts_dense": (
        "Reciprocal rank fusion (k=60) of BM25 on plain tokens with compounds replaced by parts and multilingual-e5-small (dev only)",
        ("bm25_parts", "dense"),
    ),
}

# name -> (description, base hybrid system name)
REWRITTEN_HYBRIDS: dict[str, tuple[str, str]] = {
    "hybrid_parts_stem_dense_rewrite": (
        "hybrid_parts_stem_dense + Gemma-4 (gemma4:e2b-it-qat, temp=0) query expansion on uncertain queries",
        "hybrid_parts_stem_dense",
    ),
    "hybrid_split_stem_dense_rewrite": (
        "hybrid_split_stem_dense + Gemma-4 (gemma4:e2b-it-qat, temp=0) query expansion on uncertain queries",
        "hybrid_split_stem_dense",
    ),
}


class Workbench:
    def __init__(
        self,
        passages: dict[str, Passage],
        queries: Sequence[Query],
        lexicon: Lexicon | None = None,
        dense_encoder: Encoder | None = None,
        rewriter: Rewriter | None = None,
        cache_dir: Path | None = None,
    ):
        self.passages = passages
        self.queries = list(queries)
        self._lexicon = lexicon
        self._dense_encoder = dense_encoder
        self._rewriter = rewriter
        self.cache_dir = cache_dir
        self._lexicon_splitter: LexiconSplitter | None = None
        self._hanta_splitter: HantaSplitter | None = None
        self._dense_index: DenseIndex | None = None
        self._bm25_indices: dict[str, BM25Index] = {}
        self._views: dict[str, Views] = {}
        self.descriptions: dict[str, str] = {}
        # rewritten system -> pool -> ids of the queries that were sent to the LLM
        self.routed: dict[str, dict[str, list[str]]] = {}

    @property
    def lexicon(self) -> Lexicon:
        """Word frequencies and noun-like forms, counted on the passage pool. No queries are used."""
        if self._lexicon is None:
            self._lexicon = Lexicon.from_texts(p.text for p in self.passages.values())
        return self._lexicon

    @property
    def lexicon_splitter(self) -> LexiconSplitter:
        if self._lexicon_splitter is None:
            self._lexicon_splitter = LexiconSplitter(self.lexicon)
        return self._lexicon_splitter

    @property
    def hanta_splitter(self) -> HantaSplitter:
        if self._hanta_splitter is None:
            self._hanta_splitter = HantaSplitter()
        return self._hanta_splitter

    @property
    def dense_index(self) -> DenseIndex:
        if self._dense_index is None:
            self._dense_index = DenseIndex(encoder=self._dense_encoder, cache_dir=self.cache_dir).build(
                list(self.passages), [p.text for p in self.passages.values()]
            )
        return self._dense_index

    @property
    def rewriter(self) -> Rewriter:
        if self._rewriter is None:
            self._rewriter = OllamaRewriter(cache_dir=self.cache_dir)
        return self._rewriter

    def _bm25_index(self, system: str) -> BM25Index:
        if system not in self._bm25_indices:
            tokenizer = LEXICAL[system][1](self)
            self._bm25_indices[system] = BM25Index(tokenizer).build(
                list(self.passages), [p.text for p in self.passages.values()]
            )
        return self._bm25_indices[system]

    def _lexical_run(self, system: str) -> Run:
        index = self._bm25_index(system)
        hits = index.search([q.text for q in self.queries], k=RAW_DEPTH)
        return Run(system, {q.qid: h for q, h in zip(self.queries, hits)})

    def _dense_run(self) -> Run:
        hits = self.dense_index.search([q.text for q in self.queries], k=RAW_DEPTH)
        return Run(DENSE_SYSTEM, {q.qid: h for q, h in zip(self.queries, hits)})

    def _hybrid_views(self, members: Sequence[str]) -> Views:
        member_views = [self.views(m) for m in members]
        return {
            pool: {
                q.qid: rrf([mv[pool][q.qid] for mv in member_views], k=RRF_K, depth=DEPTH)
                for q in self.queries
            }
            for pool in POOLS
        }

    def _rewritten_hybrid_views(self, system: str, base_hybrid: str) -> Views:
        lexical_system, dense_system = HYBRIDS[base_hybrid][1]
        base_views = self.views(base_hybrid)
        lex_views = self.views(lexical_system)
        dense_views = self.views(dense_system)

        uncertain_by_pool = {
            pool: select_uncertain_qids(base_views[pool], lex_views[pool], dense_views[pool])
            for pool in POOLS
        }
        self.routed[system] = {pool: sorted(qids) for pool, qids in uncertain_by_pool.items()}
        all_uncertain_qids = set().union(*uncertain_by_pool.values())
        uncertain_queries = [q for q in self.queries if q.qid in all_uncertain_qids]

        if not uncertain_queries:
            return {pool: dict(base_views[pool]) for pool in POOLS}

        expanded_texts = [self.rewriter.rewrite(q.text) for q in uncertain_queries]
        lex_hits = self._bm25_index(lexical_system).search(expanded_texts, k=RAW_DEPTH)
        dense_hits = self.dense_index.search(expanded_texts, k=RAW_DEPTH)

        exp_lex_views = views_of_run(
            Run(f"{system}_lex", {q.qid: h for q, h in zip(uncertain_queries, lex_hits)}),
            uncertain_queries,
        )
        exp_dense_views = views_of_run(
            Run(f"{system}_dense", {q.qid: h for q, h in zip(uncertain_queries, dense_hits)}),
            uncertain_queries,
        )

        out: Views = {}
        for pool in POOLS:
            pool_ranking = {}
            uncertain_set = uncertain_by_pool[pool]
            for q in self.queries:
                qid = q.qid
                if qid in uncertain_set:
                    pool_ranking[qid] = rrf(
                        [
                            lex_views[pool][qid],
                            dense_views[pool][qid],
                            exp_lex_views[pool][qid],
                            exp_dense_views[pool][qid],
                        ],
                        k=RRF_K,
                        depth=DEPTH,
                    )
                else:
                    pool_ranking[qid] = base_views[pool][qid]
            out[pool] = pool_ranking
        return out

    def views(self, system: str) -> Views:
        """The rankings of a system under both pools. Built once per workbench."""
        if system not in self._views:
            if system in LEXICAL:
                self._views[system] = views_of_run(self._lexical_run(system), self.queries)
                self.descriptions[system] = LEXICAL[system][0]
            elif system == DENSE_SYSTEM:
                self._views[system] = views_of_run(self._dense_run(), self.queries)
                self.descriptions[system] = DENSE_DESCRIPTION
            elif system in HYBRIDS:
                desc, members = HYBRIDS[system]
                self._views[system] = self._hybrid_views(members)
                self.descriptions[system] = desc
            elif system in REWRITTEN_HYBRIDS:
                desc, base_hybrid = REWRITTEN_HYBRIDS[system]
                self._views[system] = self._rewritten_hybrid_views(system, base_hybrid)
                self.descriptions[system] = desc
            else:
                raise KeyError(system)
        return self._views[system]

    def report(self, system: str, versus: Sequence[str] = (BASELINE,)) -> dict:
        views = self.views(system)
        others = {name: self.views(name) for name in versus if name != system}
        report = build_report(system, self.descriptions[system], views, self.queries, versus=others)
        if system in self.routed:
            # How many queries the score rule sent to the LLM, so the share can be checked later.
            report["routing"] = {
                pool: {"n_routed": len(qids), "share": len(qids) / len(self.queries), "qids": qids}
                for pool, qids in self.routed[system].items()
            }
        return report
