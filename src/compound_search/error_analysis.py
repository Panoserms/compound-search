"""Stage 7: Stratified error analysis of 50 dev queries across pipeline stages.

Selects 50 dev queries (seed 13) on the headline pool (`own_hard_negatives_dropped`)
spanning five diagnostic categories:
1. `splitting_wins` (15): `bm25_parts_stem` rescues queries missed by both `bm25_plain` and `bm25_stem`.
2. `splitting_losses` (10): `bm25_parts_stem` drops queries that `bm25_stem` ranked in top 10.
3. `hybrid_preprocessing_wins_h3` (10): `hybrid_parts_stem_dense` rescues `hybrid_plain_dense` (H3).
4. `llm_rewrite_wins_h5` (10): `hybrid_parts_stem_dense_rewrite` rescues `hybrid_parts_stem_dense` (H5).
5. `residual_hard_failures` (5): Queries missed at top 10 by both `hybrid_parts_stem_dense_rewrite` and `dense`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

from compound_search.data import DEV_SEED, Passage, Query, build_task, load_germandpr, split_dev
from compound_search.runs import DROPPED
from compound_search.splitter import Lexicon, LexiconSplitter, make_split_tokenizer
from compound_search.text import raw_tokens, stem, stem_tokens

TRACKED_SYSTEMS: tuple[str, ...] = (
    "bm25_plain",
    "bm25_stem",
    "bm25_parts_stem",
    "dense",
    "hybrid_plain_dense",
    "hybrid_parts_stem_dense",
    "hybrid_parts_stem_dense_rewrite",
)

CATEGORY_QUOTAS: tuple[tuple[str, str, int], ...] = (
    (
        "splitting_wins",
        "Stemming + compound splitting rescues BM25 (bm25_parts_stem <= 10, bm25_plain > 10 & bm25_stem > 10)",
        15,
    ),
    (
        "splitting_losses",
        "Compound splitting regression vs. stemming alone (bm25_stem <= 10, bm25_parts_stem > 10)",
        10,
    ),
    (
        "hybrid_preprocessing_wins_h3",
        "H3 mechanism: preprocessed hybrid rescues plain hybrid (hybrid_parts_stem_dense <= 10, hybrid_plain_dense > 10)",
        10,
    ),
    (
        "llm_rewrite_wins_h5",
        "H5 mechanism: Gemma-4 query expansion rescues hybrid (hybrid_parts_stem_dense_rewrite <= 10, hybrid_parts_stem_dense > 10)",
        10,
    ),
    (
        "residual_hard_failures",
        "Hard residual failures (both hybrid_parts_stem_dense_rewrite > 10 and dense > 10)",
        5,
    ),
)


def in_top10(rank: int | None) -> bool:
    return rank is not None and rank <= 10


def compound_splits_in_text(text: str, splitter: LexiconSplitter) -> list[dict[str, object]]:
    """Return every capitalised token in `text` that `splitter` decomposes into 2+ parts."""
    seen: set[str] = set()
    out: list[dict[str, object]] = []
    for tok in raw_tokens(text):
        low = tok.lower()
        if not tok[0].isupper() or low in seen:
            continue
        seen.add(low)
        parts = splitter.terms(tok)[1:]
        if len(parts) > 1:
            out.append({"word": low, "parts": list(parts), "stems": [stem(p) for p in parts]})
    return out


def build_query_record(
    query: Query,
    category: str,
    passages: Mapping[str, Passage],
    ranks_by_system: Mapping[str, Mapping[str, int | None]],
    splitter: LexiconSplitter,
    rewrites_cache: Mapping[str, str],
) -> dict:
    split_tok = make_split_tokenizer(splitter, stem, keep_whole=False)
    gold_pid = query.gold[0]
    gold_passage = passages[gold_pid]

    q_stem_set = set(stem_tokens(query.text))
    p_stem_set = set(stem_tokens(gold_passage.text))
    q_split_set = set(split_tok(query.text))
    p_split_set = set(split_tok(gold_passage.text))

    gained_stems = sorted((q_split_set & p_split_set) - (q_stem_set & p_stem_set))
    body_lines = [line.strip() for line in gold_passage.text.splitlines() if line.strip()]
    body_text = " ".join(body_lines[1:]) if len(body_lines) > 1 else gold_passage.text
    snippet = body_text[:260] + ("…" if len(body_text) > 260 else "")

    return {
        "category": category,
        "qid": query.qid,
        "question": query.text,
        "answers": list(query.answers),
        "gold_passage_id": gold_pid,
        "gold_title": gold_passage.title,
        "gold_snippet": snippet,
        "ranks": {sys_name: ranks_by_system[sys_name].get(query.qid) for sys_name in TRACKED_SYSTEMS},
        "query_compound_splits": compound_splits_in_text(query.text, splitter),
        "shared_stems_before_splitting": sorted(q_stem_set & p_stem_set),
        "shared_stems_gained_by_splitting": gained_stems,
        "llm_expansion": rewrites_cache.get(query.text),
    }


def category_predicates(ranks_by_system: Mapping[str, Mapping[str, int | None]]) -> dict:
    """For each category, the test that says whether a query (by id) belongs to it."""

    def r(sys_name: str, qid: str) -> int | None:
        return ranks_by_system[sys_name].get(qid)

    return {
        "splitting_wins": lambda qid: (
            not in_top10(r("bm25_plain", qid))
            and not in_top10(r("bm25_stem", qid))
            and in_top10(r("bm25_parts_stem", qid))
        ),
        "splitting_losses": lambda qid: (
            in_top10(r("bm25_stem", qid)) and not in_top10(r("bm25_parts_stem", qid))
        ),
        "hybrid_preprocessing_wins_h3": lambda qid: (
            not in_top10(r("hybrid_plain_dense", qid)) and in_top10(r("hybrid_parts_stem_dense", qid))
        ),
        "llm_rewrite_wins_h5": lambda qid: (
            not in_top10(r("hybrid_parts_stem_dense", qid))
            and in_top10(r("hybrid_parts_stem_dense_rewrite", qid))
        ),
        "residual_hard_failures": lambda qid: (
            not in_top10(r("hybrid_parts_stem_dense_rewrite", qid)) and not in_top10(r("dense", qid))
        ),
    }


def category_facts(
    queries: Sequence[Query],
    passages: Mapping[str, Passage],
    ranks_by_system: Mapping[str, Mapping[str, int | None]],
    splitter: LexiconSplitter,
) -> dict:
    """Counts over ALL dev queries of each category, not only the sampled ones.

    They are what the write-up may say about a category as a whole:
    - question_has_cut_word: the question contains a word the splitter cuts;
    - gains_shared_term: splitting gives question and gold passage at least one index term they did not share as stems;
    - loses_shared_term: a stem both texts shared is gone after splitting (a whole word was replaced by its parts);
    - plain_bm25_outside_top100: plain BM25 does not have the gold passage in its first 100;
    - best_lexical_ranks_higher_than_plain: bm25_parts_stem ranks the gold passage above bm25_plain;
    - dense_in_top10: the dense model alone has the gold passage in its top ten.
    """
    split_tok = make_split_tokenizer(splitter, stem, keep_whole=False)
    worst = 10**6
    facts: dict[str, dict[str, int]] = {}
    for name, pred in category_predicates(ranks_by_system).items():
        counts = dict.fromkeys(
            (
                "n",
                "question_has_cut_word",
                "gains_shared_term",
                "loses_shared_term",
                "plain_bm25_outside_top100",
                "best_lexical_ranks_higher_than_plain",
                "dense_in_top10",
            ),
            0,
        )
        for q in queries:
            if not pred(q.qid):
                continue
            gold = passages[q.gold[0]]
            shared_stems = set(stem_tokens(q.text)) & set(stem_tokens(gold.text))
            shared_split = set(split_tok(q.text)) & set(split_tok(gold.text))
            plain = ranks_by_system["bm25_plain"].get(q.qid)
            parts = ranks_by_system["bm25_parts_stem"].get(q.qid)
            counts["n"] += 1
            counts["question_has_cut_word"] += bool(compound_splits_in_text(q.text, splitter))
            counts["gains_shared_term"] += bool(shared_split - shared_stems)
            counts["loses_shared_term"] += bool(shared_stems - shared_split)
            counts["plain_bm25_outside_top100"] += plain is None
            counts["best_lexical_ranks_higher_than_plain"] += (parts or worst) < (plain or worst)
            counts["dense_in_top10"] += in_top10(ranks_by_system["dense"].get(q.qid))
        facts[name] = counts
    return facts


def select_50_queries(
    queries: Sequence[Query],
    passages: Mapping[str, Passage],
    ranks_by_system: Mapping[str, Mapping[str, int | None]],
    splitter: LexiconSplitter | None = None,
    rewrites_cache: Mapping[str, str] | None = None,
    seed: int = DEV_SEED,
) -> dict:
    if splitter is None:
        splitter = LexiconSplitter(Lexicon.from_texts(p.text for p in passages.values()))
    if rewrites_cache is None:
        rewrites_cache = {}

    predicates = category_predicates(ranks_by_system)

    rng = random.Random(seed)
    used_qids: set[str] = set()
    selected_records: list[dict] = []
    category_counts: dict[str, dict[str, int]] = {}

    for cat_key, cat_desc, quota in CATEGORY_QUOTAS:
        pred = predicates[cat_key]
        all_matching = [q for q in queries if pred(q.qid)]
        available = [q for q in all_matching if q.qid not in used_qids]
        rng.shuffle(available)
        chosen = sorted(available[:quota], key=lambda q: q.qid)
        for q in chosen:
            used_qids.add(q.qid)
            selected_records.append(
                build_query_record(q, cat_key, passages, ranks_by_system, splitter, rewrites_cache)
            )
        category_counts[cat_key] = {
            "description": cat_desc,
            "total_matching_in_dev": len(all_matching),
            "sampled": len(chosen),
            "quota": quota,
        }

    return {
        "split": "dev",
        "pool": DROPPED,
        "seed": seed,
        "total_sampled": len(selected_records),
        "categories": category_counts,
        "facts_over_all_matching_queries": category_facts(queries, passages, ranks_by_system, splitter),
        # Every word the splitter cuts in a question of the "splitting_losses" category, so that a reader can see
        # which of them are names.
        "splitting_losses_cut_words": sorted(
            {
                cut["word"]
                for q in queries
                if predicates["splitting_losses"](q.qid)
                for cut in compound_splits_in_text(q.text, splitter)
            }
        ),
        "queries": selected_records,
    }


def format_markdown(analysis: dict) -> str:
    lines = [
        "# Stage 7: Error Analysis of 50 Dev Queries",
        "",
        f"- **Split:** `{analysis['split']}` (1,000 train questions, seed {analysis['seed']})",
        f"- **Pool:** `{analysis['pool']}` (headline pool)",
        f"- **Total sampled queries:** {analysis['total_sampled']}",
        "",
        "## 1. Category Summary",
        "",
        "| Category | Description | Total in Dev | Sampled |",
        "|---|---|---:|---:|",
    ]
    for cat_key, info in analysis["categories"].items():
        lines.append(
            f"| `{cat_key}` | {info['description']} | {info['total_matching_in_dev']} | {info['sampled']} |"
        )

    facts = analysis.get("facts_over_all_matching_queries")
    if facts:
        keys = [k for k in next(iter(facts.values())) if k != "n"]
        lines.extend(
            [
                "",
                "### Counts over all matching dev queries (not only the sampled ones)",
                "",
                "| Category | n | " + " | ".join(k.replace("_", " ") for k in keys) + " |",
                "|---|---:|" + "---:|" * len(keys),
            ]
        )
        for cat_key, counts in facts.items():
            lines.append(f"| `{cat_key}` | {counts['n']} | " + " | ".join(str(counts[k]) for k in keys) + " |")

    lines.extend(["", "## 2. Query-by-Query Breakdown (50 Queries)", ""])
    for idx, item in enumerate(analysis["queries"], start=1):
        rk = item["ranks"]
        splits_str = (
            ", ".join(f"`{s['word']}` -> `{' + '.join(s['parts'])}`" for s in item["query_compound_splits"])
            or "*(none in query; compound split occurred in passage)*"
        )
        gained_str = ", ".join(f"`{st}`" for st in item["shared_stems_gained_by_splitting"]) or "*(none)*"
        lines.extend(
            [
                f"### {idx}. [{item['category']}] Query `{item['qid']}`",
                f"- **Question:** {item['question']}",
                f"- **Gold Answer(s):** {', '.join(item['answers'])} | **Gold Article:** *{item['gold_title']}* (`{item['gold_passage_id']}`)",
                f"- **Gold Ranks (`own_hard_negatives_dropped`):** "
                f"`plain`={rk['bm25_plain']}, `stem`={rk['bm25_stem']}, `parts_stem`={rk['bm25_parts_stem']}, "
                f"`dense`={rk['dense']}, `hyb_plain`={rk['hybrid_plain_dense']}, "
                f"`hyb_parts`={rk['hybrid_parts_stem_dense']}, `hyb_rewrite`={rk['hybrid_parts_stem_dense_rewrite']}",
                f"- **Query Compound Splits:** {splits_str}",
                f"- **New Shared Stems (Query ∩ Gold Passage) from Splitting:** {gained_str}",
            ]
        )
        if item.get("llm_expansion"):
            lines.append(f"- **Gemma-4 (`gemma4:e2b-it-qat`) Expansion:** `{item['llm_expansion']}`")
        lines.extend([f"- **Gold Passage Snippet:** > {item['gold_snippet']}", ""])
    return "\n".join(lines)


def load_dev_ranks(dev_dir: Path, pool: str = DROPPED) -> dict[str, dict[str, int | None]]:
    ranks: dict[str, dict[str, int | None]] = {}
    for sys_name in TRACKED_SYSTEMS:
        data = json.loads((dev_dir / f"{sys_name}.json").read_text(encoding="utf-8"))
        ranks[sys_name] = data["pools"][pool]["gold_rank"]
    return ranks


def load_rewrites_cache(cache_dir: Path) -> dict[str, str]:
    merged: dict[str, str] = {}
    for p in sorted(cache_dir.glob("rewrites_*.json")):
        merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--dev-results", type=Path, default=Path("results/dev"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out-json", type=Path, default=Path("results/error_analysis_50.json"))
    parser.add_argument("--out-md", type=Path, default=Path("results/error_analysis_50.md"))
    args = parser.parse_args()

    passages, all_queries = build_task(load_germandpr(args.raw))
    dev_queries = split_dev(all_queries["train"])
    ranks = load_dev_ranks(args.dev_results)
    rewrites = load_rewrites_cache(args.cache)

    analysis = select_50_queries(dev_queries, passages, ranks, rewrites_cache=rewrites)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(analysis, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    args.out_md.write_text(format_markdown(analysis), encoding="utf-8")

    print(f"Wrote {analysis['total_sampled']} queries to {args.out_json} and {args.out_md}")
    for cat_key, info in analysis["categories"].items():
        print(f"  {cat_key:32s}: sampled {info['sampled']:2d} / {info['total_matching_in_dev']:3d} matching dev queries")


if __name__ == "__main__":
    main()
