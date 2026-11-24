"""Stage 0: report the real data sizes and the plain-BM25 baseline on the DEV set.

The test questions are loaded only to be counted and to build the passage pool.
They are never searched here: the test set is run once, after the config is frozen.

Usage:  python -I -m compound_search.stage0 --raw data/raw --out results/stage0_baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from compound_search.bm25 import B, K1, BM25Index
from compound_search.data import DEV_SEED, DEV_SIZE, Query, build_task, load_germandpr, passage_id, split_dev
from compound_search.metrics import mean, recall_at_k, reciprocal_rank_at_k
from compound_search.stats import cluster_bootstrap_ci
from compound_search.text import plain_tokens

KS = (1, 5, 10, 20, 100)
DEPTH = max(KS)


def data_report(rows: dict[str, list[dict]], passages: dict, queries: dict[str, list[Query]], raw: Path) -> dict:
    report: dict = {"germandpr": {}, "germanquad": {}}
    texts_by_split: dict[str, dict[str, set[str]]] = {}
    for split, split_rows in rows.items():
        pos = [t for r in split_rows for t in r["positive_ctxs"]["text"]]
        neg = [t for r in split_rows for t in r["hard_negative_ctxs"]["text"]]
        texts_by_split[split] = {"pos": set(pos), "neg": set(neg)}
        report["germandpr"][split] = {
            "rows": len(split_rows),
            "distinct_questions": len(queries[split]),
            "questions_with_more_than_one_gold": sum(1 for q in queries[split] if len(q.gold) > 1),
            "positive_contexts": len(pos),
            "distinct_positive_contexts": len(set(pos)),
            "hard_negative_contexts": len(neg),
            "distinct_hard_negative_contexts": len(set(neg)),
            "hard_negatives_per_question": dict(sorted(Counter(len(r["hard_negative_ctxs"]["text"]) for r in split_rows).items())),
            "median_question_words": statistics.median(len(q.text.split()) for q in queries[split]),
            "median_positive_words": statistics.median(len(t.split()) for t in pos),
            "median_hard_negative_words": statistics.median(len(t.split()) for t in neg),
        }
    tr, te = texts_by_split["train"], texts_by_split["test"]
    train_q = {q.text for q in queries["train"]}
    report["germandpr"]["overlap"] = {
        "test_questions_also_in_train": sum(1 for q in queries["test"] if q.text in train_q),
        "test_positives_also_train_positives": len(te["pos"] & tr["pos"]),
        "test_positives_also_train_hard_negatives": len(te["pos"] & tr["neg"]),
        "positives_that_are_also_someones_hard_negative": len((tr["pos"] | te["pos"]) & (tr["neg"] | te["neg"])),
    }
    report["germandpr"]["passage_pool"] = {
        "distinct_passages": len(passages),
        "distinct_after_whitespace_normalisation": len({" ".join(p.text.split()) for p in passages.values()}),
        "median_words": statistics.median(len(p.text.split()) for p in passages.values()),
        "total_plain_tokens": sum(len(plain_tokens(p.text)) for p in passages.values()),
    }

    all_pos = tr["pos"] | te["pos"]
    for split in ("train", "test"):
        quad = pq.read_table(raw / "germanquad" / f"{split}.parquet").to_pylist()
        contexts = {r["context"] for r in quad}
        report["germanquad"][split] = {
            "rows": len(quad),
            "distinct_questions": len({r["question"] for r in quad}),
            "distinct_contexts": len(contexts),
            "contexts_identical_to_a_germandpr_positive": len(contexts & all_pos),
            "contexts_already_in_passage_pool": sum(1 for c in contexts if passage_id(c) in passages),
            "answers_per_question": dict(sorted(Counter(len(r["answers"]["text"]) for r in quad).items())),
        }
    quad_test_q = {r["question"] for r in pq.read_table(raw / "germanquad" / "test.parquet").to_pylist()}
    report["germanquad"]["germandpr_test_questions_found_in_germanquad_test"] = sum(
        1 for q in queries["test"] if q.text in quad_test_q
    )
    return report


def evaluate(index: BM25Index, queries: list[Query], passages: dict, drop_own_hard_negatives: bool) -> dict:
    """Score one query set.

    drop_own_hard_negatives=False: the pool exactly as shipped.
    drop_own_hard_negatives=True: each question's own BM25-mined hard negatives are removed from
    its ranking. They stay in the pool as ordinary distractors for all other questions.
    """
    extra = max(len(q.own_hard_negatives) for q in queries)
    results = index.search([q.text for q in queries], k=DEPTH + extra)
    ranked = []
    for q, hits in zip(queries, results):
        skip = set(q.own_hard_negatives) if drop_own_hard_negatives else set()
        ranked.append([pid for pid, _ in hits if pid not in skip][:DEPTH])

    out: dict = {"n_queries": len(queries)}
    if not drop_own_hard_negatives:
        # How strongly do the planted passages distort plain BM25?
        gold_rank = [next((i for i, p in enumerate(r) if p in q.gold), DEPTH) for q, r in zip(queries, ranked)]
        negs_in_top10 = [sum(1 for p in r[:10] if p in q.own_hard_negatives) for q, r in zip(queries, ranked)]
        out["own_hard_negatives"] = {
            "mean_in_top10": mean(negs_in_top10),
            "share_of_queries_where_top1_is_one": mean([float(r[0] in q.own_hard_negatives) for q, r in zip(queries, ranked)]),
            "share_of_queries_where_one_outranks_gold": mean(
                [float(any(p in q.own_hard_negatives for p in r[:g])) for q, r, g in zip(queries, ranked, gold_rank)]
            ),
        }
    per_query_recall10 = [recall_at_k(r, q.gold, 10) for r, q in zip(ranked, queries)]
    per_query_rr10 = [reciprocal_rank_at_k(r, q.gold, 10) for r, q in zip(ranked, queries)]
    for k in KS:
        out[f"recall@{k}"] = mean([recall_at_k(r, q.gold, k) for r, q in zip(ranked, queries)])
    out["mrr@10"] = mean(per_query_rr10)
    # Same intervals as every later stage: resample gold passages, not questions.
    clusters = [q.gold[0] for q in queries]
    out["recall@10_ci95"] = list(cluster_bootstrap_ci(per_query_recall10, clusters))
    out["mrr@10_ci95"] = list(cluster_bootstrap_ci(per_query_rr10, clusters))

    # Label check, uses gold answers for DIAGNOSIS only: among the misses at 10, how often
    # does a retrieved non-gold passage contain the answer string? Those are likely unlabelled positives.
    misses = [(q, r) for q, r, rec in zip(queries, ranked, per_query_recall10) if rec == 0.0]
    answer_in_top10 = sum(
        1 for q, r in misses if any(a and a in passages[pid].text for pid in r[:10] for a in q.answers)
    )
    out["misses@10"] = len(misses)
    out["misses@10_with_answer_string_in_a_retrieved_passage"] = answer_in_top10

    # Rough headroom proxy for the study: questions that contain a long word (13+ letters).
    long_word = [any(len(t) >= 13 for t in plain_tokens(q.text)) for q in queries]
    out["queries_with_a_13plus_letter_word"] = sum(long_word)
    out["misses@10_with_a_13plus_letter_word"] = sum(
        1 for lw, rec in zip(long_word, per_query_recall10) if lw and rec == 0.0
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = load_germandpr(args.raw)
    passages, queries = build_task(rows)
    dev = split_dev(queries["train"])
    report = data_report(rows, passages, queries, args.raw)

    t0 = time.perf_counter()
    index = BM25Index(plain_tokens).build(list(passages), [p.text for p in passages.values()])
    build_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    dev_as_shipped = evaluate(index, dev, passages, drop_own_hard_negatives=False)
    dev_seconds = time.perf_counter() - t0
    dev_neutral = evaluate(index, dev, passages, drop_own_hard_negatives=True)
    train_as_shipped = evaluate(index, queries["train"], passages, drop_own_hard_negatives=False)
    train_neutral = evaluate(index, queries["train"], passages, drop_own_hard_negatives=True)

    result = {
        "stage": 0,
        "system": "bm25_plain",
        "note": "Dev set only. The test questions were not searched.",
        "config": {
            "bm25": {"library": "bm25s 0.3.16", "method": "lucene", "k1": K1, "b": B},
            "tokenizer": "plain_tokens: NFC, lowercase, runs of letters/digits, no stop words, no stemming",
            "passage_pool": "all GermanDPR positives and hard negatives, train and test, deduplicated by text",
            "dev": {"size": DEV_SIZE, "seed": DEV_SEED, "source": "random sample of distinct train questions"},
        },
        "data": report,
        "pools": {
            "as_shipped": "every passage competes for every question",
            "own_hard_negatives_dropped": (
                "each question's own BM25-mined hard negatives are removed from its ranking; "
                "they remain distractors for all other questions"
            ),
        },
        "dev": {"as_shipped": dev_as_shipped, "own_hard_negatives_dropped": dev_neutral},
        "all_train_questions_cross_check": {
            "as_shipped": train_as_shipped,
            "own_hard_negatives_dropped": train_neutral,
        },
        "timing_seconds": {"index_build": round(build_seconds, 2), "search_dev": round(dev_seconds, 2)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
