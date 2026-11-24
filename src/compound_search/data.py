"""Load GermanDPR and turn it into a retrieval task: one passage pool, queries with gold passages.

GermanDPR ships no passage ids (the field is empty), so a passage is identified
by a hash of its text. The same text used for several questions becomes one passage.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

SPLITS = ("train", "test")
DEV_SIZE = 1000
DEV_SEED = 13


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    qid: str
    text: str
    gold: tuple[str, ...]
    answers: tuple[str, ...]
    # The passages deepset mined for THIS question with BM25 (top hits without the answer string).
    # They sit in the pool for every query, but can be left out of this query's own ranking.
    own_hard_negatives: tuple[str, ...] = ()


def passage_id(text: str) -> str:
    """Hash of the text with whitespace runs collapsed.

    A few passages in GermanDPR differ only in whitespace. They get identical BM25
    scores, so keeping both would make the gold passage's rank depend on tie order.
    """
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()[:16]


def load_germandpr(raw_dir: Path) -> dict[str, list[dict]]:
    return {s: pq.read_table(Path(raw_dir) / "germandpr" / f"{s}.parquet").to_pylist() for s in SPLITS}


def build_task(rows_by_split: dict[str, list[dict]]) -> tuple[dict[str, Passage], dict[str, list[Query]]]:
    """Return (passages by id, queries by split).

    Passages: every positive and hard-negative context of every split, deduplicated by text.
    Queries: questions with identical text inside one split are merged; their gold sets are joined.
    """
    passages: dict[str, Passage] = {}
    queries: dict[str, list[Query]] = {}
    for split in SPLITS:
        merged: dict[str, dict[str, list[str]]] = {}
        for row in rows_by_split[split]:
            ids: dict[str, list[str]] = {"positive_ctxs": [], "hard_negative_ctxs": []}
            for field in ids:
                ctxs = row[field]
                for title, text in zip(ctxs["title"], ctxs["text"]):
                    pid = passage_id(text)
                    passages.setdefault(pid, Passage(pid, title, text))
                    ids[field].append(pid)
            entry = merged.setdefault(row["question"], {"gold": [], "answers": [], "negs": []})
            entry["gold"].extend(g for g in ids["positive_ctxs"] if g not in entry["gold"])
            entry["negs"].extend(n for n in ids["hard_negative_ctxs"] if n not in entry["negs"])
            entry["answers"].extend(a for a in row["answers"] if a not in entry["answers"])
        queries[split] = [
            Query(
                qid=f"{split}-{i:05d}",
                text=text,
                gold=tuple(e["gold"]),
                answers=tuple(e["answers"]),
                # A passage that is gold for the merged question is never treated as its negative.
                own_hard_negatives=tuple(n for n in e["negs"] if n not in e["gold"]),
            )
            for i, (text, e) in enumerate(merged.items())
        ]
    return passages, queries


def split_dev(train_queries: list[Query], size: int = DEV_SIZE, seed: int = DEV_SEED) -> list[Query]:
    """A fixed random sample of the train questions. This is the only set used for development."""
    rng = random.Random(seed)
    picked = sorted(rng.sample(range(len(train_queries)), size))
    return [train_queries[i] for i in picked]
