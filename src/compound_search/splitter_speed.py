"""Time both compound splitters on the same words, each starting cold.

    python -I -m compound_search.splitter_speed --raw data/raw --out results/splitter_speed.json

The words are every distinct capitalised word form of the passage pool (the words a
split tokeniser hands to a splitter). For the project's own splitter the time includes
building its word list from the pool, because it cannot work without it. For HanTa the
time includes loading its model. One run each, on whatever machine this is run on:
read the numbers as a rough ratio, not as a benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from compound_search.data import build_task, load_germandpr
from compound_search.splitter import HantaSplitter, Lexicon, LexiconSplitter
from compound_search.text import raw_tokens


def capitalised_words(texts) -> list[str]:
    """Distinct capitalised tokens, in first-seen order."""
    seen: dict[str, None] = {}
    for text in texts:
        for token in raw_tokens(text):
            if token[0].isupper():
                seen.setdefault(token)
    return list(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/splitter_speed.json"))
    args = parser.parse_args()

    passages, _ = build_task(load_germandpr(args.raw))
    texts = [p.text for p in passages.values()]
    words = capitalised_words(texts)

    t0 = time.perf_counter()
    lexicon = Lexicon.from_texts(texts)
    lexicon_seconds = time.perf_counter() - t0
    own = LexiconSplitter(lexicon)
    t0 = time.perf_counter()
    own_split = sum(1 for w in words if len(own.terms(w)) > 1)
    own_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    hanta = HantaSplitter()
    hanta_load_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    hanta_split = sum(1 for w in words if len(hanta.terms(w)) > 1)
    hanta_seconds = time.perf_counter() - t0

    result = {
        "n_passages": len(passages),
        "n_distinct_capitalised_words": len(words),
        "machine": f"{platform.machine()} {platform.system()}",
        "lexicon": {
            "build_word_list_seconds": round(lexicon_seconds, 1),
            "split_seconds": round(own_seconds, 1),
            "total_seconds": round(lexicon_seconds + own_seconds, 1),
            "words_split": own_split,
        },
        "hanta": {
            "load_model_seconds": round(hanta_load_seconds, 1),
            "split_seconds": round(hanta_seconds, 1),
            "total_seconds": round(hanta_load_seconds + hanta_seconds, 1),
            "words_split": hanta_split,
        },
    }
    result["hanta_total_over_lexicon_total"] = round(result["hanta"]["total_seconds"] / result["lexicon"]["total_seconds"], 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
