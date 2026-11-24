"""The hand-labelled compound sheet: draw the sample, read the labels, score the splitters.

    python -I -m compound_search.labels make  --raw data/raw --out labels
    python -I -m compound_search.labels score --raw data/raw --labels labels/compounds_labels.csv --out results/splitter_eval.json
    python -I -m compound_search.labels score --raw data/raw --labels labels/compounds_labels_ai.csv --out results/splitter_eval_ai_labels.json
    python -I -m compound_search.labels agree --first labels/compounds_labels.csv --second labels/compounds_labels_ai.csv --out results/label_agreement.json

The sample comes from DEV questions only. The sheet never shows what a splitter did,
so the labels are made blind. `labels/compounds_labels.csv` holds the labels of a person
(exported from the labelling page); they are the ones the write-up reports.
`labels/compounds_labels_ai.csv` holds an earlier set made by an AI, kept for comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from compound_search.data import Query
from compound_search.text import raw_tokens

MIN_LEN = 6
LONG_FROM = 12
SHEET_SEED = 13
TYPES = ("compound", "lexicalised compound", "name", "not a compound")


@dataclass(frozen=True)
class SheetWord:
    word: str
    context: str
    stratum: str  # "long" (12+ letters) or "short" (6 to 11)


def sample_words(
    queries: Sequence[Query], n_long: int = 50, n_short: int = 50, seed: int = SHEET_SEED
) -> list[SheetWord]:
    """Capitalised words from the questions, half long and half short, in random order.

    The first word of a question is skipped: it is capitalised whatever it is.
    Sampling by length does not depend on any splitter, so no system is favoured.
    """
    first_context: dict[str, str] = {}
    for query in queries:
        for token in raw_tokens(query.text)[1:]:
            if token[0].isupper() and token.isalpha() and len(token) >= MIN_LEN:
                first_context.setdefault(token, query.text)
    rng = random.Random(seed)
    picked: list[SheetWord] = []
    for stratum, wanted, keep in (
        ("long", n_long, lambda w: len(w) >= LONG_FROM),
        ("short", n_short, lambda w: len(w) < LONG_FROM),
    ):
        candidates = sorted(w for w in first_context if keep(w))
        chosen = rng.sample(candidates, min(wanted, len(candidates)))
        picked.extend(SheetWord(w, first_context[w], stratum) for w in chosen)
    rng.shuffle(picked)
    return picked


def parse_split(word: str, label: str) -> tuple[int, ...]:
    """'Bundes+verfassungs+gericht' -> (6, 17). The label must spell the word."""
    pieces = label.strip().split("+")
    if "".join(pieces).lower() != word.lower() or any(not p for p in pieces):
        raise ValueError(f"label {label!r} does not spell {word!r}")
    offsets, position = [], 0
    for piece in pieces[:-1]:
        position += len(piece)
        offsets.append(position)
    return tuple(offsets)


def score_splits(gold: Mapping[str, tuple[int, ...]], predicted: Mapping[str, tuple[int, ...]]) -> dict:
    """Word-level outcomes as in Koehn & Knight (2003), plus precision and recall of single cuts."""
    counts = {"correct_split": 0, "wrong_cuts": 0, "correct_whole": 0, "missed": 0, "over_split": 0}
    cuts_gold = cuts_predicted = cuts_matching = 0
    for word, gold_cuts in gold.items():
        cuts = tuple(predicted[word])
        cuts_gold += len(gold_cuts)
        cuts_predicted += len(cuts)
        cuts_matching += len(set(gold_cuts) & set(cuts))
        if gold_cuts and cuts:
            counts["correct_split" if cuts == tuple(gold_cuts) else "wrong_cuts"] += 1
        elif gold_cuts:
            counts["missed"] += 1
        elif cuts:
            counts["over_split"] += 1
        else:
            counts["correct_whole"] += 1

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    precision = ratio(counts["correct_split"], counts["correct_split"] + counts["wrong_cuts"] + counts["over_split"])
    recall = ratio(counts["correct_split"], counts["correct_split"] + counts["wrong_cuts"] + counts["missed"])
    return {
        "n": len(gold),
        **counts,
        "accuracy": ratio(counts["correct_split"] + counts["correct_whole"], len(gold)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "cut_precision": ratio(cuts_matching, cuts_predicted),
        "cut_recall": ratio(cuts_matching, cuts_gold),
    }


def read_labels(path: Path) -> list[dict]:
    """Rows of the labelled sheet: word, split (with + signs), type, note. Unlabelled rows are skipped."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if (row.get("type") or "").strip()]
    for row in rows:
        row["cuts"] = parse_split(row["word"], row["split"] or row["word"])
    return rows


PERSON_EVAL = "splitter_eval.json"
AI_EVAL = "splitter_eval_ai_labels.json"


def load_splitter_eval(results: Path) -> tuple[dict, str] | None:
    """The splitter score the write-up reports, and who made its labels: "person" if such a score exists, else "ai"."""
    for name, labelled_by in ((PERSON_EVAL, "person"), (AI_EVAL, "ai")):
        path = Path(results) / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), labelled_by
    return None


def label_agreement(first: Sequence[dict], second: Sequence[dict]) -> dict:
    """How two label sets of the same words agree: on the cuts, and on cuts and type together."""
    if [r["word"] for r in first] != [r["word"] for r in second]:
        raise ValueError("the two label sets do not cover the same words in the same order")
    result: dict = {
        "n": len(first),
        "same_cuts": 0,
        "same_cuts_and_type": 0,
        "only_first_cuts": [],
        "only_second_cuts": [],
        "both_cut_differently": [],
    }
    for a, b in zip(first, second):
        cuts_a, cuts_b = tuple(a["cuts"]), tuple(b["cuts"])
        if cuts_a == cuts_b:
            result["same_cuts"] += 1
            result["same_cuts_and_type"] += a["type"].strip().lower() == b["type"].strip().lower()
        elif cuts_a and not cuts_b:
            result["only_first_cuts"].append(a["word"])
        elif cuts_b and not cuts_a:
            result["only_second_cuts"].append(a["word"])
        else:
            result["both_cut_differently"].append(a["word"])
    return result


def _agree(args: argparse.Namespace) -> None:
    result = {"first": str(args.first), "second": str(args.second), **label_agreement(read_labels(args.first), read_labels(args.second))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"{result['n']} words: same cuts {result['same_cuts']}, same cuts and type {result['same_cuts_and_type']}; "
        f"cut only in the first {len(result['only_first_cuts'])}, only in the second {len(result['only_second_cuts'])}, "
        f"cut differently {len(result['both_cut_differently'])}"
    )


def _make(args: argparse.Namespace) -> None:
    from compound_search.data import build_task, load_germandpr, split_dev
    from compound_search.label_page import render_page

    _, queries = build_task(load_germandpr(args.raw))
    words = sample_words(split_dev(queries["train"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "compounds_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "word", "split", "type", "note", "stratum", "context"])
        for i, w in enumerate(words, start=1):
            writer.writerow([i, w.word, "", "", "", w.stratum, w.context])
    (args.out / "label_compounds.html").write_text(render_page(words), encoding="utf-8")
    print(f"{len(words)} words: {sum(w.stratum == 'long' for w in words)} long, {sum(w.stratum == 'short' for w in words)} short")
    print(f"wrote {args.out / 'compounds_sheet.csv'} and {args.out / 'label_compounds.html'}")


def _score(args: argparse.Namespace) -> None:
    from compound_search.data import build_task, load_germandpr
    from compound_search.splitter import HantaSplitter, Lexicon, LexiconSplitter

    rows = read_labels(args.labels)
    passages, _ = build_task(load_germandpr(args.raw))
    splitters = [LexiconSplitter(Lexicon.from_texts(p.text for p in passages.values())), HantaSplitter()]
    subsets = {"all": rows}
    for name in TYPES:
        subsets[name] = [r for r in rows if r["type"].strip().lower() == name]
    for stratum in ("long", "short"):
        subsets[f"stratum: {stratum}"] = [r for r in rows if (r.get("stratum") or "") == stratum]
    result: dict = {"n_labelled": len(rows), "types": {name: len(subsets[name]) for name in TYPES}, "splitters": {}}
    for splitter in splitters:
        predicted = {r["word"]: splitter.boundaries(r["word"]) for r in rows}
        result["splitters"][splitter.name] = {
            name: score_splits({r["word"]: r["cuts"] for r in subset}, predicted) for name, subset in subsets.items() if subset
        }
        result["splitters"][splitter.name]["per_word"] = {
            r["word"]: {"gold": list(r["cuts"]), "predicted": list(predicted[r["word"]]), "type": r["type"]} for r in rows
        }
        s = result["splitters"][splitter.name]["all"]
        print(
            f"{splitter.name:8s} n={s['n']} accuracy={s['accuracy']:.2f} precision={s['precision']:.2f} recall={s['recall']:.2f}"
            f" f1={s['f1']:.2f} | correct split {s['correct_split']}, wrong cuts {s['wrong_cuts']}, missed {s['missed']},"
            f" over-split {s['over_split']}, correctly whole {s['correct_whole']}"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="draw the sample and write the sheet and the labelling page")
    make.add_argument("--raw", type=Path, required=True)
    make.add_argument("--out", type=Path, required=True)
    make.set_defaults(run=_make)
    score = sub.add_parser("score", help="score the splitters against the labelled sheet")
    score.add_argument("--raw", type=Path, required=True)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    score.set_defaults(run=_score)
    agree = sub.add_parser("agree", help="compare two labelled sheets of the same words")
    agree.add_argument("--first", type=Path, required=True)
    agree.add_argument("--second", type=Path, required=True)
    agree.add_argument("--out", type=Path, required=True)
    agree.set_defaults(run=_agree)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
