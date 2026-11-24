"""Write the data file behind the project's web page.

    python -I -m compound_search.export_site --raw data/raw --results results \\
        --out site/compound_search_study.json

Everything in the file comes from the saved reports under results/ or is computed
here from the data with the study's own tokenisers and splitter. Nothing is typed in.
The test questions are not searched: test numbers are read from results/test_summary.json.

The page has a "search bench" with three switches (stems, split compounds, dense model).
Each of the eight switch positions is one system of the study, run on the dev set:

    stems  split  dense   system
      -      -      -     bm25_plain
      x      -      -     bm25_stem
      -      x      -     bm25_parts           (added for the page, dev only)
      x      x      -     bm25_parts_stem
      -      -      x     hybrid_plain_dense
      x      -      x     hybrid_stem_dense
      -      x      x     hybrid_parts_dense   (added for the page, dev only)
      x      x      x     hybrid_parts_stem_dense

For the example questions the file also says which words of the question and of the
gold passage share an index term under each lexical setting, so the page can show
what the search engine actually matches. The export checks its own word-by-word
terms against the real tokenisers and stops if they ever differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from compound_search.data import Passage, Query, build_task, load_germandpr, split_dev
from compound_search.labels import AI_EVAL, load_splitter_eval
from compound_search.runs import DROPPED, SHIPPED
from compound_search.splitter import Lexicon, LexiconSplitter, make_split_tokenizer
from compound_search.tables import LABELS, PREDICTIONS, prediction_outcome
from compound_search.text import plain_tokens, stem, stem_tokens

# switches "stems split dense" -> system
CONFIGS: dict[str, str] = {
    "000": "bm25_plain",
    "100": "bm25_stem",
    "010": "bm25_parts",
    "110": "bm25_parts_stem",
    "001": "hybrid_plain_dense",
    "101": "hybrid_stem_dense",
    "011": "hybrid_parts_dense",
    "111": "hybrid_parts_stem_dense",
}
# The four lexical settings behind the first two switches. "Split" always means the same thing:
# a compound is replaced by its parts, with or without stems.
MODES = ("plain", "stem", "parts", "parts_stem")
REFERENCE_SYSTEMS = ("dense", "hybrid_parts_stem_dense_rewrite")

# Dev questions shown on the bench. Picked from the error analysis and the dev reports:
# three where the preprocessing helps and two where compound splitting hurts.
EXAMPLES = ("train-00668", "train-01644", "train-06045", "train-00463", "train-03241")

# Words for the "how a word is cut" view: right cuts, words left whole, and wrong cuts.
SPLITTER_WORDS = (
    "Einwohnerdichte",
    "Bundesverfassungsgericht",
    "Geschichtsbuch",
    "Musikhochschule",
    "Bahnhof",
    "Jahrhundert",
    "Straßburg",
    "Deutschland",
    "Newell",
    "Insektensterben",
)

FAMILY = {"bm25": "lexical", "dense": "dense", "hybrid": "hybrid"}
_WORD = re.compile(r"[^\W_]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ„\"(])")
MAX_EXCERPT_WORDS = 60


def family(system: str) -> str:
    return FAMILY[system.split("_")[0]]


def lucene_idf(df: int, n_docs: int) -> float:
    """The IDF that bm25s uses with method='lucene'."""
    return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))


# --- word by word: the index terms of one word under each lexical setting -----------------------


def word_terms(token: str, mode: str, splitter: LexiconSplitter) -> list[str]:
    """The index terms one word produces. Mirrors text.py and splitter.make_split_tokenizer."""
    if mode == "plain":
        return [token.lower()]
    if mode == "stem":
        return [stem(token)]
    parts = splitter.terms(token)[1:] if token[0].isupper() else []
    if mode == "parts":
        return [part.capitalize().lower() for part in parts] if parts else [token.lower()]
    if mode == "parts_stem":
        return [stem(part.capitalize()) for part in parts] if parts else [stem(token)]
    raise ValueError(mode)


def real_tokenizer(mode: str, splitter: LexiconSplitter) -> Callable[[str], list[str]]:
    return {
        "plain": plain_tokens,
        "stem": stem_tokens,
        "parts": make_split_tokenizer(splitter, str.lower, keep_whole=False),
        "parts_stem": make_split_tokenizer(splitter, stem, keep_whole=False),
    }[mode]


def segments(text: str) -> list[tuple[str, str]]:
    """(word, what follows it up to the next word). Text before the first word is dropped."""
    text = unicodedata.normalize("NFC", text)
    matches = list(_WORD.finditer(text))
    out = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(), text[m.end() : end]))
    return out


def cuts_of(token: str, splitter: LexiconSplitter) -> list[int]:
    """Where the splitter cuts a word, as character offsets. Only capitalised words are split."""
    return list(splitter.boundaries(token)) if token[0].isupper() else []


def annotate(
    text: str,
    other_terms: Mapping[str, set[str]],
    idf: Mapping[str, Mapping[str, float]],
    splitter: LexiconSplitter,
) -> list[dict]:
    """Every word of a text with its cuts and, per lexical setting, which of its terms the other side has.

    "m" maps a setting to one number per index term of the word, in tokeniser order:
    0 if the other text lacks the term, else the term's IDF (how much a match is worth).
    """
    words = []
    for token, gap in segments(text):
        marks = {}
        for mode in MODES:
            marks[mode] = [
                round(idf[mode].get(term, 0.0), 2) if term in other_terms[mode] else 0
                for term in word_terms(token, mode, splitter)
            ]
        words.append({"t": token, "gap": gap, "cuts": cuts_of(token, splitter), "m": marks})
    return words


def body_of(passage: Passage) -> str:
    """The passage as running text: without its first line (the article title), and with
    wiki section headings such as "=== Befehlskette ===" written as "Befehlskette:"."""
    lines = [line.strip() for line in unicodedata.normalize("NFC", passage.text).split("\n")]
    out = []
    for line in lines[1:]:
        if not line:
            continue
        if line.startswith("="):
            heading = line.strip("= ").strip()
            if heading:
                out.append(heading + ":")
        else:
            out.append(line)
    return " ".join(out) or " ".join(line for line in lines if line)


def pick_excerpt(body: str, query_terms: set[str], idf: Mapping[str, float], splitter: LexiconSplitter) -> tuple[str, bool]:
    """Up to two sentences of the passage that carry the most valuable matches (richest setting).

    Returns the excerpt and whether anything was left out.
    """

    def matched(sentence: str) -> set[str]:
        terms = {t for token, _ in segments(sentence) for t in word_terms(token, "parts_stem", splitter)}
        return terms & query_terms

    sentences = [s for s in _SENTENCE_END.split(body) if s.strip()]
    chosen: list[int] = []
    covered: set[str] = set()
    for _ in range(2):
        best, best_gain = None, 0.0
        for i, sentence in enumerate(sentences):
            if i in chosen:
                continue
            gain = sum(idf.get(t, 0.0) for t in matched(sentence) - covered)
            if gain > best_gain:
                best, best_gain = i, gain
        if best is None:
            break
        chosen.append(best)
        covered |= matched(sentences[best])
    if not chosen:
        chosen = [0]
    picked = [sentences[i] for i in sorted(chosen)]
    # A very long sentence is cut to the window of words with the most valuable matches.
    trimmed = False
    budget = MAX_EXCERPT_WORDS // len(picked)
    out = []
    for sentence in picked:
        segs = segments(sentence)
        if len(segs) <= budget:
            out.append(sentence.strip())
            continue
        trimmed = True
        worth = [
            sum(idf.get(t, 0.0) for t in set(word_terms(token, "parts_stem", splitter)) & query_terms) for token, _ in segs
        ]
        start = max(range(len(segs) - budget + 1), key=lambda i: (sum(worth[i : i + budget]), -i))
        window = "".join(token + gap for token, gap in segs[start : start + budget]).strip()
        out.append(("… " if start > 0 else "") + window + (" …" if start + budget < len(segs) else ""))
    text = " ".join(out) if sorted(chosen) == list(range(min(chosen), max(chosen) + 1)) else " … ".join(out)
    return text, trimmed or len(picked) < len(sentences)


# --- the blocks of the file ------------------------------------------------------------------------


def _summary(block: dict) -> dict:
    return {
        "r1": block["recall@1"],
        "r5": block["recall@5"],
        "r10": block["recall@10"],
        "r10_ci": block["recall@10_ci95"],
        "r100": block["recall@100"],
        "mrr": block["mrr@10"],
        "mrr_ci": block["mrr@10_ci95"],
    }


def systems_block(dev: Mapping[str, dict], test_summary: dict) -> dict:
    out = {}
    for name, report in dev.items():
        tested = test_summary["systems"].get(name)
        out[name] = {
            "label": LABELS[name],
            "family": family(name),
            "llm": name.endswith("_rewrite"),
            # Wall-clock time of the system's dev run on the study laptop (one run; index build, search and statistics).
            "dev_seconds": report.get("seconds"),
            "dev": {pool: _summary(report["pools"][pool]["summary"]) for pool in (DROPPED, SHIPPED)},
            "test": {pool: _summary(tested[pool]) for pool in (DROPPED, SHIPPED)} if tested else None,
        }
    return out


def hypotheses_block(test_summary: dict) -> dict:
    out = {}
    for pool in (DROPPED, SHIPPED):
        rows = []
        for r in test_summary["hypotheses"][pool]:
            text, lo, hi = PREDICTIONS[r["id"]]
            rows.append(
                {
                    "id": r["id"],
                    "claim": r["claim"],
                    "system": r["system"],
                    "baseline": r["baseline"],
                    "predicted": text,
                    "predicted_lo": lo,
                    "predicted_hi": hi,
                    "diff": r["recall@10_diff"],
                    "ci": r["recall@10_ci95"],
                    "p_raw": r["recall@10_p_raw"],
                    "p_holm": r["recall@10_p_holm"],
                    "gained": r["gained"],
                    "lost": r["lost"],
                    "mrr_diff": r["mrr@10_diff"],
                    "mrr_p": r["mrr@10_p_raw"],
                    "supported": r["supported"],
                    "outcome": prediction_outcome(r["id"], r),
                }
            )
        out[pool] = rows
    return out


def comparison(reports: Mapping[str, dict], a: str, b: str, pool: str = DROPPED) -> dict:
    """The saved paired comparison a minus b."""
    if b in reports[a]["pools"][pool]["versus"]:
        c, sign = reports[a]["pools"][pool]["versus"][b], 1.0
    else:
        c, sign = reports[b]["pools"][pool]["versus"][a], -1.0
    r, m = c["recall@10"], c["rr@10"]
    lo, hi = sorted(sign * x for x in r["ci95"])
    return {
        "a": a,
        "b": b,
        "diff": sign * r["diff"],
        "ci": [lo, hi],
        "p": r["p"],
        "gained": r["gained"] if sign > 0 else r["lost"],
        "lost": r["lost"] if sign > 0 else r["gained"],
        "mrr_diff": sign * m["diff"],
        "mrr_p": m["p"],
    }


def bench_block(
    dev_queries: Sequence[Query],
    passages: Mapping[str, Passage],
    dev: Mapping[str, dict],
    splitter: LexiconSplitter,
    idf: Mapping[str, Mapping[str, float]],
    rewrites: Mapping[str, str],
) -> dict:
    by_id = {q.qid: q for q in dev_queries}
    tokenizers = {mode: real_tokenizer(mode, splitter) for mode in MODES}
    ranked_systems = (*CONFIGS.values(), *REFERENCE_SYSTEMS)
    examples = []
    for qid in EXAMPLES:
        query = by_id[qid]
        gold = passages[query.gold[0]]
        q_terms = {mode: set(tokenizers[mode](query.text)) for mode in MODES}
        p_terms = {mode: set(tokenizers[mode](gold.text)) for mode in MODES}
        excerpt, shortened = pick_excerpt(body_of(gold), q_terms["parts_stem"], idf["parts_stem"], splitter)

        # The word-by-word terms must be exactly what the real tokenisers produce.
        for mode in MODES:
            for text in (query.text, excerpt):
                mine = [t for token, _ in segments(text) for t in word_terms(token, mode, splitter)]
                if mine != tokenizers[mode](text):
                    raise RuntimeError(f"{qid} {mode}: word terms differ from the tokeniser")

        examples.append(
            {
                "qid": qid,
                "question": [*annotate(query.text, p_terms, idf, splitter)],
                "question_text": query.text.strip(),
                "answer": query.answers[0].strip() if query.answers else "",
                "gold_title": " ".join(gold.title.replace("_", " ").split()),
                "excerpt": annotate(excerpt, q_terms, idf, splitter),
                "excerpt_shortened": shortened,
                "passage_words": len(segments(body_of(gold))),
                # Shared index terms, the most valuable (rarest) first.
                "shared_terms": {
                    mode: [
                        {"term": term, "idf": round(idf[mode].get(term, 0.0), 2)}
                        for term in sorted(q_terms[mode] & p_terms[mode], key=lambda t: (-idf[mode].get(t, 0.0), t))
                    ]
                    for mode in MODES
                },
                "ranks": {name: dev[name]["pools"][DROPPED]["gold_rank"][qid] for name in ranked_systems},
                "llm_expansion": rewrites.get(query.text),
            }
        )
    return {
        "configs": CONFIGS,
        "modes": {"00": "plain", "10": "stem", "01": "parts", "11": "parts_stem"},
        "pool": DROPPED,
        "split": "dev",
        "examples": examples,
    }


def best_cut(word: str, lexicon: Lexicon, splitter: LexiconSplitter) -> tuple[list, float] | None:
    """The best-scoring cut of a word, whether or not it beats the whole word. None if no cut is possible."""
    form = unicodedata.normalize("NFC", word).lower()
    best = None
    for parts in splitter._segmentations(form, 0, splitter.max_parts):
        if len(parts) < 2:
            continue
        score = math.exp(sum(math.log(lexicon.pooled_freq(p.base)) for p in parts) / len(parts))
        if best is None or score > best[1]:
            best = (parts, score)
    return best


def splitter_block(lexicon: Lexicon, splitter: LexiconSplitter, evaluation: tuple[dict, str] | None) -> dict:
    def describe(parts) -> list[dict]:
        return [{"surface": p.surface, "base": p.base, "link": p.link, "freq": lexicon.pooled_freq(p.base)} for p in parts]

    words = []
    for word in SPLITTER_WORDS:
        parts = splitter.analyse(word)
        split = len(parts) > 1
        candidate = best_cut(word, lexicon, splitter)
        if split and (candidate is None or [p.surface for p in candidate[0]] != [p.surface for p in parts]):
            raise RuntimeError(f"{word}: the best cut is not the cut the splitter made")
        words.append(
            {
                "word": word,
                "whole_freq": lexicon.pooled_freq(word.lower()),
                "split": split,
                "cuts": list(splitter.boundaries(word)),
                "score": candidate[1] if split else None,
                "parts": describe(parts) if split else [],
                # For a word that stays whole: the best cut it was compared with, if there is one.
                "rejected": None if split or candidate is None else {"parts": describe(candidate[0]), "score": candidate[1]},
            }
        )
    return {
        "n_forms": len(lexicon.freq),
        "n_nouns": len(lexicon.nouns),
        "words": words,
        "eval": eval_block(*evaluation) if evaluation else None,
    }


def eval_block(evaluation: dict, labelled_by: str) -> dict:
    """The splitter scores for the page. labelled_by is "person" or "ai" (see labels.load_splitter_eval)."""

    def cut(word: str, offsets: Sequence[int]) -> str:
        pieces, prev = [], 0
        for c in offsets:
            pieces.append(word[prev:c])
            prev = c
        return "|".join([*pieces, word[prev:]])

    scores = {}
    for name, s in evaluation["splitters"].items():
        errors = [
            {"word": w, "gold": cut(w, v["gold"]), "predicted": cut(w, v["predicted"]), "type": v["type"]}
            for w, v in s["per_word"].items()
            if v["gold"] != v["predicted"]
        ]
        scores[name] = {**{k: s["all"][k] for k in s["all"]}, "errors": errors}
    return {"n": evaluation["n_labelled"], "types": evaluation["types"], "labelled_by": labelled_by, "splitters": scores}


def ai_comparison(results: Path) -> dict | None:
    """How the person's labels compare with the earlier AI labels, and what the splitters score on the AI labels."""
    agreement_file, ai_file = results / "label_agreement.json", results / AI_EVAL
    if not (agreement_file.exists() and ai_file.exists()):
        return None
    agreement = json.loads(agreement_file.read_text(encoding="utf-8"))
    ai = json.loads(ai_file.read_text(encoding="utf-8"))
    return {
        "n": agreement["n"],
        "same_cuts": agreement["same_cuts"],
        "cut_only_by_the_person": len(agreement["only_first_cuts"]),
        "cut_only_by_the_ai": len(agreement["only_second_cuts"]),
        "cut_differently": len(agreement["both_cut_differently"]),
        "f1_on_ai_labels": {name: s["all"]["f1"] for name, s in ai["splitters"].items()},
    }


def first_pass(results: Path) -> dict | None:
    """The splitter scores on the person's labels before their one revision, if a first pass was kept."""
    score_file, agreement_file = results / "splitter_eval_first_pass.json", results / "label_agreement_first_pass.json"
    if not (score_file.exists() and agreement_file.exists()):
        return None
    scores = json.loads(score_file.read_text(encoding="utf-8"))
    agreement = json.loads(agreement_file.read_text(encoding="utf-8"))
    return {"f1": {name: s["all"]["f1"] for name, s in scores["splitters"].items()}, "same_cuts_with_ai": agreement["same_cuts"]}


def document_frequencies(passages: Mapping[str, Passage], splitter: LexiconSplitter) -> dict[str, Counter]:
    """In how many passages each index term occurs, per lexical setting."""
    tokenizers = {mode: real_tokenizer(mode, splitter) for mode in MODES}
    df: dict[str, Counter] = {mode: Counter() for mode in MODES}
    for passage in passages.values():
        for mode in MODES:
            df[mode].update(set(tokenizers[mode](passage.text)))
    return df


def build(raw: Path, results: Path, cache: Path) -> dict:
    passages, queries = build_task(load_germandpr(raw))
    dev_queries = split_dev(queries["train"])
    dev = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((results / "dev").glob("*.json"))}
    test_reports = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((results / "test").glob("*.json"))}
    test_summary = json.loads((results / "test_summary.json").read_text(encoding="utf-8"))
    stage0 = json.loads((results / "stage0_baseline.json").read_text(encoding="utf-8"))
    errors = json.loads((results / "error_analysis_50.json").read_text(encoding="utf-8"))
    residual = json.loads((results / "residual_check.json").read_text(encoding="utf-8"))
    evaluation = load_splitter_eval(results)  # (scores, "person" or "ai")
    speed = json.loads((results / "splitter_speed.json").read_text(encoding="utf-8"))

    lexicon = Lexicon.from_texts(p.text for p in passages.values())
    splitter = LexiconSplitter(lexicon)
    df = document_frequencies(passages, splitter)
    idf = {mode: {term: lucene_idf(n, len(passages)) for term, n in df[mode].items()} for mode in MODES}

    rewrites: dict[str, str] = {}
    for path in sorted(cache.glob("rewrites_*.json")):
        rewrites.update(json.loads(path.read_text(encoding="utf-8")))
    test_texts = {q.text for q in queries["test"]}
    dev_rewrite = dev["hybrid_parts_stem_dense_rewrite"]
    test_routing = test_reports["hybrid_parts_stem_dense_rewrite"].get("routing")

    return {
        "fixture": False,
        "source": "compound_search.export_site",
        "counts": {
            "passages": len(passages),
            "dev_questions": len(dev_queries),
            "dev_gold_passages": dev["bm25_plain"]["pools"][DROPPED]["summary"]["n_gold_passages"],
            "test_questions": test_summary["n_queries"],
            "test_gold_passages": test_summary["n_gold_passages"],
        },
        "systems": systems_block(dev, test_summary),
        "hypotheses": hypotheses_block(test_summary),
        "comparisons": {
            "lexical_gain": comparison(test_reports, "bm25_parts_stem", "bm25_plain"),
            "dense_vs_plain_hybrid": comparison(test_reports, "dense", "hybrid_plain_dense"),
            "final_vs_dense": comparison(test_reports, "hybrid_parts_stem_dense_rewrite", "dense"),
            "hybrid_vs_dense_as_shipped": comparison(test_reports, "hybrid_parts_stem_dense", "dense", SHIPPED),
            "dev_hybrid_vs_dense": comparison(dev, "hybrid_parts_stem_dense", "dense"),
            "dev_rewrite": comparison(dev, "hybrid_parts_stem_dense_rewrite", "hybrid_parts_stem_dense"),
        },
        "bench": bench_block(dev_queries, passages, dev, splitter, idf, rewrites),
        "splitter": splitter_block(lexicon, splitter, evaluation),
        "routing": {
            "dev": {pool: {"n": dev_rewrite["routing"][pool]["n_routed"], "share": dev_rewrite["routing"][pool]["share"]} for pool in (DROPPED, SHIPPED)},
            "test_routed_in_either_pool": sum(1 for text in rewrites if text in test_texts),
            # Per pool, from the repeat of the frozen test run (the first run did not record it).
            "test": {pool: {"n": test_routing[pool]["n_routed"], "share": test_routing[pool]["share"]} for pool in (DROPPED, SHIPPED)}
            if test_routing
            else None,
        },
        "splitter_speed": speed,
        # Only when the reported score rests on a person's labels: the comparison with the earlier AI labels.
        "label_comparison": ai_comparison(results) if evaluation and evaluation[1] == "person" else None,
        # The person's labels were revised once after the scores were known; these are the scores before that.
        "label_first_pass": first_pass(results) if evaluation and evaluation[1] == "person" else None,
        "error_facts": errors["facts_over_all_matching_queries"],
        "loss_cut_words": errors["splitting_losses_cut_words"],
        "hard_negatives": stage0["dev"]["as_shipped"]["own_hard_negatives"],
        "error_categories": {k: v["total_matching_in_dev"] for k, v in errors["categories"].items()},
        "residual": {
            "n": residual["n_residual"],
            "with_answer_string": residual["n_residual_with_answer_string_in_final_top10"],
        },
        "headline_sentence": test_summary["headline"]["sentence_dense"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--cache", type=Path, default=None, help="folder with rewrites_*.json (default: next to --raw)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    data = build(args.raw, args.results, args.cache or args.raw.parent / "cache")
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB): {len(data['bench']['examples'])} examples, {len(data['systems'])} systems")


if __name__ == "__main__":
    main()
