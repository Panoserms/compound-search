"""Stage 5: LLM query rewriting behind a retrieval-score uncertainty rule.

Rules from HYPOTHESES.md (Sections 4 & 5):
- Fixed local model: `gemma4:e2b-it-qat` in Ollama, temperature 0.
- The routing rule may use retrieval scores and rankings only, never gold passages
  or gold answers. By taking only `Sequence[tuple[str, float]]` rankings, the router
  cannot see gold labels even by accident.
- For queries flagged as uncertain, the LLM generates German search expansion terms
  which are searched and fused with the base hybrid ranking via RRF (k = 60).
  Queries not flagged as uncertain keep their exact base hybrid ranking.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from compound_search.runs import RRF_K

OLLAMA_MODEL = "gemma4:e2b-it-qat"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# Default threshold on top-1 RRF score (k=60 over 2 systems):
# Max possible RRF score when both systems agree on rank 1 is 2 / 61 = 0.032787.
# When a document is rank 1 in one system and missing from top 100 of the other,
# its RRF score is 1 / 61 = 0.016393.
# A threshold of 1/61 + 1/70 = 0.030679 selects queries where no single passage
# sits in the very top ranks of both the lexical and dense retrievers.
DEFAULT_MAX_RRF_TOP1 = (1.0 / (RRF_K + 1)) + (1.0 / (RRF_K + 10))

PROMPT_TEMPLATE = (
    "Du hilfst bei der deutschen Volltextsuche in Wikipedia.\n"
    "Gib zur folgenden Suchanfrage 4 bis 6 präzise deutsche Suchbegriffe, Synonyme "
    "oder zerlegte zusammengesetzte Nomen (Komposita-Bestandteile) in einer einzigen Zeile aus, "
    "getrennt durch Leerzeichen. Keine Erklärungen, keine Aufzählungszeichen, keine Satzzeichen.\n\n"
    "Suchanfrage: {query}\n"
    "Suchbegriffe:"
)

_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


Hits = Sequence[tuple[str, float]]


class Rewriter(Protocol):
    name: str

    def rewrite(self, query_text: str) -> str: ...


def clean_llm_expansion(raw_response: str) -> str:
    """Strip any reasoning tags, punctuation clutter, or multi-line chatter from the LLM output."""
    text = _THINK_RE.sub("", raw_response).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    first_line = lines[-1] if len(lines) > 1 and ":" in lines[0] else lines[0]
    first_line = re.sub(r"^(suchbegriffe|begriffe|keywords|synonyme)\s*:\s*", "", first_line, flags=re.IGNORECASE)
    first_line = re.sub(r"[•*#\-–—,;\"'()\[\]{}?!.]+", " ", first_line)
    tokens = [tok for tok in first_line.split() if len(tok) >= 2]
    return " ".join(tokens[:10])


def expand_query_text(original_query: str, expansion: str) -> str:
    """Combine the original query with the LLM's expansion terms."""
    cleaned = clean_llm_expansion(expansion)
    if not cleaned:
        return original_query
    return f"{original_query} {cleaned}"


def is_uncertain_ranking(
    hybrid_hits: Hits,
    lexical_hits: Hits | None = None,
    dense_hits: Hits | None = None,
    *,
    max_rrf_top1: float = DEFAULT_MAX_RRF_TOP1,
    require_top1_disagreement: bool = True,
) -> bool:
    """Decide whether a query's ranking is uncertain using ONLY retrieval scores and ranks.

    Gold passages and gold answers are never passed to this function.
    A query is routed to the LLM when:
    1. If both lexical and dense rankings are provided and `require_top1_disagreement` is True,
       their rank-1 passages disagree (`lexical_hits[0][0] != dense_hits[0][0]`).
    2. The top-1 fused RRF score is strictly below `max_rrf_top1` (meaning no candidate
       is ranked near the very top by both systems).
    """
    if not hybrid_hits:
        return True
    if require_top1_disagreement and lexical_hits and dense_hits:
        if lexical_hits[0][0] == dense_hits[0][0]:
            return False
    top1_score = hybrid_hits[0][1]
    return top1_score < max_rrf_top1


def select_uncertain_qids(
    hybrid_view: Mapping[str, Hits],
    lexical_view: Mapping[str, Hits] | None = None,
    dense_view: Mapping[str, Hits] | None = None,
    *,
    max_rrf_top1: float = DEFAULT_MAX_RRF_TOP1,
    require_top1_disagreement: bool = True,
) -> set[str]:
    """Return the set of query IDs whose retrieval rankings trigger the uncertainty rule."""
    selected: set[str] = set()
    for qid, h_hits in hybrid_view.items():
        l_hits = lexical_view[qid] if lexical_view is not None and qid in lexical_view else None
        d_hits = dense_view[qid] if dense_view is not None and qid in dense_view else None
        if is_uncertain_ranking(
            h_hits,
            l_hits,
            d_hits,
            max_rrf_top1=max_rrf_top1,
            require_top1_disagreement=require_top1_disagreement,
        ):
            selected.add(qid)
    return selected


class OllamaRewriter:
    """Calls local Ollama (`gemma4:e2b-it-qat`, temperature 0) and caches rewrites on disk."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        prompt_template: str = PROMPT_TEMPLATE,
        url: str = OLLAMA_URL,
        cache_dir: Path | None = None,
        timeout: float = 60.0,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.url = url
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self._cache_file = self._init_cache_file(cache_dir)
        self._cache: dict[str, str] = self._load_cache()

    def _init_cache_file(self, cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None
        h = hashlib.sha1()
        h.update(self.model.encode("utf-8"))
        h.update(b"\x00")
        h.update(self.prompt_template.encode("utf-8"))
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.model)
        return cache_dir / f"rewrites_{slug}_{h.hexdigest()[:10]}.json"

    def _load_cache(self) -> dict[str, str]:
        if self._cache_file is not None and self._cache_file.exists():
            return json.loads(self._cache_file.read_text(encoding="utf-8"))
        return {}

    def _save_cache(self) -> None:
        if self._cache_file is not None:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(
                json.dumps(self._cache, indent=1, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def rewrite(self, query_text: str) -> str:
        if query_text in self._cache:
            return expand_query_text(query_text, self._cache[query_text])
        prompt = self.prompt_template.format(query=query_text)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 48,
                "seed": 13,
            },
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        raw = str(body.get("response", "")).strip()
        self._cache[query_text] = raw
        self._save_cache()
        return expand_query_text(query_text, raw)
