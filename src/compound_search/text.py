"""Tokenisation. The plain tokeniser is the baseline that every other system is compared to.

Every tokeniser maps a text to a list of index terms. Stems and lemmas are looked
up per word form and cached, so a 4-million-token pool costs one lookup per distinct form.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import simplemma
import snowballstemmer

# A token is a run of letters or digits. Underscores count as separators,
# because GermanDPR titles are written like "Recht_der_Vereinigten_Staaten".
_TOKEN = re.compile(r"[^\W_]+")
_STEMMER = snowballstemmer.stemmer("german")


def raw_tokens(text: str) -> list[str]:
    """Runs of letters or digits, case kept. Lemmatisers need the capital letter of German nouns."""
    return _TOKEN.findall(unicodedata.normalize("NFC", text))


def plain_tokens(text: str) -> list[str]:
    """Lowercase, split on anything that is not a letter or digit.

    No stop words are removed and nothing is stemmed: BM25's IDF already
    down-weights frequent words, and the baseline should be the simplest
    setup a search engine would ship with.
    """
    return [t.lower() for t in raw_tokens(text)]


@lru_cache(maxsize=None)
def stem(word: str) -> str:
    """Snowball German stem of a lowercased word (also folds umlauts and ß)."""
    return _STEMMER.stemWord(word.lower())


@lru_cache(maxsize=None)
def lemma(word: str) -> str:
    """Dictionary lemma of a word form (simplemma, no context), lowercased."""
    return simplemma.lemmatize(word, lang="de").lower()


def stem_tokens(text: str) -> list[str]:
    return [stem(t) for t in raw_tokens(text)]


def lemma_tokens(text: str) -> list[str]:
    return [lemma(t) for t in raw_tokens(text)]
