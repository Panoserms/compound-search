"""A lexicon-based splitter for German noun compounds, with linking elements.

Method: Koehn & Knight (2003). A word may be cut into parts that are known
words; the cut with the highest geometric mean of part frequencies wins, and
the uncut word competes with its own frequency. So a frequent, lexicalised
compound ("Bahnhof") stays whole, while a rare or unseen one is split.

Additions for German:
- Linking elements between the parts (Bundes|gericht, Sonnen|schein, Verfassungs|gericht),
  each allowed only after the sounds it really follows (no "Energie-er").
- A dropped final -e (Geschichts|buch from Geschichte), tried only when no other reading fits.
- Frequencies are pooled over the inflected forms of a word (same Snowball stem),
  so a rare form like "Unfällen" counts as the frequent noun "Unfall".
- Parts must be noun-like, which keeps function words and prefixes out
  ("vor" + "der" + "Grund"). Only noun + noun compounds are split.
- Suffixes that also exist as nouns ("Schaft", "Tum") are never parts.

The lexicon comes from the passage pool itself, so no outside word list is needed.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from compound_search.text import stem

MIN_WORD = 6  # shorter words are never split
MIN_PART = 3  # shortest part, so that Tag, Rat, See, Tor can be parts
MIN_FREQ = 5  # a part (with its inflected forms) must occur at least this often in the pool
MAX_PARTS = 4
LINKS = ("s", "es", "n", "en", "er", "e", "ens", "ns")
DROPPED_E = "-e"
DROPPED_E_PLUS_S = "-e+s"
# Derivational suffixes that also occur as capitalised words.
SUFFIXES = frozenset(
    {"schaft", "schaften", "heit", "keit", "tum", "ung", "ungen", "lich", "bar", "los", "haft", "isch", "chen", "lein", "ling", "nis", "sal", "weise"}
)
_SUFFIX_STEMS = frozenset(stem(s) for s in SUFFIXES)
_VOWELS = frozenset("aeiouäöüy")

_WORD = re.compile(r"[^\W\d_]+")
_SENTENCE_BREAK = re.compile(r"[.!?:;\"'„“”‚‘’»«()\[\]–—]")


def link_allowed(base: str, link: str) -> bool:
    """Where each linking element occurs: Kind-er, Bund-es, Hund-e, Herz-ens after a consonant;
    Sonne-n after -e, -el, -er; Name-ns after -e; Frau-en, Student-en not after -e;
    Arbeit-s, Liebe-s anywhere except after an s sound."""
    last = base[-1]
    if link in ("es", "er", "e", "ens"):
        return last not in _VOWELS
    if link == "en":
        return last != "e"
    if link == "n":
        return base.endswith(("e", "el", "er"))
    if link == "ns":
        return last == "e"
    if link == "s":
        return last not in "sßzx"
    return False


@dataclass(frozen=True)
class Part:
    surface: str  # the stretch of the word, including its linking element
    base: str  # the word behind it; this is what gets indexed
    link: str  # linking element after the base; "-e" or "-e+s" for a dropped e; "" for none


@dataclass
class Lexicon:
    freq: dict[str, int]  # lowercased form -> count
    nouns: set[str]  # forms that are mostly capitalised inside sentences

    def __post_init__(self) -> None:
        self.stem_freq: Counter[str] = Counter()
        for form, n in self.freq.items():
            self.stem_freq[stem(form)] += n
        self.noun_stems: set[str] = {stem(form) for form in self.nouns}

    def pooled_freq(self, form: str) -> int:
        """Frequency of a form together with all forms that share its stem."""
        return self.stem_freq.get(stem(form), 0)

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_noun_evidence: int = 2) -> "Lexicon":
        """Count word forms. A form is noun-like if, inside sentences, it is capitalised at
        least as often as not. Sentence starts, the title line and heading lines are no
        evidence, because everything is capitalised there."""
        freq: Counter[str] = Counter()
        upper: Counter[str] = Counter()
        lower: Counter[str] = Counter()
        for text in texts:
            lines = unicodedata.normalize("NFC", text).split("\n")
            for line_no, line in enumerate(lines):
                evidence_line = line_no > 0 and not line.lstrip().startswith("=")
                previous_end = None
                for match in _WORD.finditer(line):
                    word = match.group()
                    form = word.lower()
                    freq[form] += 1
                    inside = previous_end is not None and not _SENTENCE_BREAK.search(line[previous_end : match.start()])
                    if evidence_line and inside:
                        (upper if word[0].isupper() else lower)[form] += 1
                    previous_end = match.end()
        nouns = {form for form, n in upper.items() if n >= min_noun_evidence and n >= lower[form]}
        return cls(dict(freq), nouns)


class LexiconSplitter:
    name = "lexicon"

    def __init__(self, lexicon: Lexicon, min_part: int = MIN_PART, min_freq: int = MIN_FREQ, max_parts: int = MAX_PARTS):
        self.lexicon = lexicon
        self.min_part = min_part
        self.min_freq = min_freq
        self.max_parts = max_parts
        self._cache: dict[str, list[Part]] = {}

    def _is_modifier(self, form: str) -> bool:
        """A non-final part must itself be a noun form seen in the pool."""
        return (
            len(form) >= self.min_part
            and form in self.lexicon.nouns
            and form not in SUFFIXES
            and self.lexicon.pooled_freq(form) >= self.min_freq
        )

    def _is_head(self, form: str) -> bool:
        """The final part may be any inflected form of a noun, also one never seen in the pool."""
        form_stem = stem(form)
        return (
            len(form) >= self.min_part
            and form_stem in self.lexicon.noun_stems
            and form not in SUFFIXES
            and form_stem not in _SUFFIX_STEMS
            and self.lexicon.stem_freq[form_stem] >= self.min_freq
        )

    def _modifier(self, segment: str) -> Part | None:
        """The best reading of a non-final stretch: a noun, or a noun plus linking element.
        Only if neither fits: a noun that lost its final -e."""
        readings = []
        if self._is_modifier(segment):
            readings.append(Part(segment, segment, ""))
        for link in LINKS:
            base = segment[: -len(link)]
            if segment.endswith(link) and self._is_modifier(base) and link_allowed(base, link):
                readings.append(Part(segment, base, link))
        if not readings:
            if self._is_modifier(segment + "e"):
                readings.append(Part(segment, segment + "e", DROPPED_E))
            elif segment.endswith("s") and self._is_modifier(segment[:-1] + "e") and link_allowed(segment[:-1], "s"):
                readings.append(Part(segment, segment[:-1] + "e", DROPPED_E_PLUS_S))
        if not readings:
            return None
        # The most frequent base form wins; max() keeps the first on a tie, so "no linking element" wins.
        return max(readings, key=lambda p: self.lexicon.freq.get(p.base, 0))

    def _segmentations(self, word: str, start: int, parts_left: int) -> Iterator[list[Part]]:
        rest = word[start:]
        if self._is_head(rest):
            yield [Part(rest, rest, "")]
        if parts_left <= 1:
            return
        for end in range(start + self.min_part, len(word) - self.min_part + 1):
            modifier = self._modifier(word[start:end])
            if modifier is None:
                continue
            for tail in self._segmentations(word, end, parts_left - 1):
                yield [modifier, *tail]

    def analyse(self, word: str) -> list[Part]:
        """The parts of a word. A word that is not split comes back as one part."""
        form = unicodedata.normalize("NFC", word).lower()
        if form in self._cache:
            return self._cache[form]
        best = [Part(form, form, "")]
        if len(form) >= MIN_WORD and form.isalpha():
            best_score = float(self.lexicon.pooled_freq(form))
            for parts in self._segmentations(form, 0, self.max_parts):
                if len(parts) < 2:
                    continue
                score = math.exp(sum(math.log(self.lexicon.pooled_freq(p.base)) for p in parts) / len(parts))
                if score > best_score:
                    best, best_score = parts, score
        self._cache[form] = best
        return best

    def terms(self, word: str) -> list[str]:
        """Index terms for a word: the whole word, followed by its parts if it was split."""
        parts = self.analyse(word)
        whole = "".join(p.surface for p in parts)
        return [whole] if len(parts) == 1 else [whole, *(p.base for p in parts)]

    def boundaries(self, word: str) -> tuple[int, ...]:
        """Character offsets of the cuts, e.g. (6, 17) for Bundes|verfassungs|gericht."""
        offsets, position = [], 0
        for part in self.analyse(word)[:-1]:
            position += len(part.surface)
            offsets.append(position)
        return tuple(offsets)


class HantaSplitter:
    """Compound parts from the morphological analysis of HanTa, an open tagger and analyser for German (Wartena, 2019).

    HanTa cuts a word into morphemes with tags. A word counts as a compound if it has
    two or more content morphemes (noun, name, adjective, verb or foreign stem). Linking
    elements, endings, prefixes and verb particles are left out.
    """

    name = "hanta"
    _CONTENT = ("NN", "NE", "ADJ", "VV", "FM")

    def __init__(self) -> None:
        from HanTa import HanoverTagger

        self._tagger = HanoverTagger.HanoverTagger("morphmodel_ger.pgz")
        self._cache: dict[str, tuple[list[str], tuple[int, ...]]] = {}

    def _analyse(self, word: str) -> tuple[list[str], tuple[int, ...]]:
        form = unicodedata.normalize("NFC", word)
        if form not in self._cache:
            parts: list[str] = []
            starts: list[int] = []
            if len(form) >= MIN_WORD and form.isalpha():
                position = 0
                for morpheme, tag in self._tagger.analyze(form, taglevel=3)[1]:
                    if tag.startswith(self._CONTENT):
                        parts.append(morpheme)
                        starts.append(position)
                    position += len(morpheme)
            if len(parts) < 2:
                parts, starts = [], []
            self._cache[form] = (parts, tuple(starts[1:]))
        return self._cache[form]

    def terms(self, word: str) -> list[str]:
        return [word.lower(), *self._analyse(word)[0]]

    def boundaries(self, word: str) -> tuple[int, ...]:
        return self._analyse(word)[1]


def make_split_tokenizer(splitter, normalise, keep_whole: bool = True):
    """A tokeniser that splits capitalised words (German nouns) with the given splitter.

    normalise turns one cased word into its index term: str.lower, text.stem or text.lemma.
    Parts are handed over capitalised, because they are nouns and the lemmatiser uses that.
    keep_whole=True indexes the whole word and its parts; False indexes the parts only.
    """
    from compound_search.text import raw_tokens

    def tokenize(text: str) -> list[str]:
        terms: list[str] = []
        for token in raw_tokens(text):
            parts = splitter.terms(token)[1:] if token[0].isupper() else []
            if keep_whole or not parts:
                terms.append(normalise(token))
            terms.extend(normalise(part.capitalize()) for part in parts)
        return terms

    return tokenize
