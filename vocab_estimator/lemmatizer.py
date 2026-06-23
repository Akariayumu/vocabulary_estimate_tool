"""Lemma normalization for English words.

The primary implementation uses spaCy's ``en_core_web_sm`` model. If spaCy or
the model is unavailable, the module falls back to a conservative rule-based
normalizer so the estimator remains runnable in constrained environments.
"""

from __future__ import annotations

import re
from functools import lru_cache

WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


class Lemmatizer:
    """Normalize inflected English word forms to lemmas.

    Proper nouns, numeric strings and likely abbreviations are deliberately not
    normalized because merging them would distort vocabulary-size estimates.
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = self._load_spacy(model_name)

    @staticmethod
    def _load_spacy(model_name: str):
        try:
            import spacy

            return spacy.load(model_name, disable=["parser", "ner"])
        except Exception:
            return None

    @staticmethod
    def is_abbreviation(word: str) -> bool:
        """Return True for likely abbreviations/acronyms such as ``NASA``."""

        stripped = word.replace(".", "")
        return (
            len(stripped) >= 2
            and stripped.isupper()
            and any(ch.isalpha() for ch in stripped)
        )

    @staticmethod
    def is_proper_like(word: str) -> bool:
        """Return True for capitalized words that should not be merged."""

        return len(word) > 1 and word[0].isupper() and word[1:].islower()

    @staticmethod
    def is_valid_word(word: str) -> bool:
        """Return True if ``word`` is an alphabetic English token."""

        return bool(WORD_RE.match(word)) and not any(ch.isdigit() for ch in word)

    def should_preserve(self, word: str) -> bool:
        """Return True when a token should bypass lemmatization."""

        if not word or any(ch.isdigit() for ch in word):
            return True
        if self.is_abbreviation(word):
            return True
        if self.is_proper_like(word):
            return True
        return False

    @lru_cache(maxsize=100_000)
    def normalize(self, word: str) -> str:
        """Return a normalized lemma for ``word``.

        Examples:
            ``running`` -> ``run``; ``cars`` -> ``car``; ``NASA`` -> ``NASA``.
        """

        word = word.strip()
        if not word:
            return ""
        if self.should_preserve(word):
            return word

        lower = word.lower()
        if self._nlp is not None:
            doc = self._nlp(lower)
            if doc and doc[0].lemma_ and doc[0].lemma_ != "-PRON-":
                return doc[0].lemma_.lower()

        return self._rule_based_lemma(lower)

    @staticmethod
    def _rule_based_lemma(word: str) -> str:
        """Conservative fallback lemmatizer for common inflections."""

        irregular = {
            "children": "child",
            "men": "man",
            "women": "woman",
            "people": "person",
            "mice": "mouse",
            "feet": "foot",
            "teeth": "tooth",
            "went": "go",
            "gone": "go",
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
        }
        if word in irregular:
            return irregular[word]

        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ing"):
            base = word[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            if base.endswith("mak"):
                return base + "e"
            return base
        if len(word) > 4 and word.endswith("ed"):
            base = word[:-2]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            if base.endswith("iz"):
                return base + "e"
            if not base.endswith("e"):
                return base
            return base
        if len(word) > 4 and word.endswith("es"):
            if word.endswith(("ses", "xes", "zes", "ches", "shes")):
                return word[:-2]
        if (
            len(word) > 3
            and word.endswith("s")
            and not word.endswith(("ss", "us", "is", "ous"))
        ):
            return word[:-1]
        return word

    def is_same_lemma(self, w1: str, w2: str) -> bool:
        """Return True if two surface forms normalize to the same lemma."""

        return self.normalize(w1) == self.normalize(w2)
