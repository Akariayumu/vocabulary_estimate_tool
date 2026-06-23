"""Document coverage analysis based on lemma-frequency ranks."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_CONFIG, EstimatorConfig
from .lemmatizer import Lemmatizer
from .vocab_bank import VocabBank

TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


class DocumentCoverageAnalyzer:
    """Measure how much of a document is covered by top-N known word ranks."""

    def __init__(
        self,
        vocab_bank: VocabBank,
        lemmatizer: Lemmatizer | None = None,
        config: EstimatorConfig = DEFAULT_CONFIG,
    ) -> None:
        self.vocab_bank = vocab_bank
        self.lemmatizer = lemmatizer or vocab_bank.lemmatizer
        self.config = config

    def analyze_paths(self, paths: Iterable[str | Path]) -> dict:
        """Analyze several documents and return per-file plus aggregate coverage."""

        per_file = {}
        all_tokens: list[str] = []
        for path in paths:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            tokens = self.extract_lemmas(text)
            per_file[str(path)] = self.coverage_from_lemmas(tokens)
            all_tokens.extend(tokens)
        return {
            "files": per_file,
            "aggregate": self.coverage_from_lemmas(all_tokens),
        }

    def analyze_text(self, text: str) -> dict:
        """Analyze a raw document string."""

        return self.coverage_from_lemmas(self.extract_lemmas(text))

    def extract_lemmas(self, text: str) -> list[str]:
        """Tokenize text, remove proper nouns/abbreviations, and lemmatize."""

        lemmas: list[str] = []
        for match in TOKEN_RE.finditer(text):
            token = match.group(0)
            if self._should_skip_token(token):
                continue
            lemma = self.lemmatizer.normalize(token)
            if lemma and lemma.isalpha() and not self._should_skip_token(lemma):
                lemmas.append(lemma.lower())
        return lemmas

    def coverage_from_lemmas(self, lemmas: Iterable[str]) -> dict:
        """Return top-N coverage and 95%/98% rank thresholds."""

        tokens = list(lemmas)
        total = len(tokens)
        if total == 0:
            return {
                "token_count": 0,
                "type_count": 0,
                "top_n_coverage": {},
                "coverage_thresholds": {str(int(t * 100)): None for t in self.config.coverage_targets},
                "unknown_token_rate": 1.0,
            }

        ranks = [self.vocab_bank.get_rank(token) for token in tokens]
        known_ranked = [rank for rank in ranks if rank is not None]

        top_n_coverage = {}
        for boundary in self.config.bucket_boundaries:
            covered = sum(1 for rank in known_ranked if rank <= boundary)
            top_n_coverage[str(boundary)] = covered / total

        coverage_thresholds = {}
        sorted_ranks = sorted(rank for rank in ranks if rank is not None)
        for target in self.config.coverage_targets:
            key = str(int(target * 100))
            required_count = int(target * total + 0.999999)
            if len(sorted_ranks) < required_count:
                coverage_thresholds[key] = None
            else:
                coverage_thresholds[key] = sorted_ranks[required_count - 1]

        counts = Counter(tokens)
        return {
            "token_count": total,
            "type_count": len(counts),
            "top_n_coverage": top_n_coverage,
            "coverage_thresholds": coverage_thresholds,
            "unknown_token_rate": (total - len(known_ranked)) / total,
            "most_common_lemmas": counts.most_common(20),
        }

    def _should_skip_token(self, token: str) -> bool:
        if not token or any(ch.isdigit() for ch in token):
            return True
        if self.lemmatizer.is_abbreviation(token):
            return True
        # Drop likely proper nouns. This intentionally treats title-case words as
        # names, which is conservative for document-coverage calibration.
        if self.lemmatizer.is_proper_like(token):
            return True
        return False
