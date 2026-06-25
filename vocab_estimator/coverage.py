"""基于 lemma-frequency ranks 的文档覆盖率分析。"""

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
    """衡量文档中有多少内容被 top-N 已知词 rank 覆盖。"""

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
        """分析多个文档，并返回逐文件覆盖率和汇总覆盖率。"""

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
        """分析原始文档字符串。"""

        return self.coverage_from_lemmas(self.extract_lemmas(text))

    def extract_lemmas(self, text: str) -> list[str]:
        """对文本分词，移除专有名词/缩写，并执行 lemmatize。"""

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
        """返回 top-N 覆盖率以及 95%/98% rank 阈值。"""

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
        # 丢弃疑似专有名词。这里有意将 title-case 词视为人名/名称，
        # 对文档覆盖率校准来说更保守。
        if self.lemmatizer.is_proper_like(token):
            return True
        return False
