"""Official exam syllabus vocabulary for calibration anchoring.

Provides built-in representative word lists for Chinese EFL exam levels:
  - 中考 (Zhongkao / Middle School Exam)
  - 高考 (Gaokao / College Entrance Exam)
  - 四级 (CET-4 / College English Test Band 4)
  - 六级 (CET-6 / College English Test Band 6)

Each word list includes representative vocabulary with known rank ranges.
The purpose is to provide *anchor points* for calibration training:
the model's predicted known-rate for each official word set should match
the expected coverage at the corresponding learner level.

Usage:
    from optim.official_vocab import (
        get_official_vocab_sets,
        match_official_to_bank,
        OfficialVocabSet,
    )

    sets = get_official_vocab_sets()
    for name, vset in sets.items():
        print(name, vset.expected_vocab_size, vset.rank_range)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank


_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Cache for loaded word lists
_WORD_LISTS: dict[str, list[str]] | None = None


def _load_all_word_lists() -> dict[str, list[str]]:
    """Load all word lists from data/ JSON files."""
    global _WORD_LISTS
    if _WORD_LISTS is not None:
        return _WORD_LISTS

    mapping = {
        "中考": "zhongkao_words.json",
        "高考": "gaokao_words.json",
        "四级": "cet4_words.json",
        "六级": "cet6_words.json",
    }

    _WORD_LISTS = {}
    for name, filename in mapping.items():
        path = _DATA_DIR / filename
        with open(path, encoding="utf-8") as f:
            _WORD_LISTS[name] = json.load(f)

    return _WORD_LISTS


@dataclass(frozen=True)
class OfficialVocabSet:
    """Metadata for an official exam vocabulary set."""

    name: str                          # e.g. "中考", "高考", "CET-4", "CET-6"
    label_en: str                      # English label
    expected_vocab_size: int           # Expected total vocabulary at this level
    rank_range: tuple[int, int]        # Expected rank range in the bank
    expected_coverage: float           # Expected known-rate at this level (0-1)
    expected_coverage_range: tuple[float, float] = field(default=(0.75, 0.95))
    weight: float = 3.0                # Loss weight during training


def get_official_vocab_sets() -> dict[str, OfficialVocabSet]:
    """Return dict of all official vocab sets with metadata."""
    return {
        "中考": OfficialVocabSet(
            name="中考",
            label_en="Zhongkao (Middle School)",
            expected_vocab_size=2000,
            rank_range=(1, 1500),
            expected_coverage=0.85,
            expected_coverage_range=(0.75, 0.92),
            weight=3.0,
        ),
        "高考": OfficialVocabSet(
            name="高考",
            label_en="Gaokao (College Entrance)",
            expected_vocab_size=3500,
            rank_range=(1500, 2500),
            expected_coverage=0.80,
            expected_coverage_range=(0.70, 0.90),
            weight=3.0,
        ),
        "四级": OfficialVocabSet(
            name="四级",
            label_en="CET-4",
            expected_vocab_size=4800,
            rank_range=(2500, 4000),
            expected_coverage=0.80,
            expected_coverage_range=(0.70, 0.90),
            weight=3.0,
        ),
        "六级": OfficialVocabSet(
            name="六级",
            label_en="CET-6",
            expected_vocab_size=6500,
            rank_range=(4000, 5500),
            expected_coverage=0.75,
            expected_coverage_range=(0.65, 0.85),
            weight=3.0,
        ),
    }


def get_set_words(set_name: str) -> set[str]:
    """Get the built-in word list for an official vocab set.

    Args:
        set_name: One of "中考", "高考", "四级", "六级"

    Returns:
        Set of lowercase words.
    """
    all_lists = _load_all_word_lists()
    words = all_lists.get(set_name, [])
    return {w.strip().lower() for w in words if w.strip()}


def match_official_to_bank(
    bank: VocabBank,
) -> dict[str, OfficialVocabSet]:
    """Build official vocab sets with bank-matched word lists.

    Returns:
        Dict mapping set name -> OfficialVocabSet with matched metadata.
        The ``expected_vocab_size``, ``rank_range`` come from the definition.
    """
    return get_official_vocab_sets()


def compute_set_coverage(
    bank: VocabBank,
    set_name: str,
    known_words: set[str] | None = None,
) -> dict[str, Any]:
    """Compute coverage statistics for an official vocab set.

    Args:
        bank: VocabBank to match against.
        set_name: One of "中考", "高考", "四级", "六级"
        known_words: Optional set of words the learner knows.

    Returns:
        Dict with coverage metrics.
    """
    sets = get_official_vocab_sets()
    info = sets.get(set_name)
    if info is None:
        return {"error": f"Unknown vocab set: {set_name}"}

    set_words = get_set_words(set_name)
    rank_range = info.rank_range

    # Find matches in the vocab bank
    matched: list[tuple[str, int]] = []
    bank_words_by_lemma = {}
    for item in bank.items:
        bank_words_by_lemma[item.lemma] = item.rank

    for word in set_words:
        rank = bank.get_rank(word)
        if rank is not None:
            matched.append((word, rank))
        else:
            lemma = bank.lemmatizer.normalize(word)
            if lemma in bank_words_by_lemma:
                matched.append((word, bank_words_by_lemma[lemma]))

    matched.sort(key=lambda x: x[1])
    ranks = [r for _, r in matched]

    result: dict[str, Any] = {
        "set_name": set_name,
        "label_en": info.label_en,
        "expected_vocab_size": info.expected_vocab_size,
        "rank_range": list(rank_range),
        "words_in_bank": len(matched),
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "rank_median": int(sorted(ranks)[len(ranks)//2]) if ranks else None,
    }

    if known_words is not None:
        known_count = sum(1 for word, _ in matched if word.lower() in known_words)
        result["known_words_in_set"] = known_count
        result["coverage"] = known_count / len(matched) if matched else 0.0

    return result


def describe_official_vocab(bank: VocabBank) -> str:
    """Return a full report on all official vocab sets."""
    lines = []
    lines.append("=== 官方考试词表锚点 ===\n")

    sets = get_official_vocab_sets()
    for name, info in sets.items():
        result = compute_set_coverage(bank, name)
        lines.append(f"[{name}] {info.label_en}")
        lines.append(f"  预期词汇量: {info.expected_vocab_size}")
        lines.append(f"  rank 范围:  {info.rank_range}")
        lines.append(f"  预期认知率: {info.expected_coverage}")
        lines.append(f"  训练权重:   {info.weight}")
        lines.append(f"  在词库中匹配: {result['words_in_bank']} 词")
        lines.append(f"  匹配词 rank: [{result['rank_min']} – {result['rank_max']}]")
        lines.append("")

    return "\n".join(lines)


# ── CLI ──

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Official exam vocab sets for calibration.")
    parser.add_argument("--describe", action="store_true", help="Print describe report")
    args = parser.parse_args()

    bank = VocabBank(DEFAULT_CONFIG)
    print(f"VocabBank: {len(bank)} words\n")

    report = describe_official_vocab(bank)
    print(report)


if __name__ == "__main__":
    main()
