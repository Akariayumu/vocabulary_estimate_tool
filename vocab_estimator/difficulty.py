"""Combined difficulty scoring for vocabulary words.

Combines two dimensions:
- Educational stage priority (curriculum ordering)
- Wordfreq rank (corpus frequency)

Score range: [0, 1], higher = harder.
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from .vocab_bank import VocabBank

# ── Normalization helpers ─────────────────────────────────────────────────────

_VOCAB_SIZE = 30_000


def _norm_stage(priority: int) -> float:
    """Min-max normalize stage priority [1..11] → [0, 1]."""
    return (priority - 1) / 10.0


def _norm_rank(rank: int) -> float:
    """Log-normalize wordfreq rank [1..30000] → [0, 1].

    Log scale is chosen because wordfreq ranks follow a Zipf distribution.
    Linear normalization would compress the top 1000 ranks into ~0.03.
    """
    return math.log(rank + 1) / math.log(_VOCAB_SIZE + 1)


# ── Missing-rank: median rank per stage ───────────────────────────────────────
# Computed from actual stage_vocab × wordfreq data.

_STAGE_MEDIAN_RANK: Dict[str, int] = {
    "primary_3": 1482,
    "primary_4": 1821,
    "primary_5": 1019,
    "primary_6": 1294,
    "junior_7": 922,
    "junior_8": 1660,
    "junior_9": 2714,
    "senior": 3608,
    "cet4": 4299,
    "cet6": 6775,
    "ielts": 7218,
}

# ── Missing-stage: estimate priority from rank ────────────────────────────────
# Smooth piecewise interpolation of empirical first_stage priority vs rank.

_RANK_TO_PRIORITY_KNOTS: List[Tuple[float, float]] = [
    (0.0, 4.5),       # ultra-common → ~junior_7
    (500.0, 7.0),     # very common → ~junior_9
    (1000.0, 7.5),    # common
    (2000.0, 8.0),    # → senior
    (3000.0, 8.5),
    (5000.0, 9.0),    # → cet4
    (8000.0, 9.5),
    (10000.0, 10.0),  # → cet6
    (18000.0, 10.5),
    (30000.0, 11.0),  # → ielts
]


def _estimate_priority_from_rank(rank: int) -> float:
    """Linear interpolation between rank→priority knots."""
    if rank <= _RANK_TO_PRIORITY_KNOTS[0][0]:
        return _RANK_TO_PRIORITY_KNOTS[0][1]
    if rank >= _RANK_TO_PRIORITY_KNOTS[-1][0]:
        return _RANK_TO_PRIORITY_KNOTS[-1][1]
    for i in range(len(_RANK_TO_PRIORITY_KNOTS) - 1):
        r1, p1 = _RANK_TO_PRIORITY_KNOTS[i]
        r2, p2 = _RANK_TO_PRIORITY_KNOTS[i + 1]
        if r1 <= rank <= r2:
            t = (rank - r1) / (r2 - r1) if r2 != r1 else 0.0
            return p1 + t * (p2 - p1)
    return _RANK_TO_PRIORITY_KNOTS[-1][1]


# ── Stage key → priority lookup ───────────────────────────────────────────────

_STAGE_PRIORITY = {
    "primary_3": 1,
    "primary_4": 2,
    "primary_5": 3,
    "primary_6": 4,
    "junior_7": 5,
    "junior_8": 6,
    "junior_9": 7,
    "senior": 8,
    "cet4": 9,
    "cet6": 10,
    "ielts": 11,
}


def compute_difficulty_scores(
    stage_vocab_path: str,
    bank: VocabBank,
    alpha: float = 0.60,
    beta: float = 0.40,
    *,
    estimate_missing_stage: bool = False,
) -> Dict[str, float]:
    """Compute combined difficulty scores for all words in stage_vocab.json.

    Args:
        stage_vocab_path: Path to ``stage_vocab.json``.
        bank: A ``VocabBank`` instance (provides ``get_rank``).
        alpha: Weight for the stage-priority term (default 0.60).
        beta: Weight for the wordfreq-rank term (default 0.40).
        estimate_missing_stage: When True, also score bank-only words that
            have no stage entry, estimating their stage from rank.

    Returns:
        ``{word: difficulty_score}`` where ``difficulty_score ∈ [0, 1]``.
    """
    with open(stage_vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    stages: Dict = data["stages"]
    word_to_stage: Dict = data["word_to_stage"]

    # ── Step 1: Compute per-stage median rank for missing-data fill ──────
    # (Use live medians rather than hard-coded constants, in case data changes.)
    stage_median_rank: Dict[str, int] = {}
    for sk, sv in stages.items():
        ranks = []
        for w in sv["words"]:
            r = bank.get_rank(w)
            if r is not None:
                ranks.append(r)
        if ranks:
            ranks.sort()
            stage_median_rank[sk] = ranks[len(ranks) // 2]
        else:
            stage_median_rank[sk] = _STAGE_MEDIAN_RANK.get(sk, 15000)

    # ── Step 2: Compute scores ──────────────────────────────────────────
    scores: Dict[str, float] = {}

    for word, info in word_to_stage.items():
        first_stage = info["first_stage"]
        priority = _STAGE_PRIORITY.get(first_stage)
        if priority is None:
            # Fallback – should not happen with well-formed data
            priority = _STAGE_PRIORITY.get("ielts", 11)

        rank = bank.get_rank(word)
        if rank is None:
            rank = stage_median_rank.get(first_stage, 15000)

        ns = _norm_stage(priority)
        nr = _norm_rank(rank)
        scores[word] = alpha * ns + beta * nr

    # ── Step 3 (optional): Score bank-only words ────────────────────────
    if estimate_missing_stage:
        bank_words_total = 0
        for item in bank.items:
            w = item.word
            if w not in scores and w not in word_to_stage:
                rank = item.rank
                est_pri = _estimate_priority_from_rank(rank)
                ns = _norm_stage(est_pri)
                nr = _norm_rank(rank)
                scores[w] = alpha * ns + beta * nr
                bank_words_total += 1

    return scores


def update_stage_vocab_with_difficulty(
    stage_vocab_path: str,
    bank: VocabBank,
    alpha: float = 0.60,
    beta: float = 0.40,
    *,
    output_path: Optional[str] = None,
) -> None:
    """Add ``difficulty`` field to every word entry in stage_vocab.json.

    Modifies the file in-place (or writes to ``output_path`` when given).
    """
    with open(stage_vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    scores = compute_difficulty_scores(
        stage_vocab_path, bank, alpha=alpha, beta=beta
    )

    for word, info in data["word_to_stage"].items():
        info["difficulty"] = round(scores.get(word, 0.0), 4)

    dest = output_path or stage_vocab_path
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
