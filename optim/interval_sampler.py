"""Interval-based word sampler for calibration test sets.

Samples words from the vocab_bank at regular rank intervals,
with a configurable number of words per interval group.

Key design:
  - Divide the 30k rank range into equal-width intervals
    (interval=50 → 600 groups, interval=100 → 300 groups)
  - Randomly sample `per_group` words from each interval
  - Total test words ≈ 30000/interval × per_group

Usage:
    from optim.interval_sampler import sample_words, describe_sampling

    bank = VocabBank()
    words = sample_words(bank, interval=50, per_group=2)
    print(describe_sampling(words, bank))
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig, bucket_label
from vocab_estimator.vocab_bank import VocabBank, VocabItem


@dataclass
class SampledWord:
    """One sampled word with its interval metadata."""

    word: str
    rank: int
    interval_start: int
    interval_end: int
    bucket: str


@dataclass
class SamplingReport:
    """Summary of an interval sampling run."""

    total_sampled: int
    interval: int
    per_group: int
    n_intervals_used: int
    n_intervals_total: int
    rank_range: tuple[int, int]
    words_by_rank: dict[str, int] = field(default_factory=dict)
    intervals_with_few_words: list[tuple[int, int, int]] = field(default_factory=list)


def sample_words(
    vocab_bank: VocabBank,
    interval: int = 50,
    per_group: int = 2,
    seed: int | None = None,
) -> list[SampledWord]:
    """Sample words from vocab_bank at regular rank intervals.

    Args:
        vocab_bank: The vocabulary bank to sample from.
        interval: Rank interval width (default 50).
        per_group: Number of words to sample per interval (default 2).
        seed: Random seed for reproducibility.

    Returns:
        List of ``SampledWord`` with metadata.
    """
    rng = random.Random(seed)

    # Build a rank-indexed lookup: for each rank, list of words at that rank
    # VocabBank's items have rank populated. Multiple lemmas may share the same rank.
    rank_to_items: dict[int, list[VocabItem]] = {}
    for item in vocab_bank.items:
        rank_to_items.setdefault(item.rank, []).append(item)

    # Total rank range: 1 to vocab_size (30000)
    max_rank = vocab_bank.config.vocab_size  # 30000
    sampled: list[SampledWord] = []
    intervals_with_few_words: list[tuple[int, int, int]] = []

    for start_rank in range(1, max_rank + 1, interval):
        end_rank = min(start_rank + interval - 1, max_rank)

        # Collect all items whose rank falls within [start_rank, end_rank]
        group_items: list[VocabItem] = []
        for r in range(start_rank, end_rank + 1):
            group_items.extend(rank_to_items.get(r, []))

        if len(group_items) < per_group:
            intervals_with_few_words.append((start_rank, end_rank, len(group_items)))
            # Sample what's available
            sampled_group = list(group_items) if not group_items else (
                group_items if len(group_items) <= per_group
                else rng.sample(group_items, per_group)
            )
        else:
            sampled_group = rng.sample(group_items, per_group)

        for item in sampled_group:
            sampled.append(SampledWord(
                word=item.word,
                rank=item.rank,
                interval_start=start_rank,
                interval_end=end_rank,
                bucket=item.bucket,
            ))

    return sampled


def get_words_in_rank_range(
    vocab_bank: VocabBank,
    start_rank: int,
    end_rank: int,
) -> list[str]:
    """Get all words whose rank falls in [start_rank, end_rank].

    Added as a convenience for the ``interval_sampler`` module.
    """
    items = []
    for item in vocab_bank.items:
        if start_rank <= item.rank <= end_rank:
            items.append(item.word)
    return items


# ── Also expose as bound method on VocabBank via monkey-patch ──

def _patch_vocab_bank_method() -> None:
    """Add ``get_words_in_rank_range`` to ``VocabBank`` if not present."""
    if not hasattr(VocabBank, "get_words_in_rank_range"):
        def get_words_in_rank_range_method(self, start_rank, end_rank):
            return get_words_in_rank_range(self, start_rank, end_rank)
        VocabBank.get_words_in_rank_range = get_words_in_rank_range_method


_patch_vocab_bank_method()


# ── Reporting ──

def describe_sampling(
    sampled: list[SampledWord],
    vocab_bank: VocabBank | None = None,
) -> str:
    """Return a human-readable report of the sampling results."""
    lines = []
    lines.append(f"Total sampled: {len(sampled)} words")
    if not sampled:
        return "\n".join(lines)

    # Group by interval
    intervals: dict[int, list[SampledWord]] = {}
    for sw in sampled:
        intervals.setdefault(sw.interval_start, []).append(sw)

    # Ranks
    ranks = [sw.rank for sw in sampled]
    lines.append(f"Rank range:     {min(ranks)} – {max(ranks)}")
    lines.append(f"Intervals used: {len(intervals)}")

    # Bucket distribution
    bucket_counts: dict[str, int] = {}
    for sw in sampled:
        bucket_counts[sw.bucket] = bucket_counts.get(sw.bucket, 0) + 1

    lines.append("\nBucket distribution:")
    for bucket in sorted(bucket_counts.keys(), key=_bucket_sort_key):
        count = bucket_counts[bucket]
        pct = 100.0 * count / len(sampled)
        if vocab_bank:
            bucket_total = len(vocab_bank.words_by_bucket.get(bucket, []))
            lines.append(f"  {bucket:>5s}: {count:4d} ({pct:5.1f}%)  [total in bank: {bucket_total}]")
        else:
            lines.append(f"  {bucket:>5s}: {count:4d} ({pct:5.1f}%)")

    # Sample words (first 20)
    lines.append(f"\nSample words (first {min(20, len(sampled))}):")
    for sw in sorted(sampled[:20], key=lambda x: x.rank):
        lines.append(f"  rank {sw.rank:>5d} [{sw.interval_start:>5d}-{sw.interval_end:>5d}] {sw.word}")

    return "\n".join(lines)


def compute_interval_known_rates(
    sampled: list[SampledWord],
    known_words: set[str],
) -> dict[tuple[int, int], float]:
    """Compute observed known-rate per interval group.

    Args:
        sampled: Sampled words with interval metadata.
        known_words: Set of words the learner knows.

    Returns:
        {(start_rank, end_rank): known_rate, ...}
    """
    groups: dict[tuple[int, int], list[bool]] = {}
    for sw in sampled:
        key = (sw.interval_start, sw.interval_end)
        groups.setdefault(key, []).append(sw.word in known_words)

    return {key: sum(flags) / len(flags) for key, flags in groups.items()}


def _bucket_sort_key(bucket: str) -> tuple[int, int]:
    """Sort buckets by numeric boundary value."""
    try:
        if bucket.endswith("k"):
            return (0, int(bucket[:-1]) * 1000)
        return (1, int(bucket))
    except ValueError:
        return (2, 0)


# ── Exam-level-aware (smart) sampling ──

# Exam levels with their expected vocabulary size (word families).
# These define where the "transition zone" (p≈0.5) falls for each level.
EXAM_LEVELS: list[tuple[str, int]] = [
    ("中考", 2000),
    ("高考", 3500),
    ("四级", 5000),
    ("六级", 6500),
    ("六级+/考研", 8500),
    ("母语级", 18000),
]


def _sigmoid_known_rate(rank: float, vocab_size: int, alpha: float = 4.0) -> float:
    """Return expected p(known) for a learner with given vocab_size at a given rank.

    Uses a sigmoid centered at rank = vocab_size:
        p(known) = 1 / (1 + exp(alpha * (rank - vocab_size) / vocab_size))

    When rank == vocab_size: p = 0.5 (maximum uncertainty).
    When rank << vocab_size: p → 1.0 (almost certainly knows).
    When rank >> vocab_size: p → 0.0 (almost certainly doesn't).
    """
    if vocab_size <= 0:
        return 0.0
    z = alpha * (rank - vocab_size) / vocab_size
    # Clamp to avoid overflow
    if z > 40:
        return 0.0
    if z < -40:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def _bucket_info(
    bucket_mid_rank: float,
    bucket_size: int,
    alpha: float = 4.0,
    info_exponent: float = 1.5,
) -> float:
    """Compute the information weight for a single bucket.

    weight = size × (Σ p(1-p))^info_exponent

    Higher info_exponent (>1) amplifies differences between informative
    and uninformative buckets, reducing samples from the tails.
    """
    total_info = 0.0
    for _name, vocab_size in EXAM_LEVELS:
        p = _sigmoid_known_rate(bucket_mid_rank, vocab_size, alpha)
        variance = p * (1.0 - p)
        total_info += variance

    return bucket_size * (total_info ** info_exponent)


def describe_smart_allocation(
    vocab_bank: VocabBank,
    total_samples: int = 3000,
    alpha: float = 4.0,
    info_exponent: float = 1.5,
) -> str:
    """Return a report of the smart sampling allocation per bucket.

    Shows the expected information per bucket, the computed weights,
    and how many words would be sampled from each bucket.
    """
    boundaries = vocab_bank.config.bucket_boundaries
    lines = []
    lines.append(f"Smart sampling allocation (exam-level-aware)")
    lines.append(f"  sigmoid α = {alpha},  info_exponent γ = {info_exponent}")
    lines.append(f"  total_samples = {total_samples}")
    lines.append("")
    lines.append("  Exam levels used:")
    for lname, vsize in EXAM_LEVELS:
        lines.append(f"    {lname}: ~{vsize} vocab")
    lines.append("")

    # ── Per-bucket computations ──
    bucket_data: list[dict[str, Any]] = []
    prev_boundary = 0
    for boundary in boundaries:
        label = bucket_label(boundary)
        start = prev_boundary + 1
        end = boundary
        mid = (start + end) / 2.0
        items = vocab_bank.words_by_bucket.get(label, [])
        size = len(items)

        # Per-level information
        per_level_p: list[float] = []
        per_level_info: list[float] = []
        for _lname, vsize in EXAM_LEVELS:
            p = _sigmoid_known_rate(mid, vsize, alpha)
            per_level_p.append(p)
            per_level_info.append(p * (1.0 - p))

        total_info = sum(per_level_info)
        weight = size * (total_info ** info_exponent)

        bucket_data.append({
            "label": label,
            "size": size,
            "start": start,
            "end": end,
            "mid": mid,
            "total_info": total_info,
            "weight": weight,
            "per_level_p": per_level_p,
            "per_level_info": per_level_info,
        })
        prev_boundary = boundary

    total_weight = sum(bd["weight"] for bd in bucket_data)
    if total_weight <= 0:
        return "No non-zero weights computed."

    # ── Per-level p table ──
    header_levels = " | ".join(f"{ln:>6s}" for ln, _ in EXAM_LEVELS)
    lines.append(f" {'Bucket':>6s}  {'Mid':>6s}  {header_levels}  {'Σ_info':>8s}")
    lines.append(f" {'-'*6:>6s}  {'-'*6:>6s}  {'-'*len(header_levels):>{len(header_levels)}s}  {'-'*8:>8s}")
    for bd in bucket_data:
        p_strs = " | ".join(f"{p:>6.3f}" for p in bd["per_level_p"])
        lines.append(f" {bd['label']:>6s}  {bd['mid']:>6.0f}  {p_strs}  {bd['total_info']:>8.4f}")
    lines.append("")

    # ── Allocation table ──
    header = f"  {'Bucket':>6s}  {'Size':>6s}  {'Weight':>10s}  {'Weight×':>8s}  {'Samples':>7s}  {'%':>6s}  Note"
    sep = f"  {'------':>6s}  {'------':>6s}  {'----------':>10s}  {'--------':>8s}  {'-------':>7s}  {'------':>6s}  ----"
    lines.append(header)
    lines.append(sep)

    allocations: dict[str, int] = {}
    allocated = 0
    for bd in bucket_data:
        raw = total_samples * bd["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[bd["label"]] = alloc
        allocated += alloc

    # Rounding adjustment
    diff = total_samples - allocated
    if diff != 0 and bucket_data:
        # Adjust bucket with highest per-word information
        adjust = max(bucket_data, key=lambda b: b["total_info"])
        allocations[adjust["label"]] = max(1, allocations[adjust["label"]] + diff)
        allocated += diff

    # Compute relative weights (normalized to the minimum raw-weight bucket)
    min_weight = min(bd["weight"] for bd in bucket_data) if bucket_data else 1.0

    for bd in bucket_data:
        label = bd["label"]
        alloc = allocations[label]
        pct = 100.0 * alloc / total_samples
        rel_weight = bd["weight"] / max(min_weight, 0.001)

        # Determine note text: which exam levels have their transition here
        # A bucket is a "boundary" for a level if 0.1 < p < 0.9
        boundary_for: list[str] = []
        for idx, (_lname, vsize) in enumerate(EXAM_LEVELS):
            p = bd["per_level_p"][idx]
            if 0.1 < p < 0.9:
                boundary_for.append(_lname)

        # Normalise total_info to [0, 1] relative to max possible (6 × 0.25 = 1.5)
        norm_info = bd["total_info"] / 1.5
        
        notes = []
        if norm_info >= 0.30:
            notes.append("★高区分度")
        elif norm_info >= 0.18:
            notes.append("中高区分度")
        elif norm_info >= 0.10:
            notes.append("中区分度")
        elif norm_info >= 0.03:
            notes.append("低区分度")
        else:
            notes.append("极低区分度")

        if boundary_for:
            notes.append(f"边界:{'/'.join(boundary_for)}")

        note_str = "; ".join(notes)

        buf = (
            f"  {label:>6s}  {bd['size']:>6d}  {bd['weight']:>10.1f}"
            f"  {rel_weight:>7.2f}x  {alloc:>7d}  {pct:>5.1f}%  {note_str}"
        )
        lines.append(buf)

    lines.append(sep)
    total_size = sum(bd["size"] for bd in bucket_data)
    lines.append(f"  {'Total':>6s}  {total_size:>6d}  {total_weight:>10.1f}  {'':>8s}  {allocated:>7d}  {100.0:>5.1f}%")

    return "\n".join(lines)


def smart_sample_words(
    vocab_bank: VocabBank,
    total_samples: int = 3000,
    alpha: float = 4.0,
    info_exponent: float = 1.5,
    seed: int | None = None,
) -> list[SampledWord]:
    """Sample words from ``vocab_bank`` using exam-level-aware smart weighting.

    The sampling density is highest for buckets that lie at the boundaries
    of multiple exam levels (where p(known) ≈ 0.5 for the most learners),
    and lowest for very common buckets (everyone knows) and very rare
    buckets (no one knows).

    Args:
        vocab_bank: The vocabulary bank to sample from.
        total_samples: Total number of words to sample (default 3000).
        alpha: Sigmoid steepness parameter (default 4.0). Higher values
               make the transition between "known" and "unknown" sharper.
        info_exponent: Exponent applied to the per-word information before
                       multiplying by bucket size (default 1.5). Higher values
                       concentrate samples more on high-information buckets.
        seed: Random seed for reproducibility.

    Returns:
        List of ``SampledWord`` with metadata, sorted by rank.
    """
    rng = random.Random(seed)
    boundaries = vocab_bank.config.bucket_boundaries

    # ── Compute per-bucket info and weight ──
    bucket_stats: dict[str, dict[str, Any]] = {}
    prev_boundary = 0
    for boundary in boundaries:
        label = bucket_label(boundary)
        start = prev_boundary + 1
        end = boundary
        mid = (start + end) / 2.0
        items = list(vocab_bank.words_by_bucket.get(label, []))
        size = len(items)
        if size == 0:
            continue

        total_info = 0.0
        for _lname, vsize in EXAM_LEVELS:
            p = _sigmoid_known_rate(mid, vsize, alpha)
            total_info += p * (1.0 - p)

        weight = size * (total_info ** info_exponent)
        bucket_stats[label] = {
            "size": size,
            "start": start,
            "end": end,
            "mid": mid,
            "weight": weight,
            "items": items,
        }
        prev_boundary = boundary

    total_weight = sum(bs["weight"] for bs in bucket_stats.values())
    if total_weight <= 0:
        return []

    # ── Compute allocations ──
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = total_samples * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # Rounding adjustment: adjust the most informative bucket
    diff = total_samples - allocated
    if diff != 0 and bucket_stats:
        most_info = max(
            bucket_stats,
            key=lambda l: bucket_stats[l]["weight"] / max(bucket_stats[l]["size"], 1),
        )
        allocations[most_info] = max(1, allocations[most_info] + diff)

    # ── Sample within each bucket ──
    sampled: list[SampledWord] = []
    for label, bs in bucket_stats.items():
        n = allocations[label]
        items = bs["items"]
        bucket_start = bs["start"]
        bucket_end = bs["end"]

        if n >= len(items):
            sampled_group = list(items)
            rng.shuffle(sampled_group)
        else:
            sampled_group = rng.sample(items, n)

        for item in sampled_group:
            sampled.append(SampledWord(
                word=item.word,
                rank=item.rank,
                interval_start=bucket_start,
                interval_end=bucket_end,
                bucket=item.bucket,
            ))

    sampled.sort(key=lambda s: s.rank)
    return sampled


# ── Weighted (power-law) sampling ──


def _bucket_rank_range(
    label: str,
    boundaries: tuple[int, ...],
) -> tuple[int, int]:
    """Return (start_rank, end_rank) for a bucket label.

    Examples:
        "1k"  → (1, 1000)
        "2k"  → (1001, 2000)
        "30k" → (20001, 30000)
    """
    end_rank = int(label.rstrip("k")) * 1000
    prev = 0
    for b in boundaries:
        if b == end_rank:
            return (prev + 1, end_rank)
        prev = b
    return (prev + 1, end_rank)


def weighted_sample_words(
    vocab_bank: VocabBank,
    total_samples: int = 3000,
    power: float = 0.7,
    seed: int | None = None,
) -> list[SampledWord]:
    """Sample words from ``vocab_bank`` using power-law weighted per-bucket sampling.

    High-frequency buckets (low rank) get proportionally more samples because
    they carry more information about overall vocabulary comprehension. The
    allocation per bucket follows:

        weight_b = size_b * (1 / median_rank_b) ** power

    where ``power`` controls the skew toward high-frequency words.
    ``power=0`` → uniform across buckets (proportional to bucket size).
    ``power=1.0`` → strong high-frequency bias.

    Args:
        vocab_bank: The vocabulary bank to sample from.
        total_samples: Total number of words to sample (default 3000).
        power: Power-law exponent for rank-based weighting (default 0.7).
               Higher values skew samples more toward high-frequency words.
        seed: Random seed for reproducibility.

    Returns:
        List of ``SampledWord`` with metadata.
    """
    rng = random.Random(seed)
    sampled: list[SampledWord] = []

    # Compute per-bucket stats and weights
    bucket_stats: dict[str, dict[str, Any]] = {}
    for label, items in vocab_bank.words_by_bucket.items():
        n = len(items)
        if n == 0:
            continue
        ranks = sorted(item.rank for item in items)
        median_rank = ranks[n // 2]
        weight = n * (1.0 / max(median_rank, 1)) ** power
        bucket_stats[label] = {
            "size": n,
            "median_rank": median_rank,
            "weight": weight,
            "items": items,
        }

    total_weight = sum(bs["weight"] for bs in bucket_stats.values())

    if total_weight <= 0:
        return sampled

    # Compute allocation per bucket
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = total_samples * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # Adjust for rounding: add/subtract difference from the largest bucket
    diff = total_samples - allocated
    if diff != 0 and bucket_stats:
        biggest = max(bucket_stats, key=lambda l: bucket_stats[l]["size"])
        allocations[biggest] = max(1, allocations[biggest] + diff)

    boundaries = vocab_bank.config.bucket_boundaries

    # Sample within each bucket
    for label, bs in bucket_stats.items():
        n = allocations[label]
        items = bs["items"]
        bucket_start, bucket_end = _bucket_rank_range(label, boundaries)

        if n >= len(items):
            sampled_group = list(items)
            rng.shuffle(sampled_group)
        else:
            sampled_group = rng.sample(items, n)

        for item in sampled_group:
            sampled.append(SampledWord(
                word=item.word,
                rank=item.rank,
                interval_start=bucket_start,
                interval_end=bucket_end,
                bucket=item.bucket,
            ))

    return sampled


def describe_weighted_allocation(
    vocab_bank: VocabBank,
    total_samples: int = 3000,
    power: float = 0.7,
) -> str:
    """Return a report of the weighted sampling allocation per bucket.

    Non-destructive: does not perform the actual sampling, only shows
    how many words would be drawn from each bucket.
    """
    lines = []
    lines.append(f"Weighted sampling allocation (power={power}, total_samples={total_samples}):")
    lines.append("")
    header = f"  {'Bucket':>6s}  {'Size':>6s}  {'Median':>7s}  {'Weight':>10s}  {'Alloc':>6s}  {'% of total':>10s}"
    sep   = f"  {'------':>6s}  {'------':>6s}  {'-------':>7s}  {'----------':>10s}  {'------':>6s}  {'----------':>10s}"
    lines.append(header)
    lines.append(sep)

    bucket_stats: dict[str, dict[str, Any]] = {}
    for label, items in vocab_bank.words_by_bucket.items():
        n = len(items)
        if n == 0:
            continue
        ranks = sorted(item.rank for item in items)
        median_rank = ranks[n // 2]
        weight = n * (1.0 / max(median_rank, 1)) ** power
        bucket_stats[label] = {"size": n, "median_rank": median_rank, "weight": weight}

    total_weight = sum(bs["weight"] for bs in bucket_stats.values())
    if total_weight <= 0:
        return "No buckets with items."

    allocated_total = 0
    for label in sorted(bucket_stats, key=_bucket_sort_key):
        bs = bucket_stats[label]
        raw = total_samples * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocated_total += alloc
        pct = 100.0 * alloc / total_samples
        lines.append(
            f"  {label:>6s}  {bs['size']:>6d}  {bs['median_rank']:>7d}  {bs['weight']:>10.3f}  {alloc:>6d}  {pct:>9.1f}%"
        )

    diff = total_samples - allocated_total
    lines.append(f"  {'':>6s}  {'':>6s}  {'':>7s}  {'':>10s}  {'------':>6s}  {'':>10s}")
    lines.append(f"  {'Total':>6s}  {'':>6s}  {'':>7s}  {total_weight:>10.3f}  {allocated_total + diff:>6d}  {100.0:>9.1f}%")
    if diff != 0:
        lines.append(f"  (rounding adjustment: {diff:+d} to largest bucket)")

    return "\n".join(lines)


# ── CLI ──

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Interval-based word sampler for calibration.")
    parser.add_argument("--interval", type=int, default=50, help="Rank interval width (default 50)")
    parser.add_argument("--per-group", type=int, default=2, help="Words per interval (default 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--weighted", action="store_true", help="Use power-law weighted per-bucket sampling")
    parser.add_argument("--power", type=float, default=0.7, help="Power-law exponent (default 0.7)")
    parser.add_argument("--total-samples", type=int, default=3000, help="Total samples for weighted mode (default 3000)")
    parser.add_argument("--describe", action="store_true", help="Show allocation plan without sampling")
    parser.add_argument("--smart", action="store_true", help="Use exam-level-aware smart sampling")
    parser.add_argument("--alpha", type=float, default=4.0, help="Sigmoid steepness for smart sampling (default 4.0)")
    parser.add_argument("--info-exponent", type=float, default=1.5, help="Info exponent for smart sampling (default 1.5)")
    args = parser.parse_args()

    bank = VocabBank(DEFAULT_CONFIG)
    print(f"VocabBank: {len(bank)} words\n")

    if args.smart:
        if args.describe:
            report = describe_smart_allocation(
                bank,
                total_samples=args.total_samples,
                alpha=args.alpha,
                info_exponent=args.info_exponent,
            )
        else:
            sampled = smart_sample_words(
                bank,
                total_samples=args.total_samples,
                alpha=args.alpha,
                info_exponent=args.info_exponent,
                seed=args.seed,
            )
            report = describe_sampling(sampled, bank)
    elif args.weighted:
        if args.describe:
            report = describe_weighted_allocation(bank, total_samples=args.total_samples, power=args.power)
        else:
            sampled = weighted_sample_words(bank, total_samples=args.total_samples, power=args.power, seed=args.seed)
            report = describe_sampling(sampled, bank)
    else:
        if args.describe:
            # For uniform interval sampling, show predicted counts
            n_intervals = (bank.config.vocab_size + args.interval - 1) // args.interval
            total_est = n_intervals * args.per_group
            report = (
                f"Uniform interval sampling (interval={args.interval}, per_group={args.per_group}):\n"
                f"  Estimated total: ~{total_est} words\n"
                f"  Intervals: {n_intervals}"
            )
            print(report)
            return
        sampled = sample_words(bank, interval=args.interval, per_group=args.per_group, seed=args.seed)
        report = describe_sampling(sampled, bank)
    print(report)


# ── Synthetic test-question sampling (high-frequency weighted) ──


def sample_test_questions(
    vocab_bank: VocabBank,
    n: int = 100,
    power: float = 0.5,
    seed: int | None = None,
) -> list[SampledWord]:
    """Sample test questions from ``vocab_bank`` using frequency-weighted distribution.

    High-frequency (low-rank) words have higher probability of appearing in test
    questions, simulating real exam design where common vocabulary is tested more
    often. The sampling follows a power-law distribution per bucket:

        weight_bucket = size_bucket * (1 / median_rank_bucket) ** power

    Then words are drawn within each bucket proportional to its allocation.

    Args:
        vocab_bank: The vocabulary bank to sample from.
        n: Number of test questions to sample (default 100).
        power: Power-law exponent (default 0.5). Higher values skew samples
               more heavily toward high-frequency words.
        seed: Random seed for reproducibility.

    Returns:
        List of ``SampledWord`` with metadata, sorted by rank.
    """
    rng = random.Random(seed)

    # ── Compute per-bucket weights ──
    bucket_stats: dict[str, dict[str, Any]] = {}
    for label, items in vocab_bank.words_by_bucket.items():
        n_items = len(items)
        if n_items == 0:
            continue
        ranks = sorted(item.rank for item in items)
        median_rank = ranks[n_items // 2]
        weight = n_items * (1.0 / max(median_rank, 1)) ** power
        bucket_stats[label] = {
            "size": n_items,
            "median_rank": median_rank,
            "weight": weight,
            "items": items,
        }

    total_weight = sum(bs["weight"] for bs in bucket_stats.values())
    if total_weight <= 0:
        return []

    # ── Compute allocation per bucket ──
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = n * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # Adjust rounding difference on the largest bucket
    diff = n - allocated
    if diff != 0 and bucket_stats:
        biggest = max(bucket_stats, key=lambda l: bucket_stats[l]["size"])
        allocations[biggest] = max(1, allocations[biggest] + diff)

    # ── Sample within each bucket ──
    boundaries = vocab_bank.config.bucket_boundaries
    sampled: list[SampledWord] = []
    for label, bs in bucket_stats.items():
        alloc_n = allocations[label]
        items = bs["items"]
        bucket_start, bucket_end = _bucket_rank_range(label, boundaries)

        if alloc_n >= len(items):
            group = list(items)
            rng.shuffle(group)
        else:
            group = rng.sample(items, alloc_n)

        for item in group:
            sampled.append(SampledWord(
                word=item.word,
                rank=item.rank,
                interval_start=bucket_start,
                interval_end=bucket_end,
                bucket=item.bucket,
            ))

    sampled.sort(key=lambda s: s.rank)
    return sampled


if __name__ == "__main__":
    main()
