"""Sampling strategies for vocabulary tests."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .config import DEFAULT_CONFIG, EstimatorConfig
from .vocab_bank import VocabBank, VocabItem


TestItem = tuple[str, int, str]
Response = tuple[str, bool]


@dataclass(frozen=True)
class BucketPerformance:
    """Observed known-rate for one frequency bucket."""

    bucket: str
    asked: int
    known_rate: float
    distance_to_boundary: float


WARMUP_SAMPLES: dict[str, list[tuple[int, int]]] = {
    "easy": [(1000, 2000)],
    "medium": [(3000, 5000)],
    "hard": [(8000, 10000)],
    "very_hard": [(15000, 20000)],
}

WARMUP_COUNTS: dict[str, int] = {
    "easy": 3,
    "medium": 3,
    "hard": 2,
    "very_hard": 2,
}


# Warmup result -> (level_label, point_estimate, exam_set, norm_mu, norm_sigma)
WARMUP_LEVELS: list[tuple[str, int, int, str, int, int]] = [
    ("高中", 3500, 0, "gaokao", 3000, 2000),
    ("四级", 4500, 1, "gaokao+cet6", 4500, 2500),
    ("六级", 6000, 2, "cet6", 6000, 3000),
    ("考研/高级", 8000, 3, "cet6", 8000, 3500),
]


def get_exam_vocab_words(exam_vocab_dir: str | None = None) -> dict[str, list[str]]:
    """Load exam vocabulary words from text files."""
    import os
    from pathlib import Path

    if exam_vocab_dir is None:
        base = Path(__file__).resolve().parent.parent
        exam_vocab_dir = str(base / "data" / "exam_vocab")

    result: dict[str, list[str]] = {}

    gaokao_path = os.path.join(exam_vocab_dir, "gaokao.txt")
    cet6_path = os.path.join(exam_vocab_dir, "cet6.txt")

    if os.path.exists(gaokao_path):
        with open(gaokao_path) as f:
            result["gaokao"] = [w.strip().lower() for w in f if w.strip()]
    else:
        result["gaokao"] = []

    if os.path.exists(cet6_path):
        with open(cet6_path) as f:
            result["cet6"] = [w.strip().lower() for w in f if w.strip()]
    else:
        result["cet6"] = []

    result["gaokao+cet6"] = list(set(result["gaokao"]) | set(result["cet6"]))

    return result


class VocabularySampler:
    """Generate stratified and adaptive word lists with deterministic sampling."""

    def __init__(
        self,
        vocab_bank: VocabBank,
        config: EstimatorConfig = DEFAULT_CONFIG,
        seed: int | None = None,
    ) -> None:
        self.vocab_bank = vocab_bank
        self.config = config
        self.rng = random.Random(config.random_seed if seed is None else seed)

    def balanced_sample(self, per_bucket: int | None = None) -> list[TestItem]:
        """Sample a fixed number of words from each frequency bucket."""

        count = per_bucket or self.config.default_sample_per_bucket
        selected: list[TestItem] = []
        for bucket in self.vocab_bank.words_by_bucket:
            selected.extend(self._sample_bucket(bucket, count, exclude=set()))
        return selected

    def adaptive_sample(
        self,
        previous_responses: Iterable[Response],
        total_count: int = 40,
        exclude_seen: bool = True,
    ) -> list[TestItem]:
        """Sample more densely around the estimated ability boundary.

        Buckets whose observed known-rate is closest to 0.5 receive the largest
        share of the next batch. The method still includes at least one item from
        every non-empty bucket to keep the test calibrated.
        """

        responses = list(previous_responses)
        seen = {word.lower() for word, _ in responses} if exclude_seen else set()
        performances = self.bucket_performance(responses)

        if not performances:
            return self.balanced_sample(max(1, total_count // len(self.vocab_bank.words_by_bucket)))

        # Smaller distance to 0.5 means closer to the learner's boundary.
        weights: dict[str, float] = {}
        for perf in performances:
            weights[perf.bucket] = 1.0 / (0.05 + perf.distance_to_boundary)

        for bucket, items in self.vocab_bank.words_by_bucket.items():
            if items and bucket not in weights:
                weights[bucket] = 0.5

        selected: list[TestItem] = []
        non_empty_buckets = [b for b, items in self.vocab_bank.words_by_bucket.items() if items]
        for bucket in non_empty_buckets:
            selected.extend(self._sample_bucket(bucket, 1, exclude=seen))
            seen.update(word.lower() for word, _, _ in selected)

        remaining = max(0, total_count - len(selected))
        weight_sum = sum(weights.get(bucket, 0.0) for bucket in non_empty_buckets) or 1.0
        allocations = {
            bucket: int(round(remaining * weights.get(bucket, 0.0) / weight_sum))
            for bucket in non_empty_buckets
        }

        while sum(allocations.values()) < remaining:
            bucket = max(non_empty_buckets, key=lambda b: weights.get(b, 0.0))
            allocations[bucket] += 1
        while sum(allocations.values()) > remaining:
            bucket = max(non_empty_buckets, key=lambda b: allocations[b])
            allocations[bucket] -= 1

        for bucket, count in allocations.items():
            batch = self._sample_bucket(bucket, count, exclude=seen)
            selected.extend(batch)
            seen.update(word.lower() for word, _, _ in batch)

        self.rng.shuffle(selected)
        return selected[:total_count]

    def generate_test_list(
        self,
        per_bucket: int | None = None,
        previous_responses: Iterable[Response] | None = None,
        adaptive_count: int = 40,
    ) -> list[TestItem]:
        """Return a balanced first-stage list or an adaptive follow-up list."""

        if previous_responses is None:
            return self.balanced_sample(per_bucket=per_bucket)
        return self.adaptive_sample(previous_responses, total_count=adaptive_count)

    def bucket_performance(self, responses: Iterable[Response]) -> list[BucketPerformance]:
        """Summarize observed known-rate by bucket."""

        stats: dict[str, list[bool]] = defaultdict(list)
        for word, known in responses:
            bucket = self.vocab_bank.get_bucket(word)
            if bucket is not None:
                stats[bucket].append(bool(known))

        result: list[BucketPerformance] = []
        for bucket, values in stats.items():
            if not values:
                continue
            rate = sum(values) / len(values)
            result.append(
                BucketPerformance(
                    bucket=bucket,
                    asked=len(values),
                    known_rate=rate,
                    distance_to_boundary=abs(rate - self.config.adaptive_boundary_rate),
                )
            )
        return sorted(result, key=lambda p: p.distance_to_boundary)

    def stage2_refine_sample(
        self,
        stage1_responses: Iterable[Response],
        extra_per_bucket: int | None = None,
    ) -> tuple[list[TestItem], list[str]]:
        """Generate Stage 2 words: refine sampling of boundary buckets.

        Returns (new_test_items, boundary_bucket_labels).
        """
        from .config import bucket_beta_prior

        responses = list(stage1_responses)
        extra = extra_per_bucket or self.config.stage2_extra_per_bucket

        # Words already seen in Stage 1
        seen = {word.lower() for word, _ in responses}

        # Compute per-bucket stats using Bayesian smoothing
        stats: dict[str, list[bool]] = defaultdict(list)
        for word, known in responses:
            bucket = self.vocab_bank.get_bucket(word)
            if bucket is not None:
                stats[bucket].append(bool(known))

        bucket_labels = list(self.vocab_bank.words_by_bucket)
        n_buckets = len(bucket_labels)

        boundary_buckets: list[str] = []
        for bucket, values in stats.items():
            if not values:
                continue
            idx = bucket_labels.index(bucket)
            alpha, beta = bucket_beta_prior(idx, n_buckets, self.config)
            known_count = sum(values)
            total = len(values)
            smoothed = (known_count + alpha) / (total + alpha + beta)
            if self.config.stage2_boundary_low <= smoothed <= self.config.stage2_boundary_high:
                boundary_buckets.append(bucket)

        # Sample extra words from each boundary bucket
        new_items: list[TestItem] = []
        for bucket in boundary_buckets:
            batch = self._sample_bucket(bucket, extra, exclude=seen)
            new_items.extend(batch)
            seen.update(word.lower() for word, _, _ in batch)

        return new_items, boundary_buckets

    def warmup_sample(self) -> tuple[list[TestItem], dict[str, int]]:
        """Generate 10 warmup questions with mixed difficulty.

        Returns (items, sources) where sources maps each word to its difficulty level.
        """
        selected: list[TestItem] = []
        sources: dict[str, int] = {}
        seen: set[str] = set()

        for level_key in ("easy", "medium", "hard", "very_hard"):
            count = WARMUP_COUNTS[level_key]
            ranges = WARMUP_SAMPLES[level_key]
            candidates: list[VocabItem] = []
            for lo, hi in ranges:
                for item in self.vocab_bank.items:
                    if lo <= item.rank <= hi and item.word.lower() not in seen:
                        candidates.append(item)
            self.rng.shuffle(candidates)
            taken = candidates[:count]
            for item in taken:
                selected.append((item.word, item.rank, item.bucket))
                seen.add(item.word.lower())
                sources[item.word] = level_key

        self.rng.shuffle(selected)
        return selected, sources

    def adaptive_normal_sample(
        self,
        warmup_correct: int,
        total_count: int = 36,
        exclude_seen: bool = True,
        seen_words: set[str] | None = None,
    ) -> list[TestItem]:
        """Generate adaptive test questions based on warmup level.

        Uses normal distribution sampling centered on the estimated level.
        The candidate pool is limited to CET-6 exam vocabulary words,
        with each word retaining its original wordfreq rank.
        """
        seen = set(seen_words) if seen_words else set()

        # Determine warmup level index
        correct = warmup_correct
        if correct <= 3:
            level_idx = 0
        elif correct <= 6:
            level_idx = 1
        elif correct <= 8:
            level_idx = 2
        else:
            level_idx = 3

        _, _, _, _, mu, sigma = WARMUP_LEVELS[level_idx]

        # Load CET-6 words as the candidate pool (always CET-6)
        exam_words_data = get_exam_vocab_words()
        exam_words = exam_words_data.get("cet6", [])

        # Filter CET-6 words that exist in vocab bank
        valid_candidates: list[VocabItem] = []
        for word in exam_words:
            item = self.vocab_bank.item_by_word.get(word) or self.vocab_bank.item_by_lemma.get(word)
            if item and item.word.lower() not in seen:
                valid_candidates.append(item)

        if not valid_candidates:
            # Fallback: use all bank items
            valid_candidates = [
                item for item in self.vocab_bank.items
                if item.word.lower() not in seen
            ]

        if not valid_candidates:
            return []

        # Ensure diversity by first selecting some items from each tier
        # around the estimated level, then fill with pure weighted sampling.

        # Tier 1 (60%): items within mu +/- sigma (core range)
        # Tier 2 (25%): items within mu +/- 2*sigma (extended range)
        # Tier 3 (15%): remaining candidates (spread)

        tier1: list[VocabItem] = []
        tier2: list[VocabItem] = []
        tier3: list[VocabItem] = []

        for item in valid_candidates:
            z = abs(item.rank - mu) / max(sigma, 1)
            if z <= 1.0:
                tier1.append(item)
            elif z <= 2.0:
                tier2.append(item)
            else:
                tier3.append(item)

        # Shuffle within each tier for stochastic selection
        self.rng.shuffle(tier1)
        self.rng.shuffle(tier2)
        self.rng.shuffle(tier3)

        n1 = min(int(total_count * 0.60), len(tier1))
        n2 = min(int(total_count * 0.25), len(tier2))
        n3 = min(total_count - n1 - n2, len(tier3))

        # Distribute remaining slots
        remaining = total_count - n1 - n2 - n3
        for _ in range(remaining):
            if len(tier1) > n1:
                n1 += 1
            elif len(tier2) > n2:
                n2 += 1
            elif len(tier3) > n3:
                n3 += 1

        selected: list[TestItem] = []
        used = set(seen)

        for item_list, count in [(tier1, n1), (tier2, n2), (tier3, n3)]:
            taken = 0
            for item in item_list:
                if taken >= count:
                    break
                if item.word.lower() in used:
                    continue
                selected.append((item.word, item.rank, item.bucket))
                used.add(item.word.lower())
                taken += 1

        # If still short, fall through to weighted fill
        if len(selected) < total_count:
            # Compute normal PDF weights for all candidates
            weighted_pool: list[tuple[float, VocabItem]] = []
            for item in valid_candidates:
                if item.word.lower() in used:
                    continue
                z = (item.rank - mu) / max(sigma, 1)
                w = 2.71828 ** (-0.5 * z * z)
                weighted_pool.append((w, item))
            weighted_pool.sort(key=lambda x: -x[0])

            for _, item in weighted_pool:
                if len(selected) >= total_count:
                    break
                if item.word.lower() not in used:
                    selected.append((item.word, item.rank, item.bucket))
                    used.add(item.word.lower())

        self.rng.shuffle(selected)
        return selected

    def _sample_bucket(self, bucket: str, count: int, exclude: set[str]) -> list[TestItem]:
        items = [
            item
            for item in self.vocab_bank.get_items_in_bucket(bucket)
            if item.word.lower() not in exclude
        ]
        if not items or count <= 0:
            return []
        if count >= len(items):
            sampled: list[VocabItem] = list(items)
        else:
            sampled = self.rng.sample(items, count)
        return [(item.word, item.rank, item.bucket) for item in sampled]
