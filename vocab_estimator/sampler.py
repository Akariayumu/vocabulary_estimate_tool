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
