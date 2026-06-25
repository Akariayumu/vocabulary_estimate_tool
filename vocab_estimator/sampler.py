"""词汇测试的采样策略。"""

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
    """某个频率 bucket 的观测已知率。"""

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


# Warmup 结果 -> (level_label, point_estimate, exam_set, norm_mu, norm_sigma)
WARMUP_LEVELS: list[tuple[str, int, int, str, int, int]] = [
    ("高中", 3500, 0, "gaokao", 3000, 2000),
    ("四级", 4500, 1, "gaokao+cet6", 4500, 2500),
    ("六级", 6000, 2, "cet6", 6000, 3000),
    ("考研/高级", 8000, 3, "cet6", 8000, 3500),
]


def get_exam_vocab_words(exam_vocab_dir: str | None = None) -> dict[str, list[str]]:
    """从文本文件加载考试词汇。"""
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
    """使用确定性采样生成分层和 adaptive 词表。"""

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
        """从每个频率 bucket 抽取固定数量的词。"""

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
        """在估计能力边界附近进行更密集采样。

        观测已知率最接近 0.5 的 bucket 会获得下一批中最大的份额。
        为保持测试校准，每个非空 bucket 仍至少包含一个 item。
        """

        responses = list(previous_responses)
        seen = {word.lower() for word, _ in responses} if exclude_seen else set()
        performances = self.bucket_performance(responses)

        if not performances:
            return self.balanced_sample(max(1, total_count // len(self.vocab_bank.words_by_bucket)))

        # 距离 0.5 越小，表示越接近学习者边界。
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
        """返回均衡的第一阶段列表，或 adaptive 后续列表。"""

        if previous_responses is None:
            return self.balanced_sample(per_bucket=per_bucket)
        return self.adaptive_sample(previous_responses, total_count=adaptive_count)

    def bucket_performance(self, responses: Iterable[Response]) -> list[BucketPerformance]:
        """按 bucket 汇总观测已知率。"""

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
        """生成 Stage 2 词：细化边界 bucket 的采样。

        返回 (new_test_items, boundary_bucket_labels)。
        """
        from .config import bucket_beta_prior

        responses = list(stage1_responses)
        extra = extra_per_bucket or self.config.stage2_extra_per_bucket

        # Stage 1 中已经见过的词
        seen = {word.lower() for word, _ in responses}

        # 使用 Bayesian smoothing 计算每个 bucket 的统计量
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

        # 从每个边界 bucket 额外采样
        new_items: list[TestItem] = []
        for bucket in boundary_buckets:
            batch = self._sample_bucket(bucket, extra, exclude=seen)
            new_items.extend(batch)
            seen.update(word.lower() for word, _, _ in batch)

        return new_items, boundary_buckets

    def warmup_sample(self) -> tuple[list[TestItem], dict[str, int]]:
        """生成 10 道混合难度 warmup 题。

        返回 (items, sources)，其中 sources 将每个词映射到其难度等级。
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
        """根据 warmup 等级生成 adaptive 测试题。

        使用以估计等级为中心的正态分布采样。
        候选池限制为 CET-6 考试词汇，每个词保留原始 wordfreq rank。
        """
        seen = set(seen_words) if seen_words else set()

        # 确定 warmup 等级索引
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

        # 加载 CET-6 词作为候选池（始终使用 CET-6）
        exam_words_data = get_exam_vocab_words()
        exam_words = exam_words_data.get("cet6", [])

        # 过滤出词库中存在的 CET-6 词
        valid_candidates: list[VocabItem] = []
        for word in exam_words:
            item = self.vocab_bank.item_by_word.get(word) or self.vocab_bank.item_by_lemma.get(word)
            if item and item.word.lower() not in seen:
                valid_candidates.append(item)

        if not valid_candidates:
            # fallback：使用全部词库 item
            valid_candidates = [
                item for item in self.vocab_bank.items
                if item.word.lower() not in seen
            ]

        if not valid_candidates:
            return []

        # 为保证多样性，先从估计等级附近的每个 tier 选取一些 item，
        # 再用纯 weighted sampling 补足。

        # Tier 1 (60%)：mu +/- sigma 内的 item（核心范围）
        # Tier 2 (25%)：mu +/- 2*sigma 内的 item（扩展范围）
        # Tier 3 (15%)：剩余候选（扩散范围）

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

        # 在每个 tier 内 shuffle，以便随机选择
        self.rng.shuffle(tier1)
        self.rng.shuffle(tier2)
        self.rng.shuffle(tier3)

        n1 = min(int(total_count * 0.60), len(tier1))
        n2 = min(int(total_count * 0.25), len(tier2))
        n3 = min(total_count - n1 - n2, len(tier3))

        # 分配剩余名额
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

        # 若仍不足，则回退到 weighted fill
        if len(selected) < total_count:
        # 为所有候选计算 normal PDF 权重
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
