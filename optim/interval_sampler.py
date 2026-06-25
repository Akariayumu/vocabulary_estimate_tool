"""用于校准测试集的 interval-based 词汇采样器。

按固定 rank interval 从 vocab_bank 中采样词，每个 interval group 的采样数可配置。

核心设计：
  - 将 30k rank 范围划分为等宽 intervals
    (interval=50 → 600 groups，interval=100 → 300 groups)
  - 从每个 interval 随机采样 `per_group` 个词
  - 总测试词数约为 30000/interval × per_group

用法：
    from optim.interval_sampler import sample_words, describe_sampling

    bank = VocabBank()
    words = sample_words(bank, interval=50, per_group=2)
    print(describe_sampling(words, bank))"""

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
    """带 interval 元数据的一个采样词。"""

    word: str
    rank: int
    interval_start: int
    interval_end: int
    bucket: str


@dataclass
class SamplingReport:
    """一次 interval sampling 运行的 summary。"""

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
    """按固定 rank interval 从 vocab_bank 采样词。

    Args:
        vocab_bank: 要采样的词库。
        interval: Rank interval 宽度（默认 50）。
        per_group: 每个 interval 的采样词数（默认 2）。
        seed: 用于复现的随机种子。

    Returns:
        带元数据的 ``SampledWord`` 列表。"""
    rng = random.Random(seed)

    # 构建按 rank 索引的查表：每个 rank 对应该 rank 的词列表
    # VocabBank items 已填充 rank；多个 lemmas 可能共享同一 rank。
    rank_to_items: dict[int, list[VocabItem]] = {}
    for item in vocab_bank.items:
        rank_to_items.setdefault(item.rank, []).append(item)

    # 总 rank 范围：1 到 vocab_size（30000）
    max_rank = vocab_bank.config.vocab_size  # 30000
    sampled: list[SampledWord] = []
    intervals_with_few_words: list[tuple[int, int, int]] = []

    for start_rank in range(1, max_rank + 1, interval):
        end_rank = min(start_rank + interval - 1, max_rank)

        # 收集 rank 落在 [start_rank, end_rank] 内的所有 items
        group_items: list[VocabItem] = []
        for r in range(start_rank, end_rank + 1):
            group_items.extend(rank_to_items.get(r, []))

        if len(group_items) < per_group:
            intervals_with_few_words.append((start_rank, end_rank, len(group_items)))
            # 对可用项进行采样
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
    """获取 rank 落在 [start_rank, end_rank] 中的所有词。

    作为 ``interval_sampler`` 模块的便利方法添加。"""
    items = []
    for item in vocab_bank.items:
        if start_rank <= item.rank <= end_rank:
            items.append(item.word)
    return items


# ── 同时通过 monkey-patch 暴露为 VocabBank 绑定方法 ──

def _patch_vocab_bank_method() -> None:
    """若 ``VocabBank`` 尚无 ``get_words_in_rank_range``，则添加该方法。"""
    if not hasattr(VocabBank, "get_words_in_rank_range"):
        def get_words_in_rank_range_method(self, start_rank, end_rank):
            return get_words_in_rank_range(self, start_rank, end_rank)
        VocabBank.get_words_in_rank_range = get_words_in_rank_range_method


_patch_vocab_bank_method()


# ── 报告 ──

def describe_sampling(
    sampled: list[SampledWord],
    vocab_bank: VocabBank | None = None,
) -> str:
    """返回采样结果的易读报告。"""
    lines = []
    lines.append(f"Total sampled: {len(sampled)} words")
    if not sampled:
        return "\n".join(lines)

    # 按 interval 分组
    intervals: dict[int, list[SampledWord]] = {}
    for sw in sampled:
        intervals.setdefault(sw.interval_start, []).append(sw)

    # Ranks 列
    ranks = [sw.rank for sw in sampled]
    lines.append(f"Rank range:     {min(ranks)} – {max(ranks)}")
    lines.append(f"Intervals used: {len(intervals)}")

    # Bucket 分布
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

    # 示例词（前 20 个）
    lines.append(f"\nSample words (first {min(20, len(sampled))}):")
    for sw in sorted(sampled[:20], key=lambda x: x.rank):
        lines.append(f"  rank {sw.rank:>5d} [{sw.interval_start:>5d}-{sw.interval_end:>5d}] {sw.word}")

    return "\n".join(lines)


def compute_interval_known_rates(
    sampled: list[SampledWord],
    known_words: set[str],
) -> dict[tuple[int, int], float]:
    """计算每个 interval group 的观测 known-rate。

    Args:
        sampled: 带 interval 元数据的采样词。
        known_words: 学习者已知词集合。

    Returns:
        {(start_rank, end_rank): known_rate, ...}"""
    groups: dict[tuple[int, int], list[bool]] = {}
    for sw in sampled:
        key = (sw.interval_start, sw.interval_end)
        groups.setdefault(key, []).append(sw.word in known_words)

    return {key: sum(flags) / len(flags) for key, flags in groups.items()}


def _bucket_sort_key(bucket: str) -> tuple[int, int]:
    """按数值边界对 buckets 排序。"""
    try:
        if bucket.endswith("k"):
            return (0, int(bucket[:-1]) * 1000)
        return (1, int(bucket))
    except ValueError:
        return (2, 0)


# ── 考试等级感知（smart）sampling ──

# 考试等级及其期望词汇量（word families）。
# 它们定义每个等级的 “transition zone”（p≈0.5）所在位置。
EXAM_LEVELS: list[tuple[str, int]] = [
    ("中考", 2000),
    ("高考", 3500),
    ("四级", 5000),
    ("六级", 6500),
    ("六级+/考研", 8500),
    ("母语级", 18000),
]


def _sigmoid_known_rate(rank: float, vocab_size: int, alpha: float = 4.0) -> float:
    """返回给定 vocab_size 的学习者在某个 rank 上的期望 p(known)。

    使用以 rank = vocab_size 为中心的 sigmoid：
        p(known) = 1 / (1 + exp(alpha * (rank - vocab_size) / vocab_size))

    当 rank == vocab_size 时：p = 0.5（最大不确定性）。
    当 rank << vocab_size 时：p → 1.0（几乎肯定认识）。
    当 rank >> vocab_size 时：p → 0.0（几乎肯定不认识）。"""
    if vocab_size <= 0:
        return 0.0
    z = alpha * (rank - vocab_size) / vocab_size
    # Clamp 以避免 overflow
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
    """计算单个 bucket 的 information weight。

    weight = size × (Σ p(1-p))^info_exponent

    较高的 info_exponent（>1）会放大有信息量 bucket 与低信息量 bucket 的差异，
    减少尾部采样。"""
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
    """返回 smart sampling 每个 bucket 分配量的报告。

    展示每个 bucket 的期望 information、计算出的 weights，
    以及每个 bucket 会采样多少词。"""
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

    # ── 每个 bucket 的计算 ──
    bucket_data: list[dict[str, Any]] = []
    prev_boundary = 0
    for boundary in boundaries:
        label = bucket_label(boundary)
        start = prev_boundary + 1
        end = boundary
        mid = (start + end) / 2.0
        items = vocab_bank.words_by_bucket.get(label, [])
        size = len(items)

        # 每个等级的 information
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

    # ── 每个等级的 p 表 ──
    header_levels = " | ".join(f"{ln:>6s}" for ln, _ in EXAM_LEVELS)
    lines.append(f" {'Bucket':>6s}  {'Mid':>6s}  {header_levels}  {'Σ_info':>8s}")
    lines.append(f" {'-'*6:>6s}  {'-'*6:>6s}  {'-'*len(header_levels):>{len(header_levels)}s}  {'-'*8:>8s}")
    for bd in bucket_data:
        p_strs = " | ".join(f"{p:>6.3f}" for p in bd["per_level_p"])
        lines.append(f" {bd['label']:>6s}  {bd['mid']:>6.0f}  {p_strs}  {bd['total_info']:>8.4f}")
    lines.append("")

    # ── Allocation 表 ──
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

    # 舍入调整
    diff = total_samples - allocated
    if diff != 0 and bucket_data:
        # 调整每词 information 最高的 bucket
        adjust = max(bucket_data, key=lambda b: b["total_info"])
        allocations[adjust["label"]] = max(1, allocations[adjust["label"]] + diff)
        allocated += diff

    # 计算相对 weights（归一化到最小 raw-weight bucket）
    min_weight = min(bd["weight"] for bd in bucket_data) if bucket_data else 1.0

    for bd in bucket_data:
        label = bd["label"]
        alloc = allocations[label]
        pct = 100.0 * alloc / total_samples
        rel_weight = bd["weight"] / max(min_weight, 0.001)

        # 确定 note 文本：哪些考试等级的 transition 位于此处
        # 若 0.1 < p < 0.9，则该 bucket 是某等级的 “boundary”
        boundary_for: list[str] = []
        for idx, (_lname, vsize) in enumerate(EXAM_LEVELS):
            p = bd["per_level_p"][idx]
            if 0.1 < p < 0.9:
                boundary_for.append(_lname)

        # 将 total_info 按最大可能值（6 × 0.25 = 1.5）归一化到 [0, 1]
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
    """使用考试等级感知的 smart weighting 从 ``vocab_bank`` 采样词。

    位于多个考试等级边界的 buckets 采样密度最高
    （对最多学习者而言 p(known) ≈ 0.5），而非常常见 bucket（人人都会）和
    非常罕见 bucket（几乎没人会）采样密度最低。

    Args:
        vocab_bank: 要采样的词库。
        total_samples: 总采样词数（默认 3000）。
        alpha: Sigmoid 陡峭度参数（默认 4.0）。值越高，known/unknown 转换越陡。
        info_exponent: 先作用在每词 information 上，再乘以 bucket size 的指数（默认 1.5）。
            值越高，采样越集中到高信息量 buckets。
        seed: 用于复现的随机种子。

    Returns:
        按 rank 排序、带元数据的 ``SampledWord`` 列表。"""
    rng = random.Random(seed)
    boundaries = vocab_bank.config.bucket_boundaries

    # ── 计算每个 bucket 的 info 和 weight ──
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

    # ── 计算 allocations ──
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = total_samples * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # 舍入调整: adjust the most informative bucket
    diff = total_samples - allocated
    if diff != 0 and bucket_stats:
        most_info = max(
            bucket_stats,
            key=lambda l: bucket_stats[l]["weight"] / max(bucket_stats[l]["size"], 1),
        )
        allocations[most_info] = max(1, allocations[most_info] + diff)

    # ── 在每个 bucket 内采样 ──
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


    # ── 加权（power-law）sampling ──


def _bucket_rank_range(
    label: str,
    boundaries: tuple[int, ...],
) -> tuple[int, int]:
    """返回 bucket 标签对应的 (start_rank, end_rank)。

    示例：
        "1k"  → (1, 1000)
        "2k"  → (1001, 2000)
        "30k" → (20001, 30000)"""
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
    """使用 power-law weighted per-bucket sampling 从 ``vocab_bank`` 采样词。

    高频 buckets（低 rank）按比例得到更多样本，因为它们对整体词汇理解更有信息量。
    每个 bucket 的分配遵循：

        weight_b = size_b * (1 / median_rank_b) ** power

    其中 ``power`` 控制向高频词倾斜的程度。
    ``power=0`` → buckets 间均匀（与 bucket size 成比例）。
    ``power=1.0`` → 强高频偏置。

    Args:
        vocab_bank: 要采样的词库。
        total_samples: 总采样词数（默认 3000）。
        power: 基于 rank 加权的 power-law 指数（默认 0.7）。值越高越偏向高频词。
        seed: 用于复现的随机种子。

    Returns:
        带元数据的 ``SampledWord`` 列表。"""
    rng = random.Random(seed)
    sampled: list[SampledWord] = []

    # 计算每个 bucket 的 stats 和 weights
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

    # 计算每个 bucket 的 allocation
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = total_samples * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # 舍入调整：在最大 bucket 上加/减差值
    diff = total_samples - allocated
    if diff != 0 and bucket_stats:
        biggest = max(bucket_stats, key=lambda l: bucket_stats[l]["size"])
        allocations[biggest] = max(1, allocations[biggest] + diff)

    boundaries = vocab_bank.config.bucket_boundaries

    # 在每个 bucket 内采样
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
    """返回 weighted sampling 每个 bucket 分配量的报告。

    非破坏性：不执行实际采样，只展示每个 bucket 会抽取多少词。"""
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


# ── CLI 入口 ──

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
            # 对 uniform interval sampling，显示预测 counts
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


# ── 合成测试题采样（high-frequency weighted）──


def sample_test_questions(
    vocab_bank: VocabBank,
    n: int = 100,
    power: float = 0.5,
    seed: int | None = None,
) -> list[SampledWord]:
    """使用 frequency-weighted distribution 从 ``vocab_bank`` 抽取测试题。

    高频（低 rank）词出现在测试题中的概率更高，模拟真实考试中常见词更常被测试的设计。
    采样在每个 bucket 上遵循 power-law distribution：

        weight_bucket = size_bucket * (1 / median_rank_bucket) ** power

    然后按 bucket 分配量在每个 bucket 内抽词。

    Args:
        vocab_bank: 要采样的词库。
        n: 测试题数量（默认 100）。
        power: Power-law 指数（默认 0.5）。值越高越强烈偏向高频词。
        seed: 用于复现的随机种子。

    Returns:
        按 rank 排序、带元数据的 ``SampledWord`` 列表。"""
    rng = random.Random(seed)

    # ── 计算每个 bucket 的 weights ──
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

    # ── 计算每个 bucket 的 allocation ──
    allocations: dict[str, int] = {}
    allocated = 0
    for label, bs in bucket_stats.items():
        raw = n * bs["weight"] / total_weight
        alloc = max(1, int(round(raw)))
        allocations[label] = alloc
        allocated += alloc

    # 在最大 bucket 上调整舍入差值
    diff = n - allocated
    if diff != 0 and bucket_stats:
        biggest = max(bucket_stats, key=lambda l: bucket_stats[l]["size"])
        allocations[biggest] = max(1, allocations[biggest] + diff)

    # ── 在每个 bucket 内采样 ──
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
