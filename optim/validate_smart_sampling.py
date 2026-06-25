#!/usr/bin/env python3
"""
考试等级感知 smart sampling 设计的验证脚本。

比较：
  1. 旧版 power-law weighted sampling
  2. 新版 smart（基于 sigmoid）sampling
  3. 均匀 interval sampling

输出 allocation 表和 summary statistics。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optim.interval_sampler import (
    VocabBank, DEFAULT_CONFIG,
    describe_weighted_allocation,
    describe_smart_allocation,
    describe_sampling,
    weighted_sample_words,
    smart_sample_words,
    sample_words,
)


def print_separator(title: str) -> None:
    w = 70
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)


def main() -> None:
    bank = VocabBank(DEFAULT_CONFIG)
    total = 3000
    seed = 42

    # ── 1. 旧版 power-law allocation ──
    print_separator("旧方案：幂律加权采样 (power=0.7)")
    print(describe_weighted_allocation(bank, total_samples=total, power=0.7))

    # ── 2. 新版 smart allocation ──
    print_separator("新方案：考试等级边界导向智能采样 (α=4.0, γ=1.5)")
    print(describe_smart_allocation(bank, total_samples=total, alpha=4.0, info_exponent=1.5))

    # ── 3. 实际 smart sample ──
    print_separator("新方案：实际采样 3000 词 (seed=42)")
    sampled = smart_sample_words(bank, total_samples=total, alpha=4.0, info_exponent=1.5, seed=seed)
    print(describe_sampling(sampled, bank))

    # ── 4. 汇总对比表 ──
    print_separator("方案对比总结：分配表")
    
    old_sample = weighted_sample_words(bank, total_samples=total, power=0.7, seed=seed)
    old_counts: dict[str, int] = {}
    for s in old_sample:
        old_counts[s.bucket] = old_counts.get(s.bucket, 0) + 1

    smart_sample = sampled
    smart_counts: dict[str, int] = {}
    for s in smart_sample:
        smart_counts[s.bucket] = smart_counts.get(s.bucket, 0) + 1
    
    uniform_sample = sample_words(bank, interval=100, per_group=10, seed=seed)
    uniform_counts: dict[str, int] = {}
    for s in uniform_sample:
        uniform_counts[s.bucket] = uniform_counts.get(s.bucket, 0) + 1

    header = (
        f"  {'Bucket':>6s}  {'Bank':>6s}  {'Unif':>6s}  {'Unif%':>7s}"
        f"  {'Old':>6s}  {'Old%':>7s}"
        f"  {'Smart':>7s}  {'Smart%':>8s}  {'说明':>30s}"
    )
    sep = (
        f"  {'------':>6s}  {'------':>6s}  {'------':>6s}  {'-------':>7s}"
        f"  {'------':>6s}  {'-------':>7s}"
        f"  {'-------':>7s}  {'--------':>8s}  {'-'*30:>30s}"
    )
    print(header)
    print(sep)

    notes = {
        "1k": "几乎人人都会 → 最少采样",
        "2k": "中考边界 → 轻采样",
        "3k": "高考/中考边界 → 高密度",
        "5k": "四级/高考边界 → 最高密度",
        "8k": "六级/四级边界 → 最高密度",
        "10k": "考研/六级边界 → 中高密度",
        "15k": "高级/考研边界 → 中密度",
        "20k": "母语级/高级边界 → 中密度",
        "30k": "几乎都不认识 → 最少采样",
    }

    all_buckets = sorted(
        set(list(old_counts.keys()) + list(smart_counts.keys()) + list(uniform_counts.keys())),
        key=lambda b: (0, int(b.rstrip("k")) * 1000) if b.endswith("k") else (1, int(b)),
    )

    for bucket in all_buckets:
        bank_size = len(bank.words_by_bucket.get(bucket, []))
        uc = uniform_counts.get(bucket, 0)
        oc = old_counts.get(bucket, 0)
        sc = smart_counts.get(bucket, 0)
        note = notes.get(bucket, "")
        print(
            f"  {bucket:>6s}  {bank_size:>6d}"
            f"  {uc:>6d}  {100*uc/total:>6.1f}%"
            f"  {oc:>6d}  {100*oc/total:>6.1f}%"
            f"  {sc:>7d}  {100*sc/total:>7.1f}%"
            f"  {note:>30s}"
        )

    # ── 5. 对比 bucket 覆盖率 ──
    print_separator("各桶采样率对比 (Samples / Bank Size)")
    print(f"  {'Bucket':>6s}  {'Bank':>6s}  {'Uniform':>8s}  {'Old':>8s}  {'Smart':>8s}  {'Smart/Old':>10s}")
    print(f"  {'------':>6s}  {'------':>6s}  {'--------':>8s}  {'--------':>8s}  {'--------':>8s}  {'----------':>10s}")
    for bucket in all_buckets:
        bank_size = len(bank.words_by_bucket.get(bucket, []))
        uc = uniform_counts.get(bucket, 0)
        oc = old_counts.get(bucket, 0)
        sc = smart_counts.get(bucket, 0)
        ur = uc / bank_size if bank_size else 0
        oor = oc / bank_size if bank_size else 0
        sr = sc / bank_size if bank_size else 0
        ratio = sr / oor if oor else 0
        print(
            f"  {bucket:>6s}  {bank_size:>6d}"
            f"  {ur:>7.1%}"
            f"  {oor:>7.1%}"
            f"  {sr:>7.1%}"
            f"  {ratio:>9.2f}x"
        )

    total_bank = len(bank)
    total_unif = len(uniform_sample)
    total_old = len(old_sample)
    total_smart = len(smart_sample)
    print(f"  {'Total':>6s}  {total_bank:>6d}  {total_unif/ total_bank:>7.1%}  {total_old/ total_bank:>7.1%}  {total_smart/ total_bank:>7.1%}")


if __name__ == "__main__":
    main()
