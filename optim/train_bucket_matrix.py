#!/usr/bin/env python3
"""
train_bucket_matrix.py — 分桶矩阵参数模型

每个频段桶独立参数 θ_b（认知率基线），用户偏移 γ_u 跨桶共享。

模型:
  P_b = sigmoid(θ_b + γ_u)                         桶 B 认知概率
  raw_vocab = Σ[bucket_size_b × P_b]                原始词汇量
  calibrated = calibrate(raw_vocab, k, knots)       最终输出

损失 = Σ_u Σ_b (sigmoid(θ_b + γ_u) - true_rate_ub)² + L2正则

用法:
  python -m optim.train_bucket_matrix --epochs 2000 --n-questions 300
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
from optim.official_vocab import get_official_vocab_sets, get_set_words


# ── CET-6 词表加载（用于训练数据改造） ──
# 使用 data/exam_vocab/cet6.txt（全量 8028 词），匹配到词库约 7523 词

def load_cet6_matched_words(
    bank: VocabBank,
    cet6_path: str = "data/exam_vocab/cet6.txt",
) -> list[tuple[str, str, int]]:
    """加载 CET-6 词表中在词库中匹配的词。

    返回: [(word, bucket_label, rank), ...]
    只保留在词库中且能确定桶归属的词。
    """
    path = Path(__file__).resolve().parents[1] / cet6_path
    with open(path, encoding="utf-8") as f:
        raw_words = {w.strip().lower() for w in f if w.strip() and not w.startswith("#")}

    # 构建 word -> bucket 映射
    word_to_bucket: dict[str, str] = {}
    for label in BUCKET_LABELS:
        for item in bank.words_by_bucket.get(label, []):
            word_to_bucket[item.word] = label
            word_to_bucket[item.lemma] = label

    # 构建 lemma -> rank 映射
    lemma_to_rank: dict[str, int] = {}
    for item in bank.items:
        lemma_to_rank[item.lemma] = item.rank

    result: list[tuple[str, str, int]] = []
    for w in raw_words:
        bucket = word_to_bucket.get(w)
        if bucket:
            rank = bank.get_rank(w)
            if rank is None:
                lemma = bank.lemmatizer.normalize(w)
                rank = lemma_to_rank.get(lemma, 30000)
            result.append((w, bucket, rank))
        else:
            # 尝试 lemma
            lemma = bank.lemmatizer.normalize(w)
            bucket = word_to_bucket.get(lemma)
            if bucket:
                rank = lemma_to_rank.get(lemma, 30000)
                result.append((w, bucket, rank))

    print(f"  CET-6 matched words: {len(result)}")
    if result:
        ranks = [r for _, _, r in result]
        print(f"    rank range: [{min(ranks)} - {max(ranks)}]")
        print(f"    rank median: {sorted(ranks)[len(ranks)//2]}")

    return result

# ── 常量 ──
BUCKET_LABELS = ['1k', '2k', '3k', '5k', '8k', '10k', '15k', '20k', '30k']
N_BUCKETS = len(BUCKET_LABELS)
CALIB_BOUNDARIES = [3000, 8000, 22000]
MAX_V = 20000.0

# ── 默认锚点参数 ──
DEFAULT_TRANSITION_WIDTH = 2500.0  # sigmoid 平滑过渡宽度
DEFAULT_ANCHOR_WEIGHT = 0.5       # 锚点损失相对权重


# ═══════════════════════════════════════════════════════════════════
# 核心函数
# ═══════════════════════════════════════════════════════════════════

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


# ═══════════════════════════════════════════════════════════════════
# 考纲锚点准备
# ═══════════════════════════════════════════════════════════════════

def prepare_official_sets(
    bank: VocabBank,
    transition_width: float = DEFAULT_TRANSITION_WIDTH,
) -> list[dict]:
    """准备考纲词汇锚点集，预计算词-桶分布。

    对每个考纲词表（中考、高考、四级、六级），找出其单词在词库各桶中的
    分布，返回带 bucket_weights 的结构体用于锚点损失。
    """
    # 构建 word -> bucket 映射
    word_to_bucket: dict[str, str] = {}
    for label in BUCKET_LABELS:
        for item in bank.words_by_bucket.get(label, []):
            word_to_bucket[item.word] = label
            word_to_bucket[item.lemma] = label

    sets_meta = get_official_vocab_sets()
    official_sets: list[dict] = []

    for name, meta in sets_meta.items():
        words = get_set_words(name)
        bucket_counts = {l: 0 for l in BUCKET_LABELS}
        found = 0
        for w in words:
            b = word_to_bucket.get(w)
            if b:
                bucket_counts[b] += 1
                found += 1

        if found == 0:
            continue

        bucket_weights = np.array(
            [bucket_counts[l] / found for l in BUCKET_LABELS],
            dtype=float,
        )

        official_sets.append({
            'name': name,
            'expected_vocab_size': meta.expected_vocab_size,
            'expected_coverage': meta.expected_coverage,
            'weight': meta.weight,
            'bucket_weights': bucket_weights,
            'found_words': found,
            'bucket_counts': bucket_counts,
        })

    return official_sets


def expected_coverage_sigmoid(
    user_vocab: float | np.ndarray,
    exam_vocab_size: float | np.ndarray,
    transition_width: float = DEFAULT_TRANSITION_WIDTH,
) -> float | np.ndarray:
    """sigmoid 平滑映射的用户对考纲词表期望认知率。

    expected_cov = sigmoid((user_vocab - exam_vocab_size) / transition_width)

    当 user_vocab == exam_vocab_size 时，expected_cov = 0.5。
    当 user_vocab >> exam_vocab_size 时，→ 1.0。
    当 user_vocab << exam_vocab_size 时，→ 0.0。
    """
    return sigmoid((np.asarray(user_vocab) - exam_vocab_size) / transition_width)


def predict_bucket_rates(theta: np.ndarray, gamma: float) -> np.ndarray:
    """各桶认知概率 P_b = sigmoid(θ_b + γ).  shape (N_BUCKETS,)"""
    return sigmoid(theta + gamma)


def estimate_raw_vocab(theta: np.ndarray, gamma: float,
                       bucket_sizes: np.ndarray) -> float:
    """Σ bucket_size_b × P_b，用于计算 raw_vocab。"""
    rates = predict_bucket_rates(theta, gamma)
    return float(np.sum(bucket_sizes * rates))


def calibrate(raw_vocab: float, k: float, ks: list[float],
              max_v: float = MAX_V) -> float:
    """tanh → 分段线性校准.

    当 k <= 0 时跳过 tanh 阶段（适合桶矩阵模型）。
    """
    if k > 0:
        cal = max_v * math.tanh(k * raw_vocab)
    else:
        cal = raw_vocab
    prev_val, prev_b = 0.0, 0.0
    for i, bn in enumerate(CALIB_BOUNDARIES):
        if cal <= bn:
            return prev_val + (cal - prev_b) * ks[i]
        prev_val += (bn - prev_b) * ks[i]
        prev_b = bn
    return prev_val + (cal - prev_b) * ks[-1]


# ═══════════════════════════════════════════════════════════════════
# 合成训练数据生成（使用真实认知率，无采样噪声）
# ═══════════════════════════════════════════════════════════════════

def generate_synthetic_users(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions_total: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    生成合成用户，训练标签使用真实认知率（无采样噪声）。

    每个用户:
      - 已知词集: 从高频到低频填充至目标词汇量
      - 各桶真实认知率 = 已知词数 / 桶大小
      - 按桶大小采样测试题（有噪声版本，用作备选的 observed）
      - 记录 bucket_true_rates = 真实认知率
      - 记录 bucket_obs_rates = 从采样中观测到的（有噪声）
    """
    if vocab_sizes is None:
        vocab_sizes = [500, 1000, 2000, 3000, 5000, 7000,
                       8000, 10000, 12000, 15000, 18000, 20000]

    # 预计算各桶单词
    bucket_words: dict[str, list[str]] = {}
    for label in BUCKET_LABELS:
        bucket_words[label] = [
            it.word for it in bank.words_by_bucket.get(label, [])
        ]

    users: list[dict[str, Any]] = []

    for vi, vsize in enumerate(sorted(vocab_sizes)):
        rng = random.Random(seed + 1000 + vi)

        # ── 构建已知词集 ──
        known: set[str] = set()
        remaining = vsize
        inner_rng = random.Random(seed * 12345 + vi)

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                continue
            if remaining >= len(words):
                for w in words:
                    known.add(w)
                remaining -= len(words)
            else:
                chosen = inner_rng.sample(words, remaining)
                for w in chosen:
                    known.add(w)
                remaining = 0
                break

        # ── 真实认知率（干净的训练标签） ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样（有噪声的观测） ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n = n_per_bucket[label]
            if n >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            'vocab_size': vsize,
            'known_set_size': len(known),
            'responses': responses,
            'n_questions': len(responses),
            'bucket_true_rates': true_rates,
            'bucket_obs_rates': bucket_obs_rates,
            'bucket_counts': bucket_counts,
        })

    return users


# ═══════════════════════════════════════════════════════════════════
# CET-6 合成训练数据（基于考纲词表的训练用户）
# ═══════════════════════════════════════════════════════════════════

def generate_cet6_users(
    bank: VocabBank,
    cet6_matched: list[tuple[str, str, int]] | None = None,
    cet6_path: str = "data/exam_vocab/cet6.txt",
    min_vocab: int = 1000,
    max_vocab: int = 6000,
    step: int = 10,
    n_questions_total: int = 300,
    seed: int = 42,
    use_weighted: bool = True,
    weight_power: float = 0.5,
) -> list[dict[str, Any]]:
    """
    从 CET-6 考纲词表生成训练用户。

    对每个目标词汇量 N：
      - 从 CET-6 词表匹配到词库的 ~7523 词中以 1/rank^weight_power 加权抽样 N 词
        作为 known set
      - 各桶真实认知率 = known_in_bucket / bucket_size
      - 按桶大小采样测试题

    相比 generate_synthetic_users() 的区别：
      - 已知词来自 CET-6 考纲而非词频 rank 排序
      - 高频词（较低 rank）在抽样中权重更高，模拟真实学习路径
      - 中低频桶也有认知率 > 0（因为 CET-6 词分布在所有桶中）

    Args:
        bank: VocabBank 实例
        cet6_matched: 预加载的 CET-6 匹配词列表，None 则从文件加载
        cet6_path: CET-6 词表文件路径
        min_vocab: 最小目标词汇量
        max_vocab: 最大目标词汇量
        step: 步进值，总用户数 = (max_vocab - min_vocab) // step + 1
        n_questions_total: 每个用户的测试题数
        seed: 随机种子
        use_weighted: 是否使用加权抽样（True=1/√rank, False=纯随机）
        weight_power: 加权幂次，0.5=1/√rank, 1.0=1/rank
    """
    if cet6_matched is None:
        cet6_matched = load_cet6_matched_words(bank, cet6_path)

    # 预计算各桶的单词列表
    bucket_words: dict[str, list[str]] = {}
    for label in BUCKET_LABELS:
        bucket_words[label] = [
            it.word for it in bank.words_by_bucket.get(label, [])
        ]

    vocab_sizes = list(range(min_vocab, max_vocab + 1, step))
    n_users = len(vocab_sizes)
    print(f"  Target vocab sizes: {vocab_sizes[0]} to {vocab_sizes[-1]} (step={step}, n={n_users})")

    # 预计算抽样权重
    if use_weighted:
        raw_weights = np.array(
            [1.0 / max(r, 1) ** weight_power for _, _, r in cet6_matched],
            dtype=float,
        )
        probs = raw_weights / raw_weights.sum()
    else:
        probs = None

    cet6_word_list = [w for w, _, _ in cet6_matched]
    n_cet6_total = len(cet6_word_list)

    users: list[dict[str, Any]] = []

    for vi, vsize in enumerate(vocab_sizes):
        rng = random.Random(seed + 2000 + vi)

        # ── 从 CET-6 词表中加权抽样 N 个词 ──
        n = min(vsize, n_cet6_total)
        if use_weighted:
            chosen_indices = rng.choices(
                range(len(cet6_matched)), weights=list(probs), k=n
            )
            known_words: set[str] = set()
            for idx in chosen_indices:
                known_words.add(cet6_word_list[idx])
            while len(known_words) < n:
                extra = rng.choices(
                    range(len(cet6_matched)), weights=list(probs),
                    k=n - len(known_words),
                )
                for idx in extra:
                    known_words.add(cet6_word_list[idx])
        else:
            known_words = set(rng.sample(cet6_word_list, n))

        # ── 真实认知率（干净的训练标签） ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known_words)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样（有噪声的观测，同原始逻辑） ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known_words
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known_words),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    return users


# ═══════════════════════════════════════════════════════════════════
# 方案1：两阶段填充（词频打底 + CET-6 加权抽样）
# ═══════════════════════════════════════════════════════════════════

def generate_twostage_users(
    bank: VocabBank,
    cet6_matched: list[tuple[str, str, int]] | None = None,
    cet6_path: str = "data/exam_vocab/cet6.txt",
    min_vocab: int = 500,
    max_vocab: int = 7000,
    step: int = 5,
    base_vocab_size: int = 2500,
    n_questions_total: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    两阶段填充方案：
    1. 前 base_vocab_size 个词用纯词频填充（高频基础词，人人都该会）
    2. 剩余从 CET-6 词表按 1/rank 加权抽样（考纲词汇）

    模拟真实学习路径：先掌握高频基础词 → 再学 CET-6 考纲词。
    这样高频桶认知率高（因为基础词覆盖），中低频桶也有适度认知率（CET-6 词分布广）。
    """
    if cet6_matched is None:
        cet6_matched = load_cet6_matched_words(bank, cet6_path)

    # 预计算各桶单词
    bucket_words: dict[str, list[str]] = {}
    for label in BUCKET_LABELS:
        bucket_words[label] = [
            it.word for it in bank.words_by_bucket.get(label, [])
        ]

    # 预计算词频 rank 排序的词列表 (按 rank 升序 = 高频优先)
    all_words_by_rank: list[str] = sorted(
        (it.word for it in bank.items if it.word),
        key=lambda w: bank.get_rank(w) or 99999,
    )

    # CET-6 加权权重 (1/rank)
    cet6_word_list = [w for w, _, _ in cet6_matched]
    cet6_set = set(cet6_word_list)
    n_cet6_total = len(cet6_word_list)
    raw_weights = np.array(
        [1.0 / max(r, 1) for _, _, r in cet6_matched], dtype=float,
    )
    cet6_probs = raw_weights / raw_weights.sum()

    vocab_sizes = list(range(min_vocab, max_vocab + 1, step))
    n_users = len(vocab_sizes)
    print(f"  Target vocab sizes: {vocab_sizes[0]} to {vocab_sizes[-1]} (step={step}, n={n_users})")
    print(f"  Base freq fill: {base_vocab_size} words")

    users: list[dict[str, Any]] = []

    for vi, vsize in enumerate(vocab_sizes):
        rng = random.Random(seed + 3000 + vi)

        # ── 阶段1: 词频填充 ──
        n_freq = min(vsize, base_vocab_size)
        freq_known = set(all_words_by_rank[:n_freq])

        # ── 阶段2: CET-6 加权抽样剩余 ──
        n_cet6 = vsize - n_freq
        cet6_known: set[str] = set()
        if n_cet6 > 0:
            n_samp = min(n_cet6, n_cet6_total)
            chosen = rng.choices(
                range(len(cet6_matched)), weights=list(cet6_probs), k=n_samp
            )
            for idx in chosen:
                cet6_known.add(cet6_word_list[idx])
            while len(cet6_known) < n_samp:
                extra = rng.choices(
                    range(len(cet6_matched)), weights=list(cet6_probs),
                    k=n_samp - len(cet6_known),
                )
                for idx in extra:
                    cet6_known.add(cet6_word_list[idx])

        known = freq_known | cet6_known

        # ── 真实认知率 ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样 ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    return users


# ═══════════════════════════════════════════════════════════════════
# 方案1：两阶段填充（词频打底 + CET-6 1/√rank 加权，平衡数据集）
# ═══════════════════════════════════════════════════════════════════

def generate_cet6_simple_users(
    bank: VocabBank,
    n_samples: int = 2000,
    min_vocab: int = 50,
    max_vocab: int = 7000,
    seed: int = 42,
    n_questions_total: int = 100,
) -> list[dict[str, Any]]:
    """
    纯 CET-6 随机抽样的训练数据生成器。

    所有训练用户都来自 CET-6 考纲词表，vocab_size 从 min_vocab 到 max_vocab
    均匀分布 n_samples 个点。每个用户从 CET-6 词表中随机抽 vocab_size 个词
    作为 known set（纯随机，不加权）。
    """
    bucket_words: dict[str, list[str]] = {}
    for label in BUCKET_LABELS:
        bucket_words[label] = [
            it.word for it in bank.words_by_bucket.get(label, [])
        ]

    cet6_matched = load_cet6_matched_words(bank)
    cet6_word_list = [w for w, _, _ in cet6_matched]
    n_cet6_total = len(cet6_word_list)

    vocab_sizes = np.linspace(min_vocab, max_vocab, n_samples, dtype=int).tolist()
    vocab_sizes = sorted(set(vocab_sizes))
    print(f"  Target vocab sizes: {vocab_sizes[0]} to {vocab_sizes[-1]} "
          f"(n={len(vocab_sizes)})")

    users: list[dict[str, Any]] = []

    for vi, vsize in enumerate(vocab_sizes):
        rng = random.Random(seed + 7000 + vi)

        # 从 CET-6 词表中纯随机抽 vsize 个词
        n = min(vsize, n_cet6_total)
        known_words = set(rng.sample(cet6_word_list, n))

        # ── 真实认知率 ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known_words)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样 ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known_words
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known_words),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    return users


def generate_two_phase_users(
    bank: VocabBank,
    bucket_words: dict[str, list[str]] | None = None,
    cet6_words: list[tuple[str, str, int]] | None = None,
    cet6_path: str = "data/exam_vocab/cet6.txt",
    freq_fill: int = 2000,
    n_samples: int = 500,
    seed: int = 42,
    n_questions_total: int = 300,
) -> list[dict[str, Any]]:
    """
    方案1：两阶段填充训练数据生成器。

    生成两组用户（共 2×n_samples=1000 人），平衡数据集：

    **组 A（n_samples 人，two-phase 用户）**：
      对每个 vocab_size V（从 1000 到 6500，均匀分布 n_samples 个点）：
      - 如果 V ≤ freq_fill：known set = 词频 rank 前 V 个词
      - 如果 V > freq_fill：known set = 词频 rank 前 freq_fill 个词
        + 从 CET-6 词表中按 1/√rank 加权抽 (V - freq_fill) 个词
      - 计算各桶 true_rates

    **组 B（n_samples 人，纯词频填充用户）**：
      - vocab size 从 500 到 20000 均匀分布 n_samples 个点
      - 纯词频填充（前 V 个高频词）
    """

    if cet6_words is None:
        cet6_words = load_cet6_matched_words(bank, cet6_path)

    # 预计算各桶单词列表
    if bucket_words is None:
        bucket_words = {}
        for label in BUCKET_LABELS:
            bucket_words[label] = [
                it.word for it in bank.words_by_bucket.get(label, [])
            ]

    # 按 rank 升序排列的所有词（高频优先）
    all_words_by_rank: list[str] = sorted(
        (it.word for it in bank.items if it.word),
        key=lambda w: bank.get_rank(w) or 99999,
    )

    # CET-6 按 1/√rank 加权
    cet6_word_list = [w for w, _, _ in cet6_words]
    n_cet6_total = len(cet6_word_list)
    raw_weights = np.array(
        [1.0 / math.sqrt(max(r, 1)) for _, _, r in cet6_words], dtype=float,
    )
    cet6_probs = raw_weights / raw_weights.sum()

    users: list[dict[str, Any]] = []

    # ── 组 A: two-phase 用户（vocab 1000-6500，n_samples 个点） ──
    vocab_sizes_a = np.linspace(1000, 6500, n_samples, dtype=int).tolist()
    # 去重 + 排序
    vocab_sizes_a = sorted(set(vocab_sizes_a))

    print(f"  Two-phase group A: {len(vocab_sizes_a)} users, vocab "
          f"{vocab_sizes_a[0]} to {vocab_sizes_a[-1]}")
    print(f"  freq_fill cutoff: {freq_fill}")

    for vi, vsize in enumerate(vocab_sizes_a):
        rng = random.Random(seed + 5000 + vi)

        # ── 阶段1: 词频填充 ──
        if vsize <= freq_fill:
            # 只用语频填充
            known = set(all_words_by_rank[:vsize])
        else:
            # 词频填充前 freq_fill 个词
            known = set(all_words_by_rank[:freq_fill])
            # CET-6 加权抽样剩余 (V - freq_fill) 个词
            n_cet6 = vsize - freq_fill
            n_samp = min(n_cet6, n_cet6_total)
            chosen = rng.choices(
                range(len(cet6_words)), weights=list(cet6_probs), k=n_samp
            )
            cet6_known: set[str] = set()
            for idx in chosen:
                cet6_known.add(cet6_word_list[idx])
            # 去重可能不足，补抽
            while len(cet6_known) < n_samp:
                extra = rng.choices(
                    range(len(cet6_words)), weights=list(cet6_probs),
                    k=n_samp - len(cet6_known),
                )
                for idx in extra:
                    cet6_known.add(cet6_word_list[idx])
            known |= cet6_known

        # ── 真实认知率 ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样 ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    # ── 组 B: 纯词频填充用户（vocab 500-20000，n_samples 个点） ──
    vocab_sizes_b = np.linspace(500, 20000, n_samples, dtype=int).tolist()
    vocab_sizes_b = sorted(set(vocab_sizes_b))

    print(f"  Two-phase group B (freq-only): {len(vocab_sizes_b)} users, vocab "
          f"{vocab_sizes_b[0]} to {vocab_sizes_b[-1]}")

    for vi, vsize in enumerate(vocab_sizes_b):
        rng = random.Random(seed + 6000 + vi)

        # 纯词频填充
        known = set(all_words_by_rank[:vsize])

        # ── 真实认知率 ──
        true_rates = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样 ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses = []
        bucket_obs_rates = {}
        bucket_counts = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    print(f"  Two-phase total users: {len(users)}")
    return users


# ═══════════════════════════════════════════════════════════════════
# 方案3：课程体系合成（中考⊂高考⊂四级⊂六级 逐级掌握）
# ═══════════════════════════════════════════════════════════════════

def build_curriculum_sets(
    bank: VocabBank,
    cet6_matched: list[tuple[str, str, int]] | None = None,
) -> tuple[list[dict], int]:
    """
    构建课程体系词集：中考 ⊂ 高考 ⊂ 四级 ⊂ 六级（内置⊂全量CET-6）

    返回:
        levels: [
            {"name": str, "cum_size": int, "cum_set": set[str], "delta_set": set[str]},
            ...
        ]
        total_cet6_matched: 完整 CET-6 文件在词库中的匹配数

    cumulative sizes:
      zhongkao:  ~821
      gaokao:    ~1564
      cet4:      ~2630
      cet6_builtin: ~3050
      cet6_full:  ~7523
    """
    # 构建 word -> bucket 映射
    word_to_bucket: dict[str, str] = {}
    for label in BUCKET_LABELS:
        for item in bank.words_by_bucket.get(label, []):
            word_to_bucket[item.word] = label
            word_to_bucket[item.lemma] = label

    def _match_set(raw_words: set[str]) -> set[str]:
        """匹配词集到词库，返回匹配后的词集合。"""
        matched: set[str] = set()
        for w in raw_words:
            b = word_to_bucket.get(w)
            if b:
                matched.add(w)
                continue
            lemma = bank.lemmatizer.normalize(w)
            b = word_to_bucket.get(lemma)
            if b:
                matched.add(w)  # 存储原始 word form
        return matched

    # 加载考纲内置词集
    set_names = ['中考', '高考', '四级', '六级']
    raw_sets = {n: get_set_words(n) for n in set_names}

    levels: list[dict] = []
    prev_set: set[str] = set()
    prev_size = 0

    for name in set_names:
        raw = raw_sets[name]
        matched = _match_set(raw)
        # 该集合是累积的（按构造包含所有前序等级）
        cum_set = matched  # official_vocab 的列表本身就是累加的
        delta = cum_set - prev_set
        cum_size = len(cum_set)
        levels.append({
            'name': name,
            'cum_size': cum_size,
            'cum_set': cum_set,
            'delta_set': delta,
            'delta_size': len(delta),
        })
        prev_set = cum_set
        prev_size = cum_size

    # 第5层：全量 CET-6 文件中的全新词
    if cet6_matched is not None:
        cet6_file_words = {w for w, _, _ in cet6_matched}
    else:
        cet6_matched_loaded = load_cet6_matched_words(bank)
        cet6_file_words = {w for w, _, _ in cet6_matched_loaded}

    cet6_builtin = levels[-1]['cum_set']
    cet6_delta = cet6_file_words - cet6_builtin
    levels.append({
        'name': 'cet6_full',
        'cum_size': len(cet6_builtin | cet6_delta),
        'cum_set': cet6_builtin | cet6_delta,
        'delta_set': cet6_delta,
        'delta_size': len(cet6_delta),
    })

    total_match = len(cet6_file_words)

    print(f"  Curriculum sets (matched to bank):")
    for lv in levels:
        print(f"    {lv['name']:>12s}: {lv['cum_size']:>5d} cumulative  (+{lv['delta_size']:>4d} new)")

    return levels, total_match


def generate_curriculum_users(
    bank: VocabBank,
    curriculum_levels: list[dict] | None = None,
    cet6_matched: list[tuple[str, str, int]] | None = None,
    min_vocab: int = 500,
    max_vocab: int = 7000,
    step: int = 5,
    n_questions_total: int = 300,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    课程体系合成方案：
    按中考⊂高考⊂四级⊂六级⊂CET-6全量的累进掌握路径生成用户。

    对每个目标词汇量 N：
      1. 找到 N 所处的课程层级
      2. 所有已掌握层级的词全部设为已知
      3. 当前层级的词按 1/rank 加权抽样补足剩余词汇量

    这模拟了最真实的考试学习路径：按考试等级逐级往上，
    每一级内的词汇按频率从高到低掌握。
    """
    if curriculum_levels is None:
        curriculum_levels, _ = build_curriculum_sets(bank, cet6_matched)

    # 预计算各桶单词
    bucket_words: dict[str, list[str]] = {}
    for label in BUCKET_LABELS:
        bucket_words[label] = [
            it.word for it in bank.words_by_bucket.get(label, [])
        ]

    vocab_sizes = list(range(min_vocab, max_vocab + 1, step))
    n_users = len(vocab_sizes)
    print(f"  Target vocab sizes: {vocab_sizes[0]} to {vocab_sizes[-1]} (step={step}, n={n_users})")

    users: list[dict[str, Any]] = []

    for vi, vsize in enumerate(vocab_sizes):
        rng = random.Random(seed + 4000 + vi)

        # ── 确定层级 ──
        # 找到第一个 cum_size >= vsize 的层级
        target_level_idx = None
        for i, lv in enumerate(curriculum_levels):
            if lv['cum_size'] >= vsize:
                target_level_idx = i
                break

        if target_level_idx is None:
            # 超过所有层级，以最后一层作为池子
            target_level_idx = len(curriculum_levels) - 1

        known: set[str] = set()

        # 所有低层级完全掌握
        for i in range(target_level_idx):
            known.update(curriculum_levels[i]['cum_set'])

        # 当前层级：已有基础 known_base 个词，需要补足 vsize
        known_base = len(known)
        if target_level_idx > 0:
            known_base = curriculum_levels[target_level_idx - 1]['cum_size']

        needed = max(0, vsize - known_base)

        if needed > 0:
            # 当前层级的 delta 词集
            delta_set = curriculum_levels[target_level_idx]['delta_set']
            delta_list = list(delta_set)

            # 如果没有 CET-6 匹配数据，无法加权；用随机抽样
            if cet6_matched is not None and target_level_idx >= 3:
                # 只有第4+层级（六级+）有时间信息
                # 对所有 delta 词查 rank，按 1/rank 加权
                if target_level_idx >= 4:
                    # 最后一级 CET-6 全量文件：用 1/rank 加权
                    # 构建 rank 映射
                    cet6_word_to_rank: dict[str, int] = {}
                    for w, _, r in cet6_matched:
                        cet6_word_to_rank[w] = r

                    delta_with_rank = [
                        (w, cet6_word_to_rank.get(w, 30000)) for w in delta_list
                    ]
                    raw_w = np.array([1.0 / max(r, 1) for _, r in delta_with_rank], dtype=float)
                    delta_probs = raw_w / raw_w.sum() if raw_w.sum() > 0 else None

                    n_samp = min(needed, len(delta_list))
                    if delta_probs is not None and len(delta_list) > 0:
                        chosen = rng.choices(range(len(delta_list)), weights=list(delta_probs), k=n_samp)
                        for idx in chosen:
                            known.add(delta_list[idx])
                        while len(known) - known_base < n_samp:
                            extra = rng.choices(range(len(delta_list)), weights=list(delta_probs), k=n_samp - (len(known) - known_base))
                            for idx in extra:
                                known.add(delta_list[idx])
                    else:
                        # 随机抽样
                        samp = rng.sample(delta_list, min(needed, len(delta_list)))
                        known.update(samp)
                else:
                    # 低层级：纯随机抽样（这些词没有 rank 权重信息）
                    samp = rng.sample(delta_list, min(needed, len(delta_list)))
                    known.update(samp)
            else:
                # 没有 CET-6 数据或低层级
                samp = rng.sample(delta_list, min(needed, len(delta_list)))
                known.update(samp)

        # ── 真实认知率 ──
        true_rates: dict[str, float] = {}
        for label in BUCKET_LABELS:
            words = bucket_words[label]
            if not words:
                true_rates[label] = 0.0
                continue
            known_count = sum(1 for w in words if w in known)
            true_rates[label] = known_count / len(words)

        # ── 测试题采样 ──
        total_bank_words = sum(len(v) for v in bucket_words.values())
        n_per_bucket: dict[str, int] = {}
        for label in BUCKET_LABELS:
            raw = n_questions_total * len(bucket_words[label]) / total_bank_words
            n_per_bucket[label] = max(2, int(round(raw)))

        responses: list[tuple[str, bool]] = []
        bucket_obs_rates: dict[str, float] = {}
        bucket_counts: dict[str, int] = {}

        for label in BUCKET_LABELS:
            words = bucket_words[label]
            n_q = n_per_bucket[label]
            if n_q >= len(words):
                sampled = list(words)
            else:
                sampled = rng.sample(words, n_q)

            known_in_sample = 0
            for w in sampled:
                is_known = w in known
                responses.append((w, is_known))
                if is_known:
                    known_in_sample += 1

            bucket_obs_rates[label] = (
                known_in_sample / len(sampled) if sampled else 0.0
            )
            bucket_counts[label] = len(sampled)

        rng.shuffle(responses)

        users.append({
            "vocab_size": vsize,
            "known_set_size": len(known),
            "responses": responses,
            "n_questions": len(responses),
            "bucket_true_rates": true_rates,
            "bucket_obs_rates": bucket_obs_rates,
            "bucket_counts": bucket_counts,
        })

    return users


# ═══════════════════════════════════════════════════════════════════
# 参数优化（PyTorch 版，若可用；否则 NumPy）
# ═══════════════════════════════════════════════════════════════════

def train_params_torch(
    users: list[dict[str, Any]],
    bucket_sizes: np.ndarray,
    n_epochs: int = 1500,
    lr: float = 0.05,
    l2_theta: float = 0.001,
    l2_gamma: float = 0.001,
    official_sets: list[dict] | None = None,
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
    transition_width: float = DEFAULT_TRANSITION_WIDTH,
    verbose: bool = True,
    print_every: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """PyTorch Adam 优化，支持考纲锚点损失。"""
    import torch

    N = len(users)
    theta_t = torch.tensor(np.zeros(N_BUCKETS), dtype=torch.float32, requires_grad=True)
    gammas_t = torch.tensor(np.zeros(N), dtype=torch.float32, requires_grad=True)
    optim = torch.optim.Adam([theta_t, gammas_t], lr=lr)

    # 预计算真实率矩阵: (N, N_BUCKETS)
    true_rates_mat = torch.tensor(np.array([
        [users[ui]['bucket_true_rates'].get(l, 0.0) for l in BUCKET_LABELS]
        for ui in range(N)
    ]), dtype=torch.float32)

    # 有数据掩码
    has_data_mat = torch.tensor(np.array([
        [1 if users[ui]['bucket_counts'].get(l, 0) > 0 else 0 for l in BUCKET_LABELS]
        for ui in range(N)
    ]), dtype=torch.float32)

    bucket_sizes_t = torch.tensor(bucket_sizes, dtype=torch.float32)

    # ── 考纲锚点预计算 ──
    if official_sets:
        # 各考纲词集在桶上的归一化权重: (N_SETS, N_BUCKETS)
        bucket_weights_mat = torch.tensor(np.array([
            s['bucket_weights'] for s in official_sets
        ]), dtype=torch.float32)
        # 各考纲的期望词汇量: (N_SETS,)
        exp_vocab_sizes = torch.tensor(np.array([
            s['expected_vocab_size'] for s in official_sets
        ]), dtype=torch.float32)
        # 各考纲的损失权重: (N_SETS,)
        set_weights_t = torch.tensor(np.array([
            s['weight'] for s in official_sets
        ]), dtype=torch.float32)
        # 各用户的真实词汇量: (N,)
        user_vocabs_t = torch.tensor(np.array([
            u['vocab_size'] for u in users
        ]), dtype=torch.float32)

    best_loss = float('inf')
    best_state = (theta_t.detach().numpy().copy(), gammas_t.detach().numpy().copy())

    for epoch in range(1, n_epochs + 1):
        optim.zero_grad()

        # P_b = sigmoid(θ_b + γ_u)  →  形状 (N, N_BUCKETS)
        logits = theta_t.unsqueeze(0) + gammas_t.unsqueeze(1)  # (N, NB)
        rates = torch.sigmoid(logits)

        # 使用 observation mask 计算 rates 上的 MSE
        err = rates - true_rates_mat
        loss = torch.sum(has_data_mat * err ** 2)

        # ── 考纲锚点损失 ──
        if official_sets:
            # 预测认知率: (N, N_SETS) = (N, N_BUCKETS) @ (N_BUCKETS, N_SETS)
            pred_cov = rates @ bucket_weights_mat.T  # 形状：(N, N_SETS)

            # 期望认知率: sigmoid((user_vocab - exam_vocab) / transition_width)
            exp_cov = torch.sigmoid(
                (user_vocabs_t.unsqueeze(1) - exp_vocab_sizes.unsqueeze(0)) / transition_width
            )  # 形状：(N, N_SETS)

            # 加权锚点损失
            anchor_err = pred_cov - exp_cov  # 形状：(N, N_SETS)
            anchor_loss = torch.sum(set_weights_t.unsqueeze(0) * anchor_err ** 2)
            loss += anchor_weight * anchor_loss

        # L2
        loss += l2_theta * torch.sum(theta_t ** 2)
        loss += l2_gamma * torch.sum(gammas_t ** 2)

        loss.backward()
        optim.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = (
                theta_t.detach().numpy().copy(),
                gammas_t.detach().numpy().copy(),
            )

        if verbose and epoch % print_every == 0:
            anchor_info = ""
            if official_sets:
                al = anchor_loss.item() if 'anchor_loss' in dir() else 0
                anchor_info = f"  anchor_loss={anchor_loss.item():.4f}"
            print(f"  epoch {epoch:5d}/{n_epochs}  loss={loss_val:.6f}  best={best_loss:.6f}{anchor_info}")

    if verbose:
        print(f"  Done. best_loss={best_loss:.6f}")

    return best_state


def train_params_numpy(
    users: list[dict[str, Any]],
    bucket_sizes: np.ndarray,
    n_epochs: int = 1500,
    lr: float = 0.05,
    l2_theta: float = 0.001,
    l2_gamma: float = 0.001,
    official_sets: list[dict] | None = None,
    anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
    transition_width: float = DEFAULT_TRANSITION_WIDTH,
    verbose: bool = True,
    print_every: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy 版 Adam（备用），支持考纲锚点损失。"""
    N = len(users)
    theta = np.zeros(N_BUCKETS, dtype=float)
    gammas = np.zeros(N, dtype=float)

    # 预计算
    true_mat = np.array([
        [users[ui]['bucket_true_rates'].get(l, 0.0) for l in BUCKET_LABELS]
        for ui in range(N)
    ], dtype=float)
    has_data = np.array([
        [1 if users[ui]['bucket_counts'].get(l, 0) > 0 else 0 for l in BUCKET_LABELS]
        for ui in range(N)
    ], dtype=float)

    # ── 考纲锚点预计算 ──
    if official_sets:
        bucket_weights_mat = np.array([
            s['bucket_weights'] for s in official_sets
        ], dtype=float)  # 形状：(N_SETS, N_BUCKETS)
        exp_vocab_sizes = np.array([
            s['expected_vocab_size'] for s in official_sets
        ], dtype=float)  # 形状：(N_SETS,)
        set_weights = np.array([
            s['weight'] for s in official_sets
        ], dtype=float)  # 形状：(N_SETS,)
        user_vocabs = np.array([u['vocab_size'] for u in users], dtype=float)  # (N,)

    # Adam 更新 更新 状态
    m_t, v_t = np.zeros(N_BUCKETS), np.zeros(N_BUCKETS)
    m_g, v_g = np.zeros(N), np.zeros(N)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0

    best_loss = float('inf')
    best_theta = theta.copy()
    best_gammas = gammas.copy()

    for epoch in range(1, n_epochs + 1):
        # 前向计算
        logits = theta[np.newaxis, :] + gammas[:, np.newaxis]  # (N, NB)
        rates = sigmoid(logits)
        err = rates - true_mat

        loss_val = float(np.sum(has_data * err ** 2))
        loss_val += l2_theta * float(np.sum(theta ** 2))
        loss_val += l2_gamma * float(np.sum(gammas ** 2))

        # ── 考纲锚点损失 ──
        anchor_loss_val = 0.0
        if official_sets:
            # 预测认知率: (N, N_SETS) = (N, N_BUCKETS) @ (N_BUCKETS, N_SETS)
            pred_cov = rates @ bucket_weights_mat.T  # 形状：(N, N_SETS)

            # 期望认知率: sigmoid((user_vocab - exam_vocab) / transition_width)
            exp_cov = sigmoid(
                (user_vocabs[:, np.newaxis] - exp_vocab_sizes[np.newaxis, :]) / transition_width
            )

            # 加权锚点损失
            anchor_err = pred_cov - exp_cov
            anchor_loss_val = float(np.sum(set_weights[np.newaxis, :] * anchor_err ** 2))
            loss_val += anchor_weight * anchor_loss_val

        # MSE + L2 的 gradient
        d_sig = rates * (1.0 - rates)
        weighted = has_data * err * d_sig  # (N, NB)

        grad_theta = 2.0 * np.sum(weighted, axis=0) + 2.0 * l2_theta * theta
        grad_gamma = 2.0 * np.sum(weighted, axis=1) + 2.0 * l2_gamma * gammas

        # anchor loss 的 gradient
        if official_sets and anchor_weight > 0:
        # pred_cov 公式 = rates @ W.T，d_pred/d_rates = W.T
        # d_anchor/d_rates 公式 = 2 * set_weights * (pred_cov - exp_cov) * W（broadcast）
            # rates = sigmoid(t + g)，因此 d_rates/d_logits = d_sig
            anchor_grad_logits = 2.0 * (
                (set_weights[np.newaxis, :] * anchor_err) @ bucket_weights_mat
            )  # (N, NB)
            anchor_grad_logits *= d_sig  # 通过 sigmoid 应用 chain rule

            grad_theta += anchor_weight * np.sum(anchor_grad_logits, axis=0)
            grad_gamma += anchor_weight * np.sum(anchor_grad_logits, axis=1)

        # Adam 更新 更新
        t += 1
        m_t = beta1 * m_t + (1 - beta1) * grad_theta
        v_t = beta2 * v_t + (1 - beta2) * grad_theta ** 2
        m_hat = m_t / (1 - beta1 ** t)
        v_hat = v_t / (1 - beta2 ** t)
        theta -= lr * m_hat / (np.sqrt(v_hat) + eps)

        m_g = beta1 * m_g + (1 - beta1) * grad_gamma
        v_g = beta2 * v_g + (1 - beta2) * grad_gamma ** 2
        m_g_hat = m_g / (1 - beta1 ** t)
        v_g_hat = v_g / (1 - beta2 ** t)
        gammas -= lr * m_g_hat / (np.sqrt(v_g_hat) + eps)

        if loss_val < best_loss:
            best_loss = loss_val
            best_theta = theta.copy()
            best_gammas = gammas.copy()

        if verbose and epoch % print_every == 0:
            info = f"  anchor_loss={anchor_loss_val:.4f}" if official_sets else ""
            print(f"  epoch {epoch:5d}/{n_epochs}  loss={loss_val:.6f}  best={best_loss:.6f}{info}")

    if verbose:
        print(f"  Done. best_loss={best_loss:.6f}")

    return best_theta, best_gammas


# ═══════════════════════════════════════════════════════════════════
# 校准参数训练
# ═══════════════════════════════════════════════════════════════════

def train_calibration_params(
    users: list[dict[str, Any]],
    theta: np.ndarray,
    gammas: np.ndarray,
    bucket_sizes: np.ndarray,
    n_epochs: int = 100,
    lr_k: float = 0.0,
    lr_ks: float = 0.0,
    verbose: bool = True,
    print_every: int = 20,
) -> tuple[float, np.ndarray]:
    """训练校准参数。

    对于桶矩阵模型，raw_vocab 已经是词汇量的直接估计，
    所以校准使用恒等映射 (k=0 跳过 tanh, ks=[1,1,1])。
    """
    # 恒等映射：跳过 tanh（k=0），piecewise slopes = 1.0
    k = 0.0
    ks = np.array([1.0, 1.0, 1.0], dtype=float)

    mk, vk = 0.0, 0.0
    mks, vks = np.zeros(3), np.zeros(3)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    eps_fd = 1e-6

    best_loss = float('inf')
    best_k, best_ks = k, ks.copy()

    for epoch in range(1, n_epochs + 1):
        # 当前点的 loss
        total = 0.0
        for ui, user in enumerate(users):
            raw_v = estimate_raw_vocab(theta, gammas[ui], bucket_sizes)
            cal_v = calibrate(raw_v, k, ks)
            total += (cal_v - user['vocab_size']) ** 2
        loss = total / len(users)

        # ∇k
        lp, lm = 0.0, 0.0
        for ui, user in enumerate(users):
            raw_v = estimate_raw_vocab(theta, gammas[ui], bucket_sizes)
            lp += (calibrate(raw_v, k + eps_fd, ks) - user['vocab_size']) ** 2
            lm += (calibrate(raw_v, k - eps_fd, ks) - user['vocab_size']) ** 2
        grad_k = (lp - lm) / (2.0 * eps_fd * len(users))

        # ∇ks
        grad_ks = np.zeros(3)
        for i in range(3):
            lp_i, lm_i = 0.0, 0.0
            ksp = ks.copy(); ksp[i] += eps_fd
            ksm = ks.copy(); ksm[i] -= eps_fd
            for ui, user in enumerate(users):
                raw_v = estimate_raw_vocab(theta, gammas[ui], bucket_sizes)
                lp_i += (calibrate(raw_v, k, ksp) - user['vocab_size']) ** 2
                lm_i += (calibrate(raw_v, k, ksm) - user['vocab_size']) ** 2
            grad_ks[i] = (lp_i - lm_i) / (2.0 * eps_fd * len(users))

        # 为稳定性 clamp gradients
        grad_k = np.clip(grad_k, -1e-3, 1e-3)
        grad_ks = np.clip(grad_ks, -1.0, 1.0)

        t += 1

        mk = beta1 * mk + (1 - beta1) * grad_k
        vk = beta2 * vk + (1 - beta2) * grad_k ** 2
        mk_hat = mk / (1 - beta1 ** t)
        vk_hat = vk / (1 - beta2 ** t)
        k -= lr_k * mk_hat / (math.sqrt(vk_hat) + eps)

        mks = beta1 * mks + (1 - beta1) * grad_ks
        vks = beta2 * vks + (1 - beta2) * grad_ks ** 2
        mks_hat = mks / (1 - beta1 ** t)
        vks_hat = vks / (1 - beta2 ** t)
        ks -= lr_ks * mks_hat / (np.sqrt(vks_hat) + eps)

        # 不钳制 k 为正：k=0 表示跳过 tanh，用恒等校准
        np.clip(ks, 0.001, 10.0, out=ks)

        if loss < best_loss:
            best_loss = loss
            best_k, best_ks = k, ks.copy()

        if verbose and epoch % print_every == 0:
            print(f"  calib epoch {epoch:4d}/{n_epochs}  loss={loss:.1f}  k={k:.8f}  ks={ks}")

    if verbose:
        print(f"  Calibration done. best_loss={best_loss:.1f}")

    return best_k, best_ks


def compute_vocab_accuracy(
    users: list[dict[str, Any]],
    theta: np.ndarray,
    gammas: np.ndarray,
    bucket_sizes: np.ndarray,
    k: float,
    ks: np.ndarray,
) -> list[dict]:
    """计算每个用户的预测精度."""
    results = []
    for ui, user in enumerate(users):
        target = user['vocab_size']
        raw_v = estimate_raw_vocab(theta, gammas[ui], bucket_sizes)
        cal_v = calibrate(raw_v, k, ks)
        results.append({
            'target': target,
            'raw_vocab': raw_v,
            'calibrated': cal_v,
            'error': cal_v - target,
        })
    return results


def format_results(theta, users, gammas, bucket_sizes, k, ks):
    """格式化为输出."""
    lines = []

    # ── 9 个 Θ 参数 ──
    lines.append("Bucket matrix parameters:")
    lines.append("-" * 55)
    for i, label in enumerate(BUCKET_LABELS):
        p = 1.0 / (1.0 + math.exp(-theta[i]))
        lines.append(f"  {label:>4s}:  \u03b8={theta[i]:+.2f}  (P\u2248{p:.2f})")
    lines.append("")

    # ── 校准参数 ──
    knot_pairs = [(CALIB_BOUNDARIES[i], float(ks[i])) for i in range(3)]
    lines.append(f"Calibration: k={k:.7f}, knots={knot_pairs}")
    lines.append("")

    # ── 预测精度表 ──
    acc = compute_vocab_accuracy(users, theta, gammas, bucket_sizes, k, ks)
    lines.append(f"{'Target V':>10s}  {'Raw V':>8s}  {'Predicted V':>12s}  {'Error':>8s}")
    lines.append(f"{'-'*10}  {'-'*8}  {'-'*12}  {'-'*8}")
    total_abs_err = 0.0
    for r in acc:
        total_abs_err += abs(r['error'])
        lines.append(
            f"  {r['target']:>8d}  {r['raw_vocab']:>8.0f}"
            f"  {r['calibrated']:>10.0f}  {r['error']:>+8.0f}"
        )
    lines.append("")
    mae = total_abs_err / len(acc)
    lines.append(f"  Mean Absolute Error: {mae:.0f} words")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Train bucket matrix params.")
    parser.add_argument("--epochs", type=int, default=2000, help="Bucket param epochs")
    parser.add_argument("--calib-epochs", type=int, default=500, help="Calibration param epochs")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="Test questions per synthetic user")
    parser.add_argument("--output", default=None, help="Save params to JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-anchor", action="store_true",
                        help="Disable official vocab anchor loss")
    parser.add_argument("--anchor-weight", type=float, default=DEFAULT_ANCHOR_WEIGHT,
                        help=f"Anchor loss weight (default {DEFAULT_ANCHOR_WEIGHT})")
    parser.add_argument("--transition-width", type=float, default=DEFAULT_TRANSITION_WIDTH,
                        help=f"Sigmoid transition width (default {DEFAULT_TRANSITION_WIDTH})")
    # ── 生成模式 ──
    parser.add_argument("--two-phase", action="store_true",
                        help="Use two-phase training data generation (Scheme 1)")
    parser.add_argument("--two-phase-fill", type=int, default=2000,
                        help="Frequency fill cutoff for two-phase mode")
    parser.add_argument("--two-phase-samples", type=int, default=500,
                        help="Number of samples per group in two-phase mode (total=2x)")
    parser.add_argument("--gen-mode", choices=["freq_only", "hybrid", "twostage", "curriculum", "two_phase"],
                        default="freq_only",
                        help="Training data generation mode")
    parser.add_argument("--cet6-min", type=int, default=500,
                        help="CET-6 min vocab size")
    parser.add_argument("--cet6-max", type=int, default=7000,
                        help="CET-6 max vocab size")
    parser.add_argument("--cet6-step", type=int, default=5,
                        help="CET-6 vocab step size")
    parser.add_argument("--cet6-base", type=int, default=2500,
                        help="Frequency fill base for twostage mode")
    # ── CET-6 Simple 模式 ──
    parser.add_argument("--cet6-simple", action="store_true",
                        help="Use pure CET-6 random sampling training data generator")
    parser.add_argument("--cet6-simple-samples", type=int, default=2000,
                        help="Number of cet6-simple training samples")
    parser.add_argument("--cet6-simple-min", type=int, default=50,
                        help="Minimum vocab size for cet6-simple mode")
    parser.add_argument("--cet6-simple-max", type=int, default=7000,
                        help="Maximum vocab size for cet6-simple mode")
    args = parser.parse_args()

    # ── 加载词库 ──
    print("Loading VocabBank...")
    bank = VocabBank(DEFAULT_CONFIG)
    bucket_sizes_arr = np.array([
        len(bank.words_by_bucket.get(l, [])) for l in BUCKET_LABELS
    ], dtype=float)
    print(f"  Total words: {len(bank)}, buckets: {len(BUCKET_LABELS)}")
    for i, l in enumerate(BUCKET_LABELS):
        print(f"    {l}: {int(bucket_sizes_arr[i])} words")

    # ── 准备考纲锚点 ──
    official_sets = None
    if not args.no_anchor:
        print(f"\n{'='*60}")
        print("Preparing official exam vocab anchor sets...")
        print(f"{'='*60}")
        official_sets = prepare_official_sets(bank, transition_width=args.transition_width)
        for s in official_sets:
            bw = s['bucket_weights']
            top_bucket = BUCKET_LABELS[int(np.argmax(bw))]
            print(f"  {s['name']:>4s}: {s['found_words']:>5d} words in bank, "
                  f"exp_vocab={s['expected_vocab_size']:>4d}, "
                  f"weight={s['weight']:.1f}, "
                  f"peak_bucket={top_bucket}")
        print(f"  transition_width={args.transition_width}, anchor_weight={args.anchor_weight}")

    # ── 生成合成用户 ──
    # 总是生成 12 个词频填充用户作边界覆盖
    freq_vocab_sizes = [500, 1000, 2000, 3000, 5000, 7000,
                        8000, 10000, 12000, 15000, 18000, 20000]
    freq_users = generate_synthetic_users(
        bank, vocab_sizes=freq_vocab_sizes,
        n_questions_total=args.n_questions,
        seed=args.seed + 9999,
    )

    gen_mode = "cet6_simple" if args.cet6_simple else ("two_phase" if args.two_phase else args.gen_mode)

    if gen_mode == "cet6_simple":
        print(f"\n{'='*60}")
        print("GEN MODE: cet6_simple — pure CET-6 random sampling")
        print(f"{'='*60}")
        print(f"  samples={args.cet6_simple_samples}, min={args.cet6_simple_min}, max={args.cet6_simple_max}")
        simple_users = generate_cet6_simple_users(
            bank,
            n_samples=args.cet6_simple_samples,
            min_vocab=args.cet6_simple_min,
            max_vocab=args.cet6_simple_max,
            seed=args.seed,
            n_questions_total=args.n_questions,
        )
        main_users = []
        freq_users = simple_users
        n_simple = len(simple_users)
        training_mode_info = {
            "mode": "cet6_simple",
            "total_users": n_simple,
            "cet6_simple_samples": args.cet6_simple_samples,
            "cet6_simple_min": args.cet6_simple_min,
            "cet6_simple_max": args.cet6_simple_max,
        }
        final_vocab_sizes = sorted(set([u['vocab_size'] for u in simple_users]))
        print(f"\n  CET-6 simple users: {n_simple}")

    elif gen_mode == "freq_only":
        print(f"\nGen mode: freq_only — only {len(freq_users)} frequency-filled users")
        main_users = []
        training_mode_info = {"mode": "frequency_only"}
        final_vocab_sizes = freq_vocab_sizes

    elif gen_mode == "hybrid":
        print(f"\n{'='*60}")
        print("GEN MODE: hybrid — pure CET-6 weighted sampling + freq users")
        print(f"{'='*60}")
        cet6_matched = load_cet6_matched_words(bank)
        cet6_users = generate_cet6_users(
            bank, cet6_matched=cet6_matched,
            min_vocab=args.cet6_min, max_vocab=args.cet6_max, step=args.cet6_step,
            n_questions_total=args.n_questions, seed=args.seed,
            use_weighted=True, weight_power=1.0,
        )
        main_users = cet6_users
        n_cet6 = len(cet6_users)
        training_mode_info = {
            "mode": "cet6_hybrid",
            "cet6_users": n_cet6, "freq_users": len(freq_users),
            "cet6_weighted": True, "cet6_weight_power": 1.0,
            "cet6_vocab_range": [args.cet6_min, args.cet6_max],
            "cet6_step": args.cet6_step,
        }
        final_vocab_sizes = sorted(
            list(range(args.cet6_min, args.cet6_max + 1, args.cet6_step))
            + freq_vocab_sizes
        )
        print(f"\n  CET-6 users: {n_cet6}")

    elif gen_mode == "twostage":
        print(f"\n{'='*60}")
        print("GEN MODE: twostage — freq fill base + CET-6 weighted remainder")
        print(f"{'='*60}")
        print(f"  Base freq fill: {args.cet6_base} words")
        cet6_matched = load_cet6_matched_words(bank)
        ts_users = generate_twostage_users(
            bank, cet6_matched=cet6_matched,
            min_vocab=args.cet6_min, max_vocab=args.cet6_max, step=args.cet6_step,
            base_vocab_size=args.cet6_base,
            n_questions_total=args.n_questions, seed=args.seed,
        )
        main_users = ts_users
        n_ts = len(ts_users)
        training_mode_info = {
            "mode": "twostage",
            "twostage_users": n_ts, "freq_users": len(freq_users),
            "base_freq_fill": args.cet6_base,
            "cet6_vocab_range": [args.cet6_min, args.cet6_max],
            "cet6_step": args.cet6_step,
        }
        final_vocab_sizes = sorted(
            list(range(args.cet6_min, args.cet6_max + 1, args.cet6_step))
            + freq_vocab_sizes
        )
        print(f"\n  Twostage users: {n_ts}")

    elif gen_mode == "two_phase":
        print(f"\n{'='*60}")
        print("GEN MODE: two_phase — freq fill base + CET-6 1/√rank weighted")
        print(f"{'='*60}")
        print(f"  two_phase_fill={args.two_phase_fill}, two_phase_samples={args.two_phase_samples}")
        # 预计算 bucket_words
        bucket_words: dict[str, list[str]] = {}
        for label in BUCKET_LABELS:
            bucket_words[label] = [
                it.word for it in bank.words_by_bucket.get(label, [])
            ]
        cet6_matched = load_cet6_matched_words(bank)
        tp_users = generate_two_phase_users(
            bank, bucket_words=bucket_words, cet6_words=cet6_matched,
            freq_fill=args.two_phase_fill,
            n_samples=args.two_phase_samples,
            seed=args.seed,
            n_questions_total=args.n_questions,
        )
        # 在 two_phase 模式下，users 已包含 A/B 两组
        # 由于组 B 已覆盖，不再单独添加 freq_users
        main_users = []
        freq_users = tp_users
        n_tp = len(tp_users)
        training_mode_info = {
            "mode": "two_phase",
            "total_users": n_tp,
            "two_phase_fill": args.two_phase_fill,
            "two_phase_samples": args.two_phase_samples,
        }
        final_vocab_sizes = sorted(set(
            [u['vocab_size'] for u in tp_users]
        ))
        print(f"\n  Two-phase users: {n_tp}")

    elif gen_mode == "curriculum":
        print(f"\n{'='*60}")
        print("GEN MODE: curriculum — exam level progression (中考⊂高考⊂四级⊂六级)")
        print(f"{'='*60}")
        cet6_matched = load_cet6_matched_words(bank)
        cur_users = generate_curriculum_users(
            bank, cet6_matched=cet6_matched,
            min_vocab=args.cet6_min, max_vocab=args.cet6_max, step=args.cet6_step,
            n_questions_total=args.n_questions, seed=args.seed,
        )
        main_users = cur_users
        n_cur = len(cur_users)
        training_mode_info = {
            "mode": "curriculum",
            "curriculum_users": n_cur, "freq_users": len(freq_users),
            "cet6_vocab_range": [args.cet6_min, args.cet6_max],
            "cet6_step": args.cet6_step,
        }
        final_vocab_sizes = sorted(
            list(range(args.cet6_min, args.cet6_max + 1, args.cet6_step))
            + freq_vocab_sizes
        )
        print(f"\n  Curriculum users: {n_cur}")

    users = main_users + freq_users
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(users)} users ({len(main_users)} main + {len(freq_users)} freq)")
    print(f"{'='*60}")

    # ── Phase 1: 训练 θ 和 γ ──
    print(f"\n{'='*60}")
    print("Phase 1: Training θ (bucket baselines) and γ (user offsets)")
    print(f"{'='*60}")
    print(f"  epochs={args.epochs}, lr={args.lr}")
    if official_sets:
        print(f"  anchor_weight={args.anchor_weight}, transition_width={args.transition_width}")

    try:
        import torch
        print("  Using PyTorch optimizer")
        theta, gammas = train_params_torch(
            users, bucket_sizes_arr,
            n_epochs=args.epochs, lr=args.lr,
            official_sets=official_sets,
            anchor_weight=args.anchor_weight,
            transition_width=args.transition_width,
            verbose=True, print_every=200,
        )
    except ImportError:
        print("  PyTorch not available, using NumPy")
        theta, gammas = train_params_numpy(
            users, bucket_sizes_arr,
            n_epochs=args.epochs, lr=args.lr,
            official_sets=official_sets,
            anchor_weight=args.anchor_weight,
            transition_width=args.transition_width,
            verbose=True, print_every=200,
        )

    # ── Phase 2: 训练 k 和 ks（校准参数） ──
    print(f"\n{'='*60}")
    print("Phase 2: Training calibration params k and knot slopes")
    print(f"{'='*60}")

    k, ks = train_calibration_params(
        users, theta, gammas, bucket_sizes_arr,
        n_epochs=args.calib_epochs,
        lr_k=1e-9, lr_ks=0.001,
        verbose=True, print_every=100,
    )

    # ── 输出 ──
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(format_results(theta, users, gammas, bucket_sizes_arr, k, ks))

    # ── 考纲覆盖率分析 ──
    if official_sets:
        print(f"\n{'='*60}")
        print("Official Exam Coverage Analysis")
        print(f"{'='*60}")
        print(f"  {'User V':>8s}  ", end="")
        for s in official_sets:
            print(f"  {s['name']:>4s}_pred  {s['name']:>4s}_exp  ", end="")
        print()
        for ui, user in enumerate(users):
            logits = theta[np.newaxis, :] + gammas[ui]
            rates = sigmoid(logits)  # 形状：(N_BUCKETS,)
            print(f"  {user['vocab_size']:>8d}  ", end="")
            for s in official_sets:
                pred = float(np.sum(s['bucket_weights'] * rates))
                exp = float(expected_coverage_sigmoid(
                    user['vocab_size'], s['expected_vocab_size'], args.transition_width
                ))
                print(f"  {pred:>6.3f}  {exp:>6.3f}  ", end="")
            print()

    # ── 保存参数 ──
    if args.output:
        params = {
            "model": "bucket_matrix",
            "theta": {BUCKET_LABELS[i]: float(theta[i]) for i in range(N_BUCKETS)},
            "calibration_k": float(k),
            "piecewise_knots": [
                [CALIB_BOUNDARIES[i], float(ks[i])] for i in range(3)
            ],
            "bucket_sizes": {BUCKET_LABELS[i]: int(bucket_sizes_arr[i]) for i in range(N_BUCKETS)},
            "vocab_sizes_trained": final_vocab_sizes,
            "n_users": len(users),
            "training_epochs": args.epochs,
            "anchor_training": not args.no_anchor,
            "anchor_weight": args.anchor_weight,
            "transition_width": args.transition_width,
            "training_mode": training_mode_info,
            "gammas": [float(g) for g in gammas],
            "accuracy": {
                f"{ui:04d}_{user['vocab_size']}": {
                    "raw_vocab": float(estimate_raw_vocab(theta, gammas[ui], bucket_sizes_arr)),
                    "calibrated": float(calibrate(
                        estimate_raw_vocab(theta, gammas[ui], bucket_sizes_arr),
                        k, ks
                    )),
                    "error": float(calibrate(
                        estimate_raw_vocab(theta, gammas[ui], bucket_sizes_arr),
                        k, ks
                    ) - user['vocab_size'])
                }
                for ui, user in enumerate(users)
            }
        }
        Path(args.output).write_text(
            json.dumps(params, ensure_ascii=False, indent=2)
        )
        print(f"\nParameters saved to {args.output}")

    # ── 精度汇总 ──
    print(f"\n{'='*60}")
    print("Accuracy Summary")
    print(f"{'='*60}")
    acc = compute_vocab_accuracy(users, theta, gammas, bucket_sizes_arr, k, ks)
    print(f"  {'Target':>8s}  {'Raw V':>8s}  {'Calibrated':>10s}  {'Error':>8s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    total_err = 0.0
    for r in acc:
        total_err += abs(r['error'])
        print(f"  {r['target']:>8d}  {r['raw_vocab']:>8.0f}  {r['calibrated']:>10.0f}  {r['error']:>+8.0f}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}")
    print(f"  MAE = {total_err/len(acc):.0f} words")


if __name__ == "__main__":
    main()
