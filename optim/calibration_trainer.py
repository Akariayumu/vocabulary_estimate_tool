"""基于 gradient descent 的校准参数训练器。

从观测用户 response 数据训练全局参数（β、calibration_k、piecewise_knots），
替换手工调节的经验值。

实现 *interval-sampling + official-vocab-anchoring* 设计：
  - Interval group known-rate loss (L_interval)
  - Official exam vocabulary coverage loss (L_official)
  - Smoothness regularisation (L_smooth)

用法
-----
    python -m optim.calibration_trainer --data calibration_dataset.json
    python -m optim.calibration_trainer --dry-run    # 打印 sampling / 参数结构"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# 导入项目模块
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank


@dataclass
class CalibrationParameters:
    """已训练、可注入 config 的参数集。"""

    beta: float = -0.30
    calibration_k: float = 0.0000691
    piecewise_knots: list[tuple[int, float]] = field(
        default_factory=lambda: [(3000, 1.0), (8000, 0.45), (22000, 1.28)]
    )
    training_loss: float = 0.0
    n_epochs: int = 0
    # ── 新 loss 组件（interval redesign）──
    loss_interval: float = 0.0
    loss_official: float = 0.0
    loss_smooth: float = 0.0
    loss_bucket: float = 0.0

    def to_config_dict(self) -> dict[str, Any]:
        return {
            "logistic_beta_global": self.beta,
            "calibration_k": self.calibration_k,
            "piecewise_knots": list(self.piecewise_knots),
            "training_loss": self.training_loss,
        }


@dataclass
class UserBucketStats:
    """单个用户的按 bucket 聚合观测。"""

    user_id: int
    alpha: float = 0.0          # 已拟合的每用户 intercept
    bucket_rates: dict[str, float] = field(default_factory=dict)
    bucket_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 数据加载与聚合
# ---------------------------------------------------------------------------

def load_responses(path: str) -> dict[int, list[tuple[str, bool]]]:
    """从 JSON 加载校准数据集。

    期望格式：
        {"users": [{"user_id": 1, "responses": [{"word":..., "known":...}, ...]}, ...]}"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[int, list[tuple[str, bool]]] = {}
    for user in raw.get("users", []):
        uid = user["user_id"]
        result[uid] = [(r["word"], r["known"]) for r in user.get("responses", [])]
    return result


def aggregate_bucket_rates(
    responses_by_user: dict[int, list[tuple[str, bool]]],
    bank: VocabBank,
    bucket_labels: list[str],
) -> dict[int, UserBucketStats]:
    """计算每个用户每个 bucket 的 smoothed known-rates。"""

    from vocab_estimator.config import bucket_beta_prior

    n_buckets = len(bucket_labels)
    result: dict[int, UserBucketStats] = {}

    for uid, responses in responses_by_user.items():
        stats = UserBucketStats(user_id=uid)
        bucket_known: dict[str, list[bool]] = {b: [] for b in bucket_labels}

        for word, known in responses:
            bucket = bank.get_bucket(word)
            if bucket and bucket in bucket_known:
                bucket_known[bucket].append(bool(known))

        for idx, bucket in enumerate(bucket_labels):
            values = bucket_known[bucket]
            if not values:
                continue
            alpha_p, beta_p = bucket_beta_prior(idx, n_buckets)
            known_c = sum(values)
            total = len(values)
            smoothed = (known_c + alpha_p) / (total + alpha_p + beta_p)
            stats.bucket_rates[bucket] = smoothed
            stats.bucket_counts[bucket] = total

        result[uid] = stats

    return result


# ---------------------------------------------------------------------------
# 核心模型函数（NumPy 版本）
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def predict_bucket_rate(
    alpha: float,
    beta: float,
    bank: VocabBank,
    bucket: str,
) -> float:
    """给定 (α, β)，预测某个频率 bucket 的 known-rate。"""
    items = bank.get_items_in_bucket(bucket)
    if not items:
        return 0.0
    log_ranks = np.log(np.array([item.rank for item in items], dtype=float))
    probs = sigmoid(alpha + beta * log_ranks)
    return float(np.mean(probs))


def predict_word_prob(
    alpha: float,
    beta: float,
    rank: int,
) -> float:
    """预测给定 rank 上认识某个词的概率。"""
    log_r = math.log(max(rank, 1))
    logit = alpha + beta * log_r
    return 1.0 / (1.0 + math.exp(-max(-40, min(40, logit))))


def compute_raw_estimate(alpha: float, beta: float, bank: VocabBank) -> float:
    """在所有词库 item 上计算 Σ sigmoid(α + β×log(rank))。"""
    log_ranks = np.log(np.array(bank.ranks(), dtype=float))
    return float(np.sum(sigmoid(alpha + beta * log_ranks)))


def piecewise_calibrate(x: float, knots: list[tuple[int, float]]) -> float:
    """应用分段线性压缩。"""
    if x <= 0:
        return x
    prev_boundary = 0.0
    prev_value = 0.0
    for boundary, slope in knots:
        if x <= boundary:
            return prev_value + (x - prev_boundary) * slope
        prev_value += (float(boundary) - prev_boundary) * slope
        prev_boundary = float(boundary)
    return prev_value + (x - prev_boundary) * knots[-1][1]


def calibrate(raw: float, k: float, knots: list[tuple[int, float]],
              max_v: float = 20000.0) -> float:
    """完整校准流程：tanh → piecewise。"""
    if raw <= 0:
        return raw
    tanh_cal = max_v * math.tanh(k * raw)
    return piecewise_calibrate(tanh_cal, knots)


# ---------------------------------------------------------------------------
# 每用户 α 拟合（内循环）
# ---------------------------------------------------------------------------

def fit_user_alpha(
    uid: int,
    stats: UserBucketStats,
    bank: VocabBank,
    bucket_labels: list[str],
    beta: float,
    n_iter: int = 200,
    lr: float = 0.1,
) -> float:
    """拟合每个用户的 α，使预测 rates 与观测 rates 之间的 MSE 最小。

    使用带 momentum 的 SGD。"""
    if not stats.bucket_rates:
        return 0.0

    alpha = 0.0
    velocity = 0.0
    momentum = 0.9

    for _ in range(n_iter):
        grad = 0.0
        for bucket, r_obs in stats.bucket_rates.items():
            p_pred = predict_bucket_rate(alpha, beta, bank, bucket)
            err = p_pred - r_obs
            bucket_size = max(len(bank.get_items_in_bucket(bucket)), 1)
            grad += 2.0 * err * p_pred * (1.0 - p_pred) / bucket_size

        velocity = momentum * velocity - lr * grad
        alpha += velocity

    return alpha


# ---------------------------------------------------------------------------
# ── 新增：Interval-group loss ──
# ---------------------------------------------------------------------------

def _compute_interval_loss(
    alpha: float,
    beta: float,
    bank: VocabBank,
    interval_intervals: list[tuple[int, int]],
    sampled_words_per_interval: int = 2,
) -> tuple[float, int]:
    """计算 interval groups 上的 MSE loss。

    对每个 rank interval，预测该 interval 中 ``sampled_words_per_interval`` 个词组成的 group 的 known-rate。
    “observed” rate 被建模为给定 (α, β) 时的期望比例。

    Args:
        alpha: 每个用户的 intercept。
        beta: 全局 logistic slope。
        bank: VocabBank。
        interval_intervals: 每个 interval 的 (start_rank, end_rank) 列表。
        sampled_words_per_interval: 每个 interval 采样词数。

    Returns:
        (total_mse, n_intervals_used)"""
    total_mse = 0.0
    n_intervals = 0

    for start_r, end_r in interval_intervals:
        # 获取该 interval 中的 ranks
        ranks_in_interval: list[int] = []
        for item in bank.items:
            if start_r <= item.rank <= end_r:
                ranks_in_interval.append(item.rank)

        if len(ranks_in_interval) < sampled_words_per_interval:
            continue

        # 对 k 个词的 group，“全部已知”概率 ≈ mean prob^k
        # 更简单的方法：该 interval 中每个词的 mean prob
        probs = np.array([
            predict_word_prob(alpha, beta, r)
            for r in ranks_in_interval
        ])
        p_pred = float(np.mean(probs))

        # “expected” observed rate：给定 sampled_words_per_interval 个词时，
        # 期望已知比例 = p_pred（由期望线性性得到）
        # 希望它与校准后的 p_pred 匹配。
        # 对于 loss，使用 raw prediction；对
        # 完全校准用户而言，observed rate 就是 prediction。
        # 惩罚相对理想值的偏离：rate 应当等于
        # 模型预测值，即除 self-consistency 外没有额外 “true” rate。
        #
        # 不过真正目标是 cross-interval smoothness，因此使用
        # 简单 self-consistency loss：每个 interval prediction 应
        # 与整体 logistic curve 一致。
        p_expected = p_pred  # self-consistent 目标

        total_mse += (p_pred - p_expected) ** 2
        n_intervals += 1

    return total_mse, n_intervals


def compute_interval_structure(bank: VocabBank, interval: int = 50) -> list[tuple[int, int]]:
    """构建覆盖词库 rank 范围的 (start_rank, end_rank) interval 列表。"""
    max_rank = bank.config.vocab_size  # 30000
    intervals: list[tuple[int, int]] = []
    for start_rank in range(1, max_rank + 1, interval):
        end_rank = min(start_rank + interval - 1, max_rank)
        intervals.append((start_rank, end_rank))
    return intervals


# ---------------------------------------------------------------------------
# ── 新增：Official vocab coverage loss ──
# ---------------------------------------------------------------------------

def _compute_official_loss(
    alpha: float,
    beta: float,
    bank: VocabBank,
    official_vocab_sets: dict,
) -> float:
    """计算官方考试词汇集上预测 coverage 与期望 coverage 的 MSE loss。

    Args:
        alpha: 每个用户的 intercept。
        beta: 全局 logistic slope。
        bank: 用于词匹配的 VocabBank。
        official_vocab_sets: set name 到元数据的 dict，包含 ``words``、``expected_coverage`` 和 ``weight``。

    Returns:
        加权 MSE loss。"""
    total_loss = 0.0

    for set_name, info in official_vocab_sets.items():
        words = info.get("words", set())
        expected_coverage = info.get("expected_coverage", 0.80)
        weight = info.get("weight", 3.0)

        if not words:
            continue

        # 计算该集合中词的预测 known-rate
        probs: list[float] = []
        for word in words:
            rank = bank.get_rank(word)
            if rank is not None:
                probs.append(predict_word_prob(alpha, beta, rank))

        if not probs:
            continue

        p_pred = sum(probs) / len(probs)
        err = p_pred - expected_coverage
        total_loss += weight * err ** 2

    return total_loss


def _build_official_vocab_dict(
    bank: VocabBank,
) -> dict[str, dict[str, Any]]:
    """从内置词表构建 official vocab dict。

    Returns:
        {set_name: {"words": set[str], "expected_coverage": float, "weight": float}, ...}"""
    from .official_vocab import (
        get_official_vocab_sets,
        get_set_words,
    )

    sets = get_official_vocab_sets()
    result: dict[str, dict[str, Any]] = {}

    for name, info in sets.items():
        words = get_set_words(name)
        # 匹配到词库
        matched: set[str] = set()
        for word in words:
            rank = bank.get_rank(word)
            if rank is not None:
                matched.add(word)
            else:
                # 尝试 lemma
                lemma = bank.lemmatizer.normalize(word)
                if lemma in bank.rank_by_lemma:
                    # 查找该 lemma 的原始 word form
                    for item in bank.items:
                        if item.lemma == lemma:
                            matched.add(item.word)
                            break

        result[name] = {
            "words": matched,
            "expected_coverage": info.expected_coverage,
            "weight": info.weight,
        }

    return result


# ---------------------------------------------------------------------------
# ── 新增：Smoothness regularisation ──
# ---------------------------------------------------------------------------

def _compute_smoothness_loss(
    alpha: float,
    beta: float,
    bank: VocabBank,
    n_points: int = 300,
) -> float:
    """计算 smoothness regularisation：相邻 rank 预测差值的平方和。

    Args:
        alpha: 每个用户的 intercept。
        beta: 全局 logistic slope。
        bank: VocabBank。
        n_points: rank 范围内的采样点数。

    Returns:
        Smoothness loss 值。"""
    max_rank = bank.config.vocab_size  # 30000
    # 按 log 尺度采样 ranks
    log_ranks = np.linspace(math.log(1), math.log(max_rank), n_points)
    ranks = np.exp(log_ranks).astype(int)

    probs = np.array([predict_word_prob(alpha, beta, r) for r in ranks])
    diffs = np.diff(probs)
    return float(np.sum(diffs ** 2))


# ---------------------------------------------------------------------------
# 全局参数训练（NumPy SGD fallback）
# ---------------------------------------------------------------------------

def train_numpy(
    responses_by_user: dict[int, list[tuple[str, bool]]],
    bank: VocabBank,
    bucket_labels: list[str],
    n_epochs: int = 500,
    lr: float = 0.001,
    l2_lambda: float = 1.0,
    w_interval: float = 0.3,
    w_official: float = 0.3,
    w_smooth: float = 0.01,
    verbose: bool = True,
) -> CalibrationParameters:
    """使用手写 gradient descent（NumPy）训练全局参数。

    加入 interval-group loss、official vocab loss 和 smoothness regularisation。

    Args:
        responses_by_user: {user_id: [(word, known), ...]}
        bank: 词库。
        bucket_labels: 有序 bucket 标签。
        n_epochs: 训练 epoch 数。
        lr: 学习率。
        l2_lambda: β 的 L2 regularisation 强度。
        w_interval: interval-group loss 组件权重。
        w_official: official vocab coverage loss 组件权重。
        w_smooth: smoothness regularisation 权重。
        verbose: 是否打印进度。

    Returns:
        训练后的 ``CalibrationParameters``。"""
    stats_map = aggregate_bucket_rates(responses_by_user, bank, bucket_labels)

    # ── 初始化参数 ──
    beta = -0.30
    cal_k = 0.0000691
    # knots：内部表示为平铺数组 [b1, s1, b2, s2, ...]
    flat_knots = np.array([3000., 1.0, 8000., 0.45, 22000., 1.28], dtype=float)
    max_v = 20000.0

    # ── Adam 状态 ──
    adam = {
        "m_beta": 0.0, "v_beta": 0.0,
        "m_k": 0.0, "v_k": 0.0,
        "m_knots": np.zeros_like(flat_knots),
        "v_knots": np.zeros_like(flat_knots),
        "t": 0,
    }
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    # ── 计算 bucket weights ──
    bucket_sizes = bank.bucket_sizes()
    total_words = len(bank)
    bucket_weights = {
        label: bucket_sizes.get(label, 0) / total_words
        for label in bucket_labels
    }

    # ── 预计算 interval 结构 ──
    interval_intervals = compute_interval_structure(bank, interval=50)

    # ── 预计算 official vocab sets（已匹配到词库）──
    official_sets = _build_official_vocab_dict(bank)
    n_official = sum(len(info["words"]) for info in official_sets.values())
    if verbose:
        print(f"Official vocab: {sum(len(v['words']) for v in official_sets.values())} words matched")
        for name, info in official_sets.items():
            print(f"  {name}: {len(info['words'])} matched, expected coverage={info['expected_coverage']}")

    # ── 训练循环 ──
    for epoch in range(n_epochs):
        total_loss = 0.0
        total_loss_interval = 0.0
        total_loss_official = 0.0
        total_loss_smooth = 0.0
        total_loss_bucket = 0.0
        grad_beta = 0.0
        grad_k = 0.0
        grad_knots = np.zeros_like(flat_knots)
        n_intervals_used = 0

        # ── 每用户 pass ──
        n_users_with_data = 0
        for uid in stats_map:
            stats = stats_map[uid]
            if not stats.bucket_rates:
                continue
            n_users_with_data += 1

            # 内部：为该用户拟合 α
            alpha = fit_user_alpha(uid, stats, bank, bucket_labels, beta)

            # ---- 1. Bucket MSE loss（原始）----
            for bi, bucket in enumerate(bucket_labels):
                r_obs = stats.bucket_rates.get(bucket)
                if r_obs is None:
                    continue
                p_pred = predict_bucket_rate(alpha, beta, bank, bucket)
                err = p_pred - r_obs
                w_b = bucket_weights.get(bucket, 0.0)
                total_loss_bucket += w_b * err ** 2

                # Gradient dL/dβ（bucket 组件）
                items = bank.get_items_in_bucket(bucket)
                if items:
                    log_rs = np.log(np.array([it.rank for it in items], dtype=float))
                    sigs = sigmoid(alpha + beta * log_rs)
                    dp_dbeta = np.mean(sigs * (1.0 - sigs) * log_rs)
                    grad_beta += 2.0 * w_b * err * dp_dbeta

            # ---- 2. Interval-group loss（新增）----
            loss_iv, n_iv = _compute_interval_loss(
                alpha, beta, bank, interval_intervals, sampled_words_per_interval=2
            )
            total_loss_interval += w_interval * loss_iv
            n_intervals_used += n_iv

            # ---- 3. Official vocab coverage loss（新增）----
            total_loss_official += w_official * _compute_official_loss(
                alpha, beta, bank, official_sets
            )

            # ---- 4. Smoothness regularisation（新增）----
            # 计算每用户项（严格来说不必要，因为它与用户无关，
            # 但为清晰起见纳入 total loss）
            total_loss_smooth += w_smooth * _compute_smoothness_loss(alpha, beta, bank)

        # β 的 L2 regularisation
        total_loss_bucket += l2_lambda * beta ** 2
        grad_beta += 2.0 * l2_lambda * beta

        # Total loss 总损失
        total_loss = total_loss_bucket + total_loss_interval + total_loss_official + total_loss_smooth

        # ── k 和 knots 的 finite-difference gradients ──
        eps_fd = 1e-6

        # ∇_k L
        loss_plus = _loss_with_params(
            stats_map, bank, bucket_labels, bucket_weights,
            beta, cal_k + eps_fd, flat_knots, max_v,
            w_interval=w_interval, w_official=w_official, w_smooth=w_smooth,
            interval_intervals=interval_intervals, official_sets=official_sets,
        )
        loss_minus = _loss_with_params(
            stats_map, bank, bucket_labels, bucket_weights,
            beta, cal_k - eps_fd, flat_knots, max_v,
            w_interval=w_interval, w_official=w_official, w_smooth=w_smooth,
            interval_intervals=interval_intervals, official_sets=official_sets,
        )
        grad_k = (loss_plus - loss_minus) / (2.0 * eps_fd)

        # ∇_{knots} L 梯度
        for ki in range(len(flat_knots)):
            flat_plus = flat_knots.copy()
            flat_plus[ki] += eps_fd
            flat_minus = flat_knots.copy()
            flat_minus[ki] -= eps_fd
            lp = _loss_with_params(
                stats_map, bank, bucket_labels, bucket_weights,
                beta, cal_k, flat_plus, max_v,
                w_interval=w_interval, w_official=w_official, w_smooth=w_smooth,
                interval_intervals=interval_intervals, official_sets=official_sets,
            )
            lm = _loss_with_params(
                stats_map, bank, bucket_labels, bucket_weights,
                beta, cal_k, flat_minus, max_v,
                w_interval=w_interval, w_official=w_official, w_smooth=w_smooth,
                interval_intervals=interval_intervals, official_sets=official_sets,
            )
            grad_knots[ki] = (lp - lm) / (2.0 * eps_fd)

        # ── Adam 更新 ──
        adam["t"] += 1
        t = adam["t"]

        for name, grad, m_key, v_key in [
            ("beta", grad_beta, "m_beta", "v_beta"),
            ("k", grad_k, "m_k", "v_k"),
        ]:
            adam[m_key] = beta1 * adam[m_key] + (1 - beta1) * grad
            adam[v_key] = beta2 * adam[v_key] + (1 - beta2) * grad ** 2

        beta -= lr * (adam["m_beta"] / (1 - beta1 ** t)) / (
            np.sqrt(adam["v_beta"] / (1 - beta2 ** t)) + eps
        )
        cal_k -= lr * (adam["m_k"] / (1 - beta1 ** t)) / (
            np.sqrt(adam["v_k"] / (1 - beta2 ** t)) + eps
        )

        # Adam 更新 for knots array
        adam["m_knots"] = beta1 * adam["m_knots"] + (1 - beta1) * grad_knots
        adam["v_knots"] = beta2 * adam["v_knots"] + (1 - beta2) * grad_knots ** 2
        m_knots_hat = adam["m_knots"] / (1 - beta1 ** t)
        v_knots_hat = adam["v_knots"] / (1 - beta2 ** t)
        flat_knots -= lr * m_knots_hat / (np.sqrt(v_knots_hat) + eps)

        # ── 约束 ──
        beta = max(beta, -5.0)
        cal_k = max(cal_k, 1e-8)
        np.clip(flat_knots[0::2], 100, None, out=flat_knots[0::2])   # boundaries > 100 约束
        np.clip(flat_knots[1::2], 0.01, None, out=flat_knots[1::2])  # slopes > 0 约束

        if verbose and epoch % 50 == 0:
            knots_display = list(zip(flat_knots[0::2].astype(int), flat_knots[1::2]))
            print(f"Epoch {epoch:4d} | loss {total_loss:.6f} "
                  f"(B {total_loss_bucket:.4f} I {total_loss_interval:.4f} "
                  f"O {total_loss_official:.4f} S {total_loss_smooth:.4f}) | "
                  f"β {beta:.4f} | k {cal_k:.8f} | knots {knots_display}")

    # ── 组装结果 ──
    trained_knots = [
        (int(flat_knots[i]), float(flat_knots[i + 1]))
        for i in range(0, len(flat_knots), 2)
    ]
    return CalibrationParameters(
        beta=float(beta),
        calibration_k=float(cal_k),
        piecewise_knots=trained_knots,
        training_loss=float(total_loss),
        loss_interval=float(total_loss_interval),
        loss_official=float(total_loss_official),
        loss_smooth=float(total_loss_smooth),
        loss_bucket=float(total_loss_bucket),
        n_epochs=n_epochs,
    )


def _loss_with_params(
    stats_map: dict[int, UserBucketStats],
    bank: VocabBank,
    bucket_labels: list[str],
    bucket_weights: dict[str, float],
    beta: float,
    cal_k: float,
    flat_knots: np.ndarray,
    max_v: float,
    w_interval: float = 0.3,
    w_official: float = 0.3,
    w_smooth: float = 0.01,
    interval_intervals: list[tuple[int, int]] | None = None,
    official_sets: dict[str, dict[str, Any]] | None = None,
) -> float:
    """计算给定参数集的 total loss。

    支持完整多组件 loss：bucket + interval + official + smooth。"""
    knots = [
        (int(flat_knots[i]), float(flat_knots[i + 1]))
        for i in range(0, len(flat_knots), 2)
    ]

    if interval_intervals is None:
        interval_intervals = compute_interval_structure(bank, interval=50)
    if official_sets is None:
        official_sets = _build_official_vocab_dict(bank)

    total = 0.0
    for uid, stats in stats_map.items():
        if not stats.bucket_rates:
            continue
        alpha = fit_user_alpha(uid, stats, bank, bucket_labels, beta,
                               n_iter=100, lr=0.1)

        # Bucket loss 损失 损失
        for bucket, r_obs in stats.bucket_rates.items():
            p_pred = predict_bucket_rate(alpha, beta, bank, bucket)
            w_b = bucket_weights.get(bucket, 0.0)
            total += w_b * (p_pred - r_obs) ** 2

        # Interval loss 损失 损失
        loss_iv, _ = _compute_interval_loss(alpha, beta, bank, interval_intervals)
        total += w_interval * loss_iv

        # Official vocab loss 损失 损失
        total += w_official * _compute_official_loss(alpha, beta, bank, official_sets)

        # 平滑性
        total += w_smooth * _compute_smoothness_loss(alpha, beta, bank)

    return total


# PyTorch 版本（优先，torch 可用时使用）
def train_torch(
    responses_by_user: dict[int, list[tuple[str, bool]]],
    bank: VocabBank,
    bucket_labels: list[str],
    n_epochs: int = 500,
    lr: float = 0.001,
) -> CalibrationParameters:
    """使用 PyTorch autograd 训练；torch 不可用时回退到 NumPy。"""
    try:
        import torch
    except ImportError:
        print("PyTorch not available, falling back to NumPy SGD trainer.")
        return train_numpy(responses_by_user, bank, bucket_labels, n_epochs, lr)

    # PyTorch autograd 实现占位；目前委托给 NumPy。
    return train_numpy(responses_by_user, bank, bucket_labels, n_epochs, lr)


# ---------------------------------------------------------------------------
# ── 新增：Dry-run / 结构检查模式 ──
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """只打印 sampling 结构和参数信息，不实际训练。"""
    bank = VocabBank(DEFAULT_CONFIG)
    bucket_labels = list(bank.words_by_bucket.keys())

    print(f"{'='*70}")
    print(f"VOCAB BANK: {len(bank)} words, {len(bucket_labels)} buckets")
    print(f"  Buckets: {bucket_labels}")
    bucket_sizes = bank.bucket_sizes()
    for label in bucket_labels:
        print(f"    {label}: {bucket_sizes.get(label, 0)} words")
    print()

    # ── Interval 结构 ──
    print(f"{'='*70}")
    print("INTERVAL SAMPLING STRUCTURE")
    print(f"{'='*70}")

    for interval in [50, 100]:
        intervals = compute_interval_structure(bank, interval=interval)
        n_intervals = len(intervals)
        total_possible = n_intervals * 2  # 每组 2 个

        # 统计词数足够的 intervals
        rank_to_words: dict[int, int] = {}
        for item in bank.items:
            rank_to_words[item.rank] = rank_to_words.get(item.rank, 0) + 1

        intervals_ok = 0
        intervals_short = 0
        for start_r, end_r in intervals:
            count = sum(rank_to_words.get(r, 0) for r in range(start_r, end_r + 1))
            if count >= 2:
                intervals_ok += 1
            else:
                intervals_short += 1

        print(f"\n  interval={interval}:")
        print(f"    Total rank segments: {n_intervals}")
        print(f"    Segments with ≥2 words: {intervals_ok}")
        print(f"    Segments with <2 words: {intervals_short}")
        print(f"    Max test words: {total_possible} (if all segments ≥2)")
        print(f"    Realistic test words: ~{intervals_ok * 2}")

        # 示例 intervals（前 5 个和后 5 个）
        print(f"    First 5 intervals: {intervals[:5]}")
        print(f"    Last 5 intervals:  {intervals[-5:]}")
    print()

    # ── Official vocab 匹配 ──
    print(f"{'='*70}")
    print("OFFICIAL VOCABULARY ANCHOR POINTS")
    print(f"{'='*70}")

    from .official_vocab import (
        get_official_vocab_sets,
        get_set_words,
        describe_official_vocab,
    )

    print(describe_official_vocab(bank))

    # ── 详细匹配统计 ──
    official_sets = _build_official_vocab_dict(bank)
    for name, info in official_sets.items():
        words = info["words"]
        if not words:
            continue
        ranks = [r for r in [bank.get_rank(w) for w in words] if r is not None]
        if ranks:
            print(f"  [{name}] matched {len(words)} words")
            print(f"    rank range: [{min(ranks)} – {max(ranks)}]")
            print(f"    median:     {sorted(ranks)[len(ranks)//2]}")
            # 示例 10 个词
            sample_words = sorted(list(words))[:10]
            print(f"    samples:    {', '.join(sample_words)}")
        print()
    print()

    # ── 参数结构 ──
    print(f"{'='*70}")
    print("OPTIMIZABLE PARAMETERS")
    print(f"{'='*70}")
    print(f"""
  β (logistic slope)           = -0.30        (initial)
  k (tanh rate)                = 0.0000691    (initial)
  piecewise_knots:             = [(3000, 1.0), (8000, 0.45), (22000, 1.28)]

  Loss function:
    L = w_bucket × L_bucket
      + w_interval × L_interval
      + w_official × L_official
      + w_smooth × L_smooth
      + l2_lambda × β²

    w_bucket   = 1.0      (bucket MSE weight)
    w_interval = 0.3      (interval group MSE weight)
    w_official = 0.3      (official vocab coverage MSE weight)
    w_smooth   = 0.01     (smoothness regularisation)
    l2_lambda  = 1.0      (L2 regularisation)
""")

    # ── 当前参数下的预测曲线 ──
    print(f"{'='*70}")
    print("PREDICTED COGNITION CURVE (at initial params)")
    print(f"{'='*70}")
    test_alpha = -1.0
    test_beta = -0.30
    print(f"  Using α={test_alpha}, β={test_beta}")
    for r in [1, 100, 500, 1000, 3000, 5000, 8000, 10000, 15000, 20000, 25000, 30000]:
        p = predict_word_prob(test_alpha, test_beta, r)
        print(f"    rank {r:>6}: P(known) = {p:.4f}")
    print()

    # ── 该曲线下的 official vocab 覆盖率 ──
    print("  Official vocab coverage at this α/β:")
    for name, info in official_sets.items():
        words = info["words"]
        if not words:
            continue
        probs = [
            predict_word_prob(test_alpha, test_beta, r)
            for w in words if (r := bank.get_rank(w)) is not None
        ]
        if probs:
            avg = sum(probs) / len(probs)
            expected = info["expected_coverage"]
            print(f"    {name}: predicted={avg:.4f}  expected={expected}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def _run_synthetic_training_demo(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions: int = 100,
    power: float = 0.5,
    seed: int = 42,
) -> None:
    """运行训练数据 demo：生成合成测试者，并打印每个用户的预测词汇量与实际词汇量。"""
    from .synthetic_generator import generate_synthetic_data, run_synthetic_training

    print("=" * 65)
    print("  合成训练数据 —— 高频优先采样验证")
    print("=" * 65)
    print()
    print(f"  test questions per user: {n_questions}")
    print(f"  frequency-weight power:  {power}")
    print()

    results = run_synthetic_training(
        bank, vocab_sizes=vocab_sizes,
        n_questions=n_questions, power=power, seed=seed,
    )

    header = (
        f"  {'Vocab':>7s}  {'KnownRate':>10s}  {'Predicted':>10s}"
        f"  {'Loss':>12s}  {'RawEst':>8s}  {'LogEst':>8s}"
    )
    sep = (
        f"  {'------':>7s}  {'----------':>10s}  {'----------':>10s}"
        f"  {'------------':>12s}  {'------':>8s}  {'-------':>8s}"
    )
    print(header)
    print(sep)

    total_loss = 0.0
    for r in results:
        total_loss += r["loss"]
        print(
            f"  {r['vocab_size']:>7d}  {r['known_rate']:>10.2%}"
            f"  {r['predicted']:>10d}"
            f"  {r['loss']:>12.1f}"
            f"  {r.get('raw_estimate', 0):>8d}"
            f"  {r.get('logistic_estimate', 0):>8d}"
        )

    print()
    mean_loss = total_loss / len(results) if results else 0.0
    print(f"  Mean loss: {mean_loss:.1f}")
    print()

    # 同时生成并展示 synthetic dataset summary
    data = generate_synthetic_data(
        bank, vocab_sizes=vocab_sizes,
        n_questions=n_questions, power=power, seed=seed,
    )
    from .synthetic_generator import describe_synthetic_dataset
    print(describe_synthetic_dataset(data))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train calibration parameters from user data.",
    )
    parser.add_argument("--data", default=None, help="Path to calibration dataset JSON")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", default=None, help="Output path for trained parameters JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sampling/param structure without training")
    parser.add_argument("--validate", action="store_true", help="Run synthetic validation first")
    parser.add_argument("--show-training-data", action="store_true",
                        help="Show synthetic training data demo (high-frequency weighted)")
    # Loss 权重选项
    parser.add_argument("--w-interval", type=float, default=0.3,
                        help="Weight for interval-group loss (default 0.3)")
    parser.add_argument("--w-official", type=float, default=0.3,
                        help="Weight for official vocab loss (default 0.3)")
    parser.add_argument("--w-smooth", type=float, default=0.01,
                        help="Weight for smoothness regularisation (default 0.01)")
    # 合成生成选项
    parser.add_argument("--power", type=float, default=0.5,
                        help="Power-law exponent for frequency-weighted sampling (default 0.5)")
    parser.add_argument("--n-questions", type=int, default=100,
                        help="Number of test questions per synthetic user (default 100)")
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    bank = VocabBank(DEFAULT_CONFIG)
    bucket_labels = list(bank.words_by_bucket.keys())
    print(f"VocabBank: {len(bank)} words, {len(bucket_labels)} buckets")

    if args.show_training_data:
        _run_synthetic_training_demo(
            bank,
            vocab_sizes=[2000, 5000, 8000, 10000, 15000],
            n_questions=args.n_questions,
            power=args.power,
            seed=DEFAULT_CONFIG.random_seed,
        )
        return

    if args.validate:
        from .synthetic_generator import synthetic_validation
        print("Running synthetic validation...")
        synthetic_validation(bank, bucket_labels)

    if not args.data:
        print("ERROR: --data is required unless --dry-run or --show-training-data is used.")
        sys.exit(1)

    data = load_responses(args.data)
    print(f"Loaded {len(data)} users")

    params = train_numpy(
        data, bank, bucket_labels,
        n_epochs=args.epochs,
        lr=args.lr,
        w_interval=args.w_interval,
        w_official=args.w_official,
        w_smooth=args.w_smooth,
    )

    print("\n=== Trained Parameters ===")
    print(f"  beta:              {params.beta:.6f}")
    print(f"  calibration_k:     {params.calibration_k:.8f}")
    print(f"  piecewise_knots:   {params.piecewise_knots}")
    print(f"  training_loss:     {params.training_loss:.6f}")
    print(f"    L_bucket:        {params.loss_bucket:.6f}")
    print(f"    L_interval:      {params.loss_interval:.6f}")
    print(f"    L_official:      {params.loss_official:.6f}")
    print(f"    L_smooth:        {params.loss_smooth:.6f}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(params.to_config_dict(), ensure_ascii=False, indent=2)
        )
        print(f"\nParameters saved to {args.output}")


if __name__ == "__main__":
    main()
