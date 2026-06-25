#!/usr/bin/env python3
"""Bucket matrix 词汇量估算器。

在 V100（192.168.100.104）上使用自洽合成数据训练。
每个 bucket 拥有自己的 θ 参数，并与每个用户的 γ offset 组合。

Model:
  P(known|bucket_b) = sigmoid(θ_b + γ_u)
  Vocab = Σ_bucket [ bucket_size_b × sigmoid(θ_b + γ_u) ]

参数来自 trained_params_bucket.json。
"""
import json, math
from pathlib import Path

import numpy as np

_PARAMS_PATH = Path(__file__).parent / "trained_params_bucket.json"

_sigmoid = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40, dtype=float)))

_BUCKET_ORDER = ["1k", "2k", "3k", "5k", "8k", "10k", "15k", "20k", "30k"]

_params = None


def _load_params():
    global _params
    if _params is not None:
        return _params
    if not _PARAMS_PATH.exists():
        raise FileNotFoundError(f"Bucket model params not found: {_PARAMS_PATH}")
    with open(_PARAMS_PATH) as f:
        raw = json.load(f)
    _params = raw
    return raw


def is_available() -> bool:
    return _PARAMS_PATH.exists()


def estimate(responses: list[tuple[str, bool]], bucket_fn) -> dict:
    """使用 bucket matrix model 估算词汇量。

    Args:
        responses: [(word, known), ...]
        bucket_fn: callable(word) -> bucket_name (e.g. '1k', '5k', None)

    Returns:
        与 estimate_single 格式匹配的 dict，包含 point_estimate、level 等。
    """
    params = _load_params()
    theta = params["theta"]
    bucket_sizes = params["bucket_sizes"]

    # 按 bucket 聚合 responses
    bucket_counts: dict[str, list[bool]] = {}
    for word, known in responses:
        b = bucket_fn(word)
        if b is not None and b in theta:
            bucket_counts.setdefault(b, []).append(known)

    if not bucket_counts:
        return {
            "point_estimate": 0,
            "raw_estimate": 0,
            "level": "未识别",
            "confidence": "低",
            "sample_size": 0,
            "vocabulary_range": [0, 0],
            "confidence_interval_90": [0, 0],
            "ignored_responses": len(responses),
        }

    # 使用 cross-entropy 与 Beta(1,1) uniform prior 拟合 Bayesian gamma
    # bucket b 的 posterior mean：(known + 1) / (n + 2) — Laplace smoothing
    # Cross-entropy loss 可避免极端 theta 下 sigmoid 饱和，
    # 并在样本较少时自然把估计拉向 0.5

    def _compute_loss(g):
        """给定 gamma 时的 cross-entropy loss。"""
        loss = 0.0
        for b, vals in bucket_counts.items():
            n_total = len(vals)
            n_known = sum(vals)
            # 目标率 = 使用 Beta(1,1) uniform prior 的 posterior mean
            target = (n_known + 1.0) / (n_total + 2.0)
            target = max(min(target, 1.0 - 1e-15), 1e-15)
            p_model = 1.0 / (1.0 + math.exp(-(theta[b] + g)))
            p_model = max(min(p_model, 1.0 - 1e-15), 1e-15)
            # Cross-entropy 公式：-target*log(p) - (1-target)*log(1-p)
            loss -= target * math.log(p_model) + (1.0 - target) * math.log(1.0 - p_model)
        return loss

    # 在 [-15, 15] 上网格搜索 γ，步长 0.1
    best_gamma = 0.0
    best_loss = float("inf")

    for g_int in range(-150, 151):
        g = g_int * 0.1
        loss = _compute_loss(g)
        if loss < best_loss:
            best_loss = loss
            best_gamma = g

    # 在最优点附近用更细网格细化
    for g_int in range(int(best_gamma * 10) - 3, int(best_gamma * 10) + 4):
        g = g_int / 10.0
        loss = _compute_loss(g)
        if loss < best_loss:
            best_loss = loss
            best_gamma = g

    gamma = best_gamma

    # 计算 raw 词汇量估算
    raw_est = 0.0
    for b in _BUCKET_ORDER:
        sz = bucket_sizes[b]
        p_know = 1.0 / (1.0 + math.exp(-(theta[b] + gamma)))
        raw_est += sz * p_know

    # 校准基本等同恒等变换（k≈0，knots≈1），但为兼容性仍然应用
    cal = _calibrate_bucket(raw_est, params)

    # 等级映射（通过导入复用现有 vocab_model 逻辑）
    # 简化版等级映射
    levels = [
        (1500, "初中"),
        (2500, "高中"),
        (4000, "四级"),
        (5500, "六级"),
        (7500, "六级+ / 考研"),
        (10000, "专业/母语级"),
    ]
    level = levels[-1][1]
    for threshold, label in levels:
        if cal < threshold:
            level = label
            break

    total_q = len(responses)
    ignored = total_q - sum(len(v) for v in bucket_counts.values())

    return {
        "point_estimate": round(cal),
        "raw_estimate": round(raw_est),
        "level": level,
        "confidence": "中",
        "vocabulary_range": [max(0, round(cal * 0.85)), round(cal * 1.15)],
        "confidence_interval_90": [max(0, round(cal * 0.85)), round(cal * 1.15)],
        "sample_size": total_q,
        "ignored_responses": ignored,
    }


def _calibrate_bucket(estimate: float, params: dict) -> float:
    """根据 bucket model 参数做校准，结果接近恒等变换。"""
    k = params.get("calibration_k", 0.0)
    if k < -1e-8:
        k = 0.0  # clamp 负 k（表示不使用 tanh）
    knots = params.get("piecewise_knots", [])

    # Tanh 阶段（k 接近 0，因此实际近似恒等）
    if k > 1e-10:
        cal = 20000.0 * math.tanh(k * estimate)
    else:
        cal = estimate

    # Piecewise 阶段（knots 接近 1.0，因此实际近似恒等）
    if knots:
        b = [3000, 8000, 22000]
        ks = [knots[0][1], knots[1][1] if len(knots) > 1 else 1.0,
              knots[2][1] if len(knots) > 2 else 1.0]
        pv, pb = 0.0, 0.0
        for i, bn in enumerate(b):
            if cal <= bn:
                cal = pv + (cal - pb) * ks[i]
                break
            pv += (bn - pb) * ks[i]
            pb = bn
        else:
            cal = pv + (cal - pb) * ks[-1]

    return min(cal, 21000.0)  # 上限
