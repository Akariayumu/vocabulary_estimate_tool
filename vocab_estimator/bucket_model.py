#!/usr/bin/env python3
"""Bucket matrix vocabulary estimator.

Trained on V100 (192.168.100.104) with self-consistent synthetic data.
Each bucket gets its own θ parameter, combined with a per-user γ offset.

Model:
  P(known|bucket_b) = sigmoid(θ_b + γ_u)
  Vocab = Σ_bucket [ bucket_size_b × sigmoid(θ_b + γ_u) ]

Params from trained_params_bucket.json.
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
    """Estimate vocabulary using bucket matrix model.

    Args:
        responses: [(word, known), ...]
        bucket_fn: callable(word) -> bucket_name (e.g. '1k', '5k', None)

    Returns:
        dict with point_estimate, level, etc. matching estimate_single format.
    """
    params = _load_params()
    theta = params["theta"]
    bucket_sizes = params["bucket_sizes"]

    # Aggregate responses by bucket
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

    # Bayesian gamma fitting using cross-entropy with Beta(1,1) uniform prior
    # Posterior mean for bucket b: (known + 1) / (n + 2) — Laplace smoothing
    # Cross-entropy loss avoids sigmoid saturation at extreme theta values
    # and naturally pulls estimates toward 0.5 when sample is small

    def _compute_loss(g):
        """Cross-entropy loss for a given gamma."""
        loss = 0.0
        for b, vals in bucket_counts.items():
            n_total = len(vals)
            n_known = sum(vals)
            # Target rate = posterior mean with Beta(1,1) uniform prior
            target = (n_known + 1.0) / (n_total + 2.0)
            target = max(min(target, 1.0 - 1e-15), 1e-15)
            p_model = 1.0 / (1.0 + math.exp(-(theta[b] + g)))
            p_model = max(min(p_model, 1.0 - 1e-15), 1e-15)
            # Cross-entropy: -target*log(p) - (1-target)*log(1-p)
            loss -= target * math.log(p_model) + (1.0 - target) * math.log(1.0 - p_model)
        return loss

    # Grid search γ in [-15, 15], step 0.1
    best_gamma = 0.0
    best_loss = float("inf")

    for g_int in range(-150, 151):
        g = g_int * 0.1
        loss = _compute_loss(g)
        if loss < best_loss:
            best_loss = loss
            best_gamma = g

    # Refine with finer grid around best
    for g_int in range(int(best_gamma * 10) - 3, int(best_gamma * 10) + 4):
        g = g_int / 10.0
        loss = _compute_loss(g)
        if loss < best_loss:
            best_loss = loss
            best_gamma = g

    gamma = best_gamma

    # Compute raw vocabulary estimate
    raw_est = 0.0
    for b in _BUCKET_ORDER:
        sz = bucket_sizes[b]
        p_know = 1.0 / (1.0 + math.exp(-(theta[b] + gamma)))
        raw_est += sz * p_know

    # Calibration is essentially identity (k≈0, knots≈1), but apply for compatibility
    cal = _calibrate_bucket(raw_est, params)

    # Level mapping (use existing vocab_model logic by importing)
    # Simplified level mapping
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
    """Calibration from bucket model params. Near identity."""
    k = params.get("calibration_k", 0.0)
    if k < -1e-8:
        k = 0.0  # clamp negative k (means tanh is unused)
    knots = params.get("piecewise_knots", [])

    # Tanh stage (k near 0, so effectively identity)
    if k > 1e-10:
        cal = 20000.0 * math.tanh(k * estimate)
    else:
        cal = estimate

    # Piecewise stage (knots near 1.0, so effectively identity)
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

    return min(cal, 21000.0)  # ceiling
