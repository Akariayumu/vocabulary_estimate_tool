"""Core vocabulary-size prediction model."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .config import DEFAULT_CONFIG, EstimatorConfig, bucket_beta_prior, bucket_label
from .vocab_bank import VocabBank


Response = tuple[str, bool]


@dataclass(frozen=True)
class PreparedResponse:
    """A learner response enriched with rank and bucket metadata."""

    word: str
    known: bool
    rank: int
    bucket: str


class VocabEstimator:
    """Estimate vocabulary size from stratified word-test responses."""

    def __init__(
        self,
        vocab_bank: VocabBank,
        config: EstimatorConfig = DEFAULT_CONFIG,
        seed: int | None = None,
    ) -> None:
        self.vocab_bank = vocab_bank
        self.config = config
        self.rng = random.Random(config.random_seed if seed is None else seed)

    def prepare_responses(self, responses: Iterable[Response]) -> list[PreparedResponse]:
        """Attach frequency rank and bucket to raw ``(word, known)`` records."""

        prepared: list[PreparedResponse] = []
        for word, known in responses:
            rank = self.vocab_bank.get_rank(word)
            if rank is None:
                continue
            bucket = self.vocab_bank.bucket_for_rank(rank)
            if bucket is None:
                continue
            prepared.append(PreparedResponse(word=word, known=bool(known), rank=rank, bucket=bucket))
        return prepared

    def baseline_estimate(self, responses: Iterable[Response] | list[PreparedResponse]) -> dict:
        """Estimate V = sum(N_b * p_b), bucket size times known-rate."""

        prepared = self._ensure_prepared(responses)
        bucket_known: dict[str, list[bool]] = {bucket: [] for bucket in self.vocab_bank.words_by_bucket}
        for row in prepared:
            bucket_known.setdefault(row.bucket, []).append(row.known)

        bucket_labels = list(self.vocab_bank.words_by_bucket)
        n_buckets = len(bucket_labels)

        # Beta-smoothed rates per bucket
        observed_rates: dict[str, float] = {}
        for bucket, values in bucket_known.items():
            if values:
                idx = bucket_labels.index(bucket)
                alpha, beta = bucket_beta_prior(idx, n_buckets, self.config)
                known = sum(values)
                total = len(values)
                smoothed_rate = (known + alpha) / (total + alpha + beta)
                observed_rates[bucket] = smoothed_rate

        global_rate = (
            sum(row.known for row in prepared) / len(prepared)
            if prepared
            else 0.0
        )

        contributions = {}
        total = 0.0
        for bucket, items in self.vocab_bank.words_by_bucket.items():
            rate = observed_rates.get(bucket)
            if rate is None:
                rate = self._nearest_observed_rate(bucket, observed_rates, global_rate)
            contribution = len(items) * rate
            contributions[bucket] = {
                "bucket_size": len(items),
                "known_rate": rate,
                "estimated_known": contribution,
                "observed_count": len(bucket_known.get(bucket, [])),
                "raw_known_count": sum(bucket_known.get(bucket, [])),
                "raw_sample_total": len(bucket_known.get(bucket, [])),
            }
            total += contribution

        return {"estimate": total, "bucket_contributions": contributions}

    def logistic_estimate(self, responses: Iterable[Response] | list[PreparedResponse]) -> dict:
        """Fit P(know)=sigmoid(alpha + beta*log(rank)) and sum over the bank."""

        prepared = self._ensure_prepared(responses)
        if len(prepared) < 4:
            baseline = self.baseline_estimate(prepared)
            return {
                "estimate": baseline["estimate"],
                "alpha": None,
                "beta": None,
                "method": "baseline_fallback_too_few_samples",
            }

        ranks = np.array([row.rank for row in prepared], dtype=float)
        y = np.array([1.0 if row.known else 0.0 for row in prepared], dtype=float)

        if np.all(y == y[0]):
            p = 0.98 if y[0] == 1.0 else 0.02
            return {
                "estimate": float(len(self.vocab_bank) * p),
                "alpha": math.log(p / (1.0 - p)),
                "beta": 0.0,
                "method": "constant_degenerate",
            }

        if self.config.enable_weighted_fitting:
            alpha, beta = self._fit_weighted_logistic(np.log(ranks), ranks, y)
            weight_method = "weighted"
        else:
            alpha, beta = self._fit_logistic(np.log(ranks), y)
            weight_method = "standard"
        bank_ranks = np.array(self.vocab_bank.ranks(), dtype=float)
        probabilities = self._sigmoid(alpha + beta * np.log(bank_ranks))
        return {
            "estimate": float(np.sum(probabilities)),
            "alpha": float(alpha),
            "beta": float(beta),
            "method": "logistic_log_rank",
            "weight_method": weight_method,
        }

    def estimate_single(self, responses: Iterable[Response]) -> dict:
        """Return point estimate, interval, level and confidence for one group."""

        response_rows = list(responses)
        prepared = self.prepare_responses(response_rows)
        baseline = self.baseline_estimate(prepared)
        logistic = self.logistic_estimate(prepared)
        point = logistic["estimate"]
        if not math.isfinite(point) or point <= 0:
            point = baseline["estimate"]

        # Apply research-based calibration to compress overestimates
        raw_point = point
        point = self.calibrate(point)

        ci_low, ci_high = self.bootstrap_interval(prepared)
        # Also calibrate the interval bounds
        ci_low, ci_high = self.calibrate(ci_low), self.calibrate(ci_high)
        confidence = self.confidence_label(point, ci_low, ci_high)

        return {
            "vocabulary_range": [round(ci_low), round(ci_high)],
            "point_estimate": round(point),
            "level": self.map_level(point),
            "confidence": confidence,
            "confidence_interval_90": [round(ci_low), round(ci_high)],
            "raw_estimate": round(raw_point),
            "baseline_estimate": round(baseline["estimate"]),
            "logistic_estimate": round(logistic["estimate"]),
            "logistic_method": logistic["method"],
            "sample_size": len(prepared),
            "ignored_responses": max(0, len(response_rows) - len(prepared)),
            "bucket_contributions": baseline["bucket_contributions"],
        }

    def estimate_groups(self, grouped_responses: Mapping[str, Iterable[Response]]) -> dict:
        """Estimate all learner classes and enforce the C > F > P > K order."""

        raw_results = {
            class_name: self.estimate_single(list(responses))
            for class_name, responses in grouped_responses.items()
        }

        ordered = [name for name in self.config.ordered_classes if name in raw_results]
        estimates = [raw_results[name]["point_estimate"] for name in ordered]
        adjusted = self.enforce_order(estimates)
        consistent = all(estimates[i] >= estimates[i + 1] for i in range(len(estimates) - 1))

        for name, value in zip(ordered, adjusted):
            raw_results[name]["order_adjusted_estimate"] = round(value)
            raw_results[name]["order_adjusted_level"] = self.map_level(value)

        return {
            "classes": raw_results,
            "ordering_consistency": {
                "expected_order": ">".join(self.config.ordered_classes),
                "checked_classes": ordered,
                "was_consistent": consistent,
                "original_estimates": dict(zip(ordered, estimates)),
                "isotonic_estimates": dict(zip(ordered, [round(v) for v in adjusted])),
            },
        }

    def bootstrap_interval(self, prepared: list[PreparedResponse]) -> tuple[float, float]:
        """Return a bootstrap confidence interval for the point estimate."""

        if not prepared:
            return 0.0, 0.0

        estimates = []
        n = len(prepared)
        for _ in range(self.config.bootstrap_iterations):
            sample = [prepared[self.rng.randrange(n)] for _ in range(n)]
            estimate = self.logistic_estimate(sample)["estimate"]
            if math.isfinite(estimate):
                estimates.append(estimate)

        if not estimates:
            point = self.baseline_estimate(prepared)["estimate"]
            return point, point

        alpha = (1.0 - self.config.confidence_interval) / 2.0
        low = float(np.quantile(estimates, alpha))
        high = float(np.quantile(estimates, 1.0 - alpha))
        return max(0.0, low), min(float(len(self.vocab_bank)), high)

    def confidence_label(self, point: float, ci_low: float, ci_high: float) -> str:
        """Map CI width / estimate to 高, 中 or 低."""

        if point <= 0:
            return "低"
        ratio = (ci_high - ci_low) / point
        if ratio < self.config.confidence_high_ratio:
            return "高"
        if ratio < self.config.confidence_mid_ratio:
            return "中"
        return "低"

    def calibrate(self, estimate: float) -> float:
        """Apply compression to avoid overestimates.

        Two-stage pipeline:
          1. tanh-based smooth saturation (existing) — anchors around
             the native-speaker ceiling (~20 kf).
          2. Optional piecewise linear compression (方案 B) — applies
             progressively stronger compression in higher bands.

        Published vocabulary-size data: educated native speakers command
        ~20,000 word families (Nation 2006; Goulden et al. 1990).
        Chinese EFL learners top out around 10,000-13,000 (TEM-8 level).
        """

        if estimate <= 0:
            return estimate

        cfg = self.config
        # ---- Stage 1: tanh smooth saturation ----
        max_v = float(cfg.calibration_native_max)
        k = cfg.calibration_k
        calibrated = max_v * math.tanh(k * estimate)
        # For estimates above ~6,000 this compresses progressively.
        # For estimates below ~4,000 the curve is near-linear.

        # ---- Stage 2: piecewise linear compression ----
        if cfg.enable_piecewise_calibration:
            calibrated = self.piecewise_calibrate(calibrated)

        return float(calibrated)

    def piecewise_calibrate(self, estimate: float) -> float:
        """Apply piecewise linear compression (方案 B).

        Segments come from config.piecewise_knots.
        Current: [0,3.5k] slope 1.00, [3.5k,6.5k] slope 0.55,
        [6.5k,12k] slope 0.30, [12k,22k] slope 0.15, >22k slope 0.15.

        Designed so that a user with raw~11.4k → final~6.3k.
        """
        if estimate <= 0:
            return estimate

        knots = self.config.piecewise_knots  # sorted (upper_boundary, slope)
        prev_boundary = 0.0
        prev_value = 0.0

        for boundary, slope in knots:
            if estimate <= boundary:
                return float(prev_value + (estimate - prev_boundary) * slope)
            # Accumulate through this segment
            prev_value += (float(boundary) - prev_boundary) * slope
            prev_boundary = float(boundary)

        # Beyond the last knot — continue with the final segment's slope
        return float(prev_value + (estimate - prev_boundary) * knots[-1][1])

    def map_level(self, estimate: float) -> str:
        """Map a vocabulary-size estimate to a Chinese learner level."""

        thresholds = []
        for name, low, high in self.config.levels:
            thresholds.append((name, low, high))

        for idx, (name, low, high) in enumerate(thresholds):
            if high is None:
                if estimate >= low:
                    if abs(estimate - low) <= self.config.transition_margin and idx > 0:
                        return f"{thresholds[idx - 1][0]}/{name}过渡"
                    return name
                continue

            if abs(estimate - high) <= self.config.transition_margin and idx + 1 < len(thresholds):
                return f"{name}/{thresholds[idx + 1][0]}过渡"
            if low <= estimate < high:
                return name

        if estimate < thresholds[0][1]:
            return "初中以下"
        return thresholds[-1][0]

    def enforce_order(self, estimates: list[float]) -> list[float]:
        """Apply isotonic regression to enforce C ≥ F ≥ P ≥ K.

        Uses PAVA (pool-adjacent-violators algorithm) when sklearn is
        available; falls back to a simple greedy fix.
        """

        if len(estimates) <= 1:
            return list(estimates)
        try:
            from sklearn.isotonic import IsotonicRegression

            x = np.arange(len(estimates))
            model = IsotonicRegression(increasing=False, out_of_bounds="clip")
            return [float(v) for v in model.fit_transform(x, np.array(estimates, dtype=float))]
        except Exception:
            return self._pava_decreasing(estimates)

    def _fit_logistic(self, log_ranks: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Fit logistic regression using sklearn when available, else NumPy GD."""

        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1.0 / self.config.logistic_l2, solver="lbfgs")
            model.fit(log_ranks.reshape(-1, 1), y.astype(int))
            return float(model.intercept_[0]), float(model.coef_[0][0])
        except Exception:
            return self._fit_logistic_numpy(log_ranks, y)

    def _fit_logistic_numpy(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mean = float(np.mean(x))
        std = float(np.std(x) or 1.0)
        z = (x - mean) / std
        design = np.column_stack([np.ones_like(z), z])
        weights = np.zeros(2, dtype=float)

        for _ in range(self.config.logistic_max_iter):
            logits = design @ weights
            p = self._sigmoid(logits)
            grad = design.T @ (p - y) / len(y)
            grad[1] += self.config.logistic_l2 * weights[1] / len(y)
            weights -= self.config.logistic_lr * grad

        beta = weights[1] / std
        alpha = weights[0] - beta * mean
        return float(alpha), float(beta)

    def _fit_weighted_logistic(
        self,
        log_ranks: np.ndarray,
        ranks: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float]:
        """Fit a *rank-weighted* logistic regression.

        Samples are weighted by ``_compute_weight(rank)`` so that errors on
        high-frequency (low-rank) words are penalised more heavily than errors
        on rare words.  This yields a more conservative vocabulary-size
        estimate than the unweighted fit.

        Attempts sklearn with ``sample_weight`` first; falls back to a
        weighted NumPy gradient descent.
        """
        sample_weights = np.array(
            [self._compute_weight(float(r)) for r in ranks],
            dtype=float,
        )
        # ---- try sklearn ----
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(C=1.0 / self.config.logistic_l2, solver="lbfgs")
            model.fit(
                log_ranks.reshape(-1, 1),
                y.astype(int),
                sample_weight=sample_weights,
            )
            return float(model.intercept_[0]), float(model.coef_[0][0])
        except Exception:
            pass

        # ---- weighted NumPy GD ----
        mean = float(np.mean(log_ranks))
        std = float(np.std(log_ranks) or 1.0)
        z = (log_ranks - mean) / std
        design = np.column_stack([np.ones_like(z), z])
        weights = np.zeros(2, dtype=float)

        for _ in range(self.config.logistic_max_iter):
            logits = design @ weights
            p = self._sigmoid(logits)
            # Weighted gradient: each sample contributes w_i * (p_i - y_i)
            grad = design.T @ (sample_weights * (p - y)) / len(y)
            grad[1] += self.config.logistic_l2 * weights[1] / len(y)
            weights -= self.config.logistic_lr * grad

        beta = weights[1] / std
        alpha = weights[0] - beta * mean
        return float(alpha), float(beta)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _compute_weight(rank: float) -> float:
        """Compute per-sample weight based on frequency rank.

        High-frequency (low-rank) words that are answered incorrectly get full
        weight (strongest penalty).  Low-frequency (high-rank) words get
        progressively less weight, so the fit focuses on common vocabulary.

        Formula:  w = 1 / (1 + log2(max(rank, 10) / 10))

        Expected values (clamped):
            rank=1   → w=1.000   (truncated to min rank=10)
            rank=5000  → w≈0.100
            rank=20000 → w≈0.083
        """
        effective_rank = max(rank, 10.0)
        return float(1.0 / (1.0 + math.log2(effective_rank / 10.0)))

    @staticmethod
    def _pava_decreasing(values: list[float]) -> list[float]:
        """Pool-adjacent-violators algorithm for non-increasing sequences."""

        blocks = [{"sum": float(v), "weight": 1.0, "start": i, "end": i} for i, v in enumerate(values)]
        i = 0
        while i < len(blocks) - 1:
            left = blocks[i]["sum"] / blocks[i]["weight"]
            right = blocks[i + 1]["sum"] / blocks[i + 1]["weight"]
            if left >= right:
                i += 1
                continue
            blocks[i]["sum"] += blocks[i + 1]["sum"]
            blocks[i]["weight"] += blocks[i + 1]["weight"]
            blocks[i]["end"] = blocks[i + 1]["end"]
            del blocks[i + 1]
            if i > 0:
                i -= 1

        output = [0.0] * len(values)
        for block in blocks:
            mean = block["sum"] / block["weight"]
            for idx in range(int(block["start"]), int(block["end"]) + 1):
                output[idx] = mean
        return output

    def _nearest_observed_rate(
        self,
        bucket: str,
        observed_rates: dict[str, float],
        global_rate: float,
    ) -> float:
        if not observed_rates:
            # No observations at all → use the Bayesian prior for this bucket
            labels = list(self.vocab_bank.words_by_bucket)
            idx = labels.index(bucket)
            alpha, beta = bucket_beta_prior(idx, len(labels), self.config)
            return alpha / (alpha + beta)
        labels = list(self.vocab_bank.words_by_bucket)
        idx = labels.index(bucket)
        return min(
            observed_rates.items(),
            key=lambda item: abs(labels.index(item[0]) - idx),
        )[1]

    def _ensure_prepared(
        self,
        responses: Iterable[Response] | list[PreparedResponse],
    ) -> list[PreparedResponse]:
        rows = list(responses)
        if not rows:
            return []
        if isinstance(rows[0], PreparedResponse):
            return rows  # type: ignore[return-value]
        return self.prepare_responses(rows)  # type: ignore[arg-type]
