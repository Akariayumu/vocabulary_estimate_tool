"""核心词汇量预测模型。"""

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
    """带有 rank 和 bucket 元数据的学习者作答。"""

    word: str
    known: bool
    rank: int
    bucket: str


class VocabEstimator:
    """根据分层词汇测试 responses 估算词汇量。"""

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
        """为原始 ``(word, known)`` 记录附加频率 rank 和 bucket。"""

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
        """估算 V = sum(N_b * p_b)，即 bucket 大小乘以已知率。"""

        prepared = self._ensure_prepared(responses)
        bucket_known: dict[str, list[bool]] = {bucket: [] for bucket in self.vocab_bank.words_by_bucket}
        for row in prepared:
            bucket_known.setdefault(row.bucket, []).append(row.known)

        bucket_labels = list(self.vocab_bank.words_by_bucket)
        n_buckets = len(bucket_labels)

        # 每个 bucket 的 Beta-smoothed 已知率
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
        """拟合 P(know)=sigmoid(alpha + beta*log(rank))，并在词库上求和。"""

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
        """返回单个组的点估计、区间、等级和置信度。"""

        response_rows = list(responses)
        prepared = self.prepare_responses(response_rows)
        baseline = self.baseline_estimate(prepared)
        logistic = self.logistic_estimate(prepared)
        point = logistic["estimate"]
        if not math.isfinite(point) or point <= 0:
            point = baseline["estimate"]

        # 应用基于研究的校准，以压缩过高估计
        raw_point = point
        point = self.calibrate(point)

        ci_low, ci_high = self.bootstrap_interval(prepared)
        # 同时校准区间边界
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
        """估算全部学习者班级，并强制满足 C > F > P > K 顺序。"""

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
        """返回点估计的 bootstrap confidence interval。"""

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
        """将 CI 宽度 / estimate 映射为高、中或低。"""

        if point <= 0:
            return "低"
        ratio = (ci_high - ci_low) / point
        if ratio < self.config.confidence_high_ratio:
            return "高"
        if ratio < self.config.confidence_mid_ratio:
            return "中"
        return "低"

    def calibrate(self, estimate: float) -> float:
        """应用压缩以避免过高估计。

        两阶段流程：
          1. 基于 tanh 的平滑饱和（现有流程）— 锚定在母语者上限附近（约 20 kf）。
          2. 可选分段线性压缩（方案 B）— 在更高区间逐步加强压缩。

        公开词汇量数据：受教育母语者掌握约 20,000 个 word families
        （Nation 2006；Goulden et al. 1990）。中国 EFL 学习者上限约
        10,000-13,000（TEM-8 水平）。
        """

        if estimate <= 0:
            return estimate

        cfg = self.config
        # ---- 阶段 1：tanh 平滑饱和 ----
        max_v = float(cfg.calibration_native_max)
        k = cfg.calibration_k
        calibrated = max_v * math.tanh(k * estimate)
        # 对约 6,000 以上的估算逐渐压缩。
        # 对约 4,000 以下的估算，曲线接近线性。

        # ---- 阶段 2：分段线性压缩 ----
        if cfg.enable_piecewise_calibration:
            calibrated = self.piecewise_calibrate(calibrated)

        return float(calibrated)

    def piecewise_calibrate(self, estimate: float) -> float:
        """应用分段线性压缩（方案 B）。

        分段来自 config.piecewise_knots。
        当前：[0,3.5k] slope 1.00，[3.5k,6.5k] slope 0.55，
        [6.5k,12k] slope 0.30, [12k,22k] slope 0.15, >22k slope 0.15.

        设计目标：raw~11.4k 的用户最终约为 ~6.3k。
        """
        if estimate <= 0:
            return estimate

        knots = self.config.piecewise_knots  # 已排序的 (upper_boundary, slope)
        prev_boundary = 0.0
        prev_value = 0.0

        for boundary, slope in knots:
            if estimate <= boundary:
                return float(prev_value + (estimate - prev_boundary) * slope)
            # 累加经过该分段后的值
            prev_value += (float(boundary) - prev_boundary) * slope
            prev_boundary = float(boundary)

        # 超过最后一个 knot 后，继续使用最后一个分段的 slope
        return float(prev_value + (estimate - prev_boundary) * knots[-1][1])

    def map_level(self, estimate: float) -> str:
        """将词汇量估算映射到中国学习者等级。"""

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
        """应用 isotonic regression 以强制满足 C ≥ F ≥ P ≥ K。

        sklearn 可用时使用 PAVA（pool-adjacent-violators algorithm）；
        否则回退到简单贪心修正。
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
        """优先使用 sklearn 拟合 logistic regression，否则使用 NumPy GD。"""

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
        """拟合 *rank-weighted* logistic regression。

        样本按 ``_compute_weight(rank)`` 加权，使高频（低 rank）词上的错误
        比罕见词上的错误受到更强惩罚。相比未加权拟合，这会得到更保守的词汇量估算。

        优先尝试带 ``sample_weight`` 的 sklearn；失败则回退到加权 NumPy gradient descent。
        """
        sample_weights = np.array(
            [self._compute_weight(float(r)) for r in ranks],
            dtype=float,
        )
        # ---- 尝试 sklearn ----
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

        # ---- 加权 NumPy GD ----
        mean = float(np.mean(log_ranks))
        std = float(np.std(log_ranks) or 1.0)
        z = (log_ranks - mean) / std
        design = np.column_stack([np.ones_like(z), z])
        weights = np.zeros(2, dtype=float)

        for _ in range(self.config.logistic_max_iter):
            logits = design @ weights
            p = self._sigmoid(logits)
            # 加权 gradient：每个样本贡献 w_i * (p_i - y_i)
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
        """根据频率 rank 计算每个样本的权重。

        高频（低 rank）词若答错会获得完整权重（最强惩罚）。
        低频（高 rank）词权重逐步降低，使拟合聚焦于常见词汇。

        公式：w = 1 / (1 + log2(max(rank, 10) / 10))

        预期值（clamped）：
            rank=1   → w=1.000   （截断到最小 rank=10）
            rank=5000  → w≈0.100
            rank=20000 → w≈0.083
        """
        effective_rank = max(rank, 10.0)
        return float(1.0 / (1.0 + math.log2(effective_rank / 10.0)))

    @staticmethod
    def _pava_decreasing(values: list[float]) -> list[float]:
        """用于非递增序列的 pool-adjacent-violators algorithm。"""

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
            # 完全没有观测时，使用该 bucket 的 Bayesian prior
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
