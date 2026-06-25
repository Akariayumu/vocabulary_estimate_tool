"""使用 Rasch model 的两阶段分层词汇测验。

使用 ``stage_vocab.json``（11950 个词，含 difficulty / cluster_20 / cluster_100）
和 ``VocabBank``（21738 个词，含 wordfreq rank）实现：

1. Phase 1 — 在 20 个难度类别中抽取可配置数量的问题（默认 30）
2. Rasch MLE — 使用 scipy.optimize 拟合用户能力 θ
3. Phase 2 — 针对低置信类别（用户得分 1/2）追加精细问题
4. 词汇量估算 — 对 21738 个词库词求和 P(known | θ)

相比现有 bucket 方法的主要优势：
- 平滑概率曲线（无硬 rank 阈值）
- 自洽：简单词答对 + 难词答错会得到同一个 θ
- 分层采样：在难度 cluster 上保持均衡覆盖
"""

from __future__ import annotations

import json
import math
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import DEFAULT_CONFIG, EstimatorConfig
# 不再需要 VocabBank，直接使用 stage_vocab.json

# 类型别名
Response = tuple[str, bool]  # (word, known) 作答元组
QuizItem = dict[str, Any]  # word、difficulty、cluster_20、cluster_100、source 字段

STREAMING_CLUSTER_ORDER = [0, 19, 5, 15, 10, 2, 7, 12, 17, 4, 9, 14, 18, 1, 6, 11, 16, 3, 8, 13]
MIDDLE_CLUSTER_ORDER = [c20 for c20 in STREAMING_CLUSTER_ORDER if 5 <= c20 <= 14]

# ── Sigmoid（为数值稳定做 clamp）───────────────────────────────────────────────

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: float) -> float:
    """Logit 变换，并做 clamp 以避免无穷值。"""
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))


def _sigmoid_scalar(x: float) -> float:
    """标量 sigmoid，并做 clamp。"""
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ── 项目路径 ─────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAGE_VOCAB_PATH = _PROJECT_ROOT / "data" / "stage_vocab.json"


# ═══════════════════════════════════════════════════════════════════════════════
# 核心类
# ═══════════════════════════════════════════════════════════════════════════════


class StratifiedQuiz:
    """带 Rasch model 拟合的两阶段分层词汇测验。

    Usage::

        sq = StratifiedQuiz(vocab_bank)
        phase1 = sq.phase1_sample()
        # 用户作答 → responses
        theta, ci = sq.fit_ability(responses)  # Rasch MLE
        phase2 = sq.phase2_sample(theta, None)
        # 用户继续作答 → all_responses
        theta2, ci2 = sq.fit_ability(all_responses)
        estimate = sq.estimate_vocab(theta2)
    """

    # ── 公共 API ──────────────────────────────────────────────────────────
    _V1_VOCAB_SCALE = 0.8
    _V2_VOCAB_SCALE = 1.0
    _FIT_ABILITY_CI_Z = 1.96
    _V2_THETA_CI_Z = 1.28
    _V2_CONFIDENCE_HIGH_RATIO = 0.25
    _V2_CONFIDENCE_MID_RATIO = 0.50

    def __init__(
        self,
        vocab_bank = None,  # 已弃用，为 API 兼容性保留
        config: EstimatorConfig = DEFAULT_CONFIG,
        seed: int | None = None,
        stage_vocab_path: str | Path | None = None,
        phase1_question_count: int = 30,
    ) -> None:
        self.config = config
        if phase1_question_count < 1 or phase1_question_count > 40:
            raise ValueError("phase1_question_count must be between 1 and 40")
        self.phase1_question_count = int(phase1_question_count)
        effective_seed = seed if seed is not None else config.random_seed
        self.rng = random.Random() if effective_seed == 0 else random.Random(effective_seed)
        self.bank = None  # 不再加载 wordfreq

        # 加载 stage_vocab
        sv_path = Path(stage_vocab_path or _STAGE_VOCAB_PATH)
        with open(sv_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.vocab_version = self._detect_vocab_version(sv_path, raw)

        self._word_to_stage: dict[str, dict] = raw["word_to_stage"]
        # 仅保留带有 difficulty、cluster_20、cluster_100 的词
        self._candidates: list[dict] = []
        for word, info in self._word_to_stage.items():
            diff = info.get("difficulty")
            c20 = info.get("cluster_20")
            c100 = info.get("cluster_100")
            if diff is not None and c20 is not None and c100 is not None:
                self._candidates.append({
                    "word": word,
                    "difficulty": diff,
                    "cluster_20": int(c20),
                    "cluster_100": int(c100),
                    "source": "stage_vocab",
                })

        # 按 cluster_20 和 cluster_100 建立候选索引
        self._by_c20: dict[int, list[dict]] = {}
        self._by_c100: dict[int, list[dict]] = {}
        for c in self._candidates:
            self._by_c20.setdefault(c["cluster_20"], []).append(c)
            self._by_c100.setdefault(c["cluster_100"], []).append(c)

        # 每个 cluster_20 内按难度排序
        self._c20_sorted: dict[int, list[dict]] = {}
        for c20, items in self._by_c20.items():
            self._c20_sorted[c20] = sorted(items, key=lambda x: x["difficulty"])

        # 为所有词库词预计算 logit difficulty（包括不在 stage_vocab 中的词）
        self._word_difficulties: dict[str, float] = self._build_word_difficulties()

    def phase1_sample(self, adaptive: bool = True, rng: random.Random | None = None) -> list[dict]:
        """按适合流式展示的顺序生成 Phase 1 问题。

        默认 30 题路径会先覆盖全部 20 个难度类别各一次，
        再从 10 个中间类别各追加一题。40 题兼容路径保留原先的
        20 类 × 2 结构。

        Args:
            adaptive: True 时使用分散 + 按类别采样；False 时使用均衡采样。
            rng: 每个请求独立的随机源（避免并发问题）。
        """
        _rng = rng or self.rng
        if not adaptive:
            return self._phase1_balanced(rng=_rng)

        return self._phase1_streaming(rng=_rng)

    def _phase1_streaming(self, rng: random.Random | None = None) -> list[dict]:
        """按可用前缀排序的可配置 Phase 1 采样器。"""
        _rng = rng or self.rng
        selected: list[dict] = []
        seen_words: set[str] = set()
        per_class_needed = self._phase1_class_counts()
        picks_by_class: dict[int, list[dict]] = {}

        for c20 in STREAMING_CLUSTER_ORDER:
            needed = per_class_needed.get(c20, 0)
            if needed <= 0:
                continue
            picks = self._pick_from_class(c20, needed, exclude=seen_words, strategy="balanced", rng=_rng)
            picks_by_class[c20] = picks
            for p in picks:
                seen_words.add(p["word"])

        for c20 in STREAMING_CLUSTER_ORDER:
            picks = picks_by_class.get(c20, [])
            if picks:
                selected.append(picks[0])

        for c20 in self._phase1_topup_order():
            picks = picks_by_class.get(c20, [])
            if len(picks) > 1:
                selected.append(picks[1])

        return selected[: self.phase1_question_count]

    def fit_ability(
        self,
        responses: list[Response],
    ) -> tuple[float, tuple[float, float]]:
        """通过 MLE 从 (word, known) responses 拟合 Rasch 能力 θ。

        Returns:
            (theta, (ci_low, ci_high))，其中 ci 是 95% confidence interval。
        """
        prepared = self._prepare_responses(responses)
        if len(prepared) < 3:
            return (0.0, (-2.0, 2.0))

        difficulties = np.array([d for _, d, _ in prepared], dtype=float)
        y = np.array([y for _, _, y in prepared], dtype=float)

        # 对 difficulty 做 logit 变换，使其匹配 θ 的尺度
        logit_d = np.array([_logit(max(0.001, min(0.999, d))) for d in difficulties], dtype=float)

        # 通过 scipy 做 MLE
        result = self._mle_theta(logit_d, y)
        theta = float(result["theta"])

        # Fisher information 信息量
        p = _SIGMOID(theta - logit_d)
        fish = float(np.sum(p * (1.0 - p)))  # Fisher 计算式 = Σ σ·(1-σ)
        se = 1.0 / math.sqrt(max(fish, 1e-10)) if fish > 0 else 5.0

        ci_low = theta - self._FIT_ABILITY_CI_Z * se
        ci_high = theta + self._FIT_ABILITY_CI_Z * se
        return (theta, (ci_low, ci_high))

    def phase2_sample(
        self,
        theta: float,
        low_confidence_classes: list[int] | None = None,
        responses: list[Response] | None = None,
        n_per_class: int = 4,
        exclude: set[str] | None = None,
    ) -> list[dict]:
        """已弃用：为低置信类别生成 Phase 2 精细问题。

        ``scripts/explore_question_count.py`` 中的实验表明，相比仅用 Phase 1，
        Phase 2 带来的精度提升可以忽略。新的生产流程保留此方法以兼容旧调用，
        但默认不再调用。

        对用户恰好得分 1/2 的每个类别，从同一个 cluster_20 类别中追加
        ``n_per_class`` 个问题。这些追加问题信息量最大
        （最接近当前 θ 估计）。

        Phase 2 的词不会与 ``exclude`` 中的词重叠
        （通常是 Phase 1 已出现的词）。

        Args:
            theta: 当前 Rasch 能力估计。
            low_confidence_classes: 可选的 cluster_20 值列表。
                若为 None，则从 responses 计算。
            responses: Phase 1 responses（当 ``low_confidence_classes`` 为 None 时需要）。
            n_per_class: 每个不确定类别追加的问题数。
            exclude: 需要排除的词（例如 Phase 1 已见词）。

        Returns:
            Phase 2 的 quiz item 列表。
        """
        if low_confidence_classes is None:
            if responses is None:
                return []
            low_confidence_classes = self._identify_low_confidence(responses)

        selected: list[dict] = []
        seen_words: set[str] = set(exclude) if exclude else set()

        for c20 in low_confidence_classes:
            items = self._c20_sorted.get(c20, [])
            # 按当前 θ 下的信息量打分
            scored: list[tuple[float, dict]] = []
            for item in items:
                if item["word"] in seen_words:
                    continue
                d_logit = _logit(max(0.001, min(0.999, item["difficulty"])))
                info_val = self._item_information(theta, d_logit)
                scored.append((info_val, item))

            scored.sort(key=lambda x: -x[0])
            picks = scored[:n_per_class]
            for _, item in picks:
                selected.append(item)
                seen_words.add(item["word"])

        return selected

    def estimate_vocab(
        self,
        theta: float,
    ) -> dict:
        """根据 Rasch θ 估算总词汇量。

        返回包含 point_estimate、各词源贡献等信息的 dict。
        """
        total = 0.0
        contributions: dict[str, float] = {}

        for word, difficulty in self._word_difficulties.items():
            d_logit = _logit(max(0.001, min(0.999, difficulty)))
            p = _sigmoid_scalar(theta - d_logit)
            total += p

        return {
            "theta": theta,
            "vocab_estimate": round(total),
            "raw_estimate": round(total),
            "method": "rasch_sum_P_known",
        }

    def estimate_with_ci(
        self,
        responses: list[Response],
    ) -> dict:
        """已弃用的兼容估算器：fit + vocab + CI。

        此方法保留给旧 API 调用方。当前默认测验流程使用 ``stream_estimate``，
        且不运行 Phase 2，因为题量实验发现仅用 Phase 1 已经足够。

        返回与现有 API 格式兼容的 dict。
        """
        theta, theta_ci = self.fit_ability(responses)
        ci_low, ci_high = self._vocab_ci_theta_bounds(theta, theta_ci)
        vocab_raw = self._vocab_at_theta(theta)
        vocab_low = self._vocab_at_theta(ci_low)
        vocab_high = self._vocab_at_theta(ci_high)

        prepared = self._prepare_responses(responses)
        sample_size = len(prepared)
        ignored = len(responses) - sample_size

        # 校准（与现有 VocabEstimator 使用同一流程）
        cal_raw = self._calibrate(vocab_raw)
        cal_low = self._calibrate(vocab_low)
        cal_high = self._calibrate(vocab_high)

        point = round(cal_raw)
        ci_vocab = [round(cal_low), round(cal_high)]

        return {
            "theta": round(theta, 4),
            "theta_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "point_estimate": point,
            "vocabulary_range": ci_vocab,
            "confidence_interval_90": ci_vocab,
            "raw_vocab_estimate": round(vocab_raw),
            "confidence": self._confidence_label(cal_raw, cal_low, cal_high),
            "sample_size": sample_size,
            "ignored_responses": ignored,
            "level": self._map_level(cal_raw),
            "method": "rasch_stratified_v2",
        }

    def stream_estimate(
        self,
        responses: list[Response],
        next_count: int = 5,
        rng: random.Random | None = None,
        phase1_items: list[dict] | None = None,
    ) -> dict:
        """基于全部已答 responses 估算，并建议下一批题目。

        ``continue_available`` 基于配置的 Phase 1 题量判断。
        建议题目沿用相同的流式 cluster 顺序，并排除已答词。
        """
        theta, theta_ci = self.fit_ability(responses)
        se = self._theta_se_from_fit_ci(theta_ci)
        ci_low, ci_high = self._vocab_ci_theta_bounds(theta, theta_ci)
        vocab_raw = self._vocab_at_theta(theta)
        vocab_low = self._vocab_at_theta(ci_low)
        vocab_high = self._vocab_at_theta(ci_high)

        cal_raw = self._calibrate(vocab_raw)
        cal_low = self._calibrate(vocab_low)
        cal_high = self._calibrate(vocab_high)
        vocab_ci = [round(cal_low), round(cal_high)]

        prepared = self._prepare_responses(responses)
        n_questions = len(prepared)
        continue_available = n_questions < self.phase1_question_count

        answered = {word.strip().lower() for word, _ in responses}
        candidates = phase1_items if phase1_items is not None else self.phase1_sample(rng=rng)
        suggested_items = [
            item for item in candidates
            if item["word"].strip().lower() not in answered
        ][: max(0, next_count)]

        point = round(cal_raw)
        return {
            "theta": round(theta, 4),
            "se": round(se, 4),
            "theta_ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "vocab_raw": round(vocab_raw),
            "vocab_ci": vocab_ci,
            "point_estimate": point,
            "vocabulary_range": vocab_ci,
            "confidence_interval_90": vocab_ci,
            "raw_vocab_estimate": round(vocab_raw),
            "confidence": self._confidence_label(cal_raw, cal_low, cal_high),
            "level": self._map_level(cal_raw),
            "n_questions": n_questions,
            "sample_size": n_questions,
            "ignored_responses": len(responses) - n_questions,
            "continue_available": continue_available,
            "suggested_items": suggested_items,
            "method": "rasch_stratified_v2_stream",
        }

    def get_sampling_info(self) -> dict:
        """返回词池结构的元数据。"""
        return {
            "phase1_question_count": self.phase1_question_count,
            "streaming_cluster_order": STREAMING_CLUSTER_ORDER,
            "middle_cluster_order": MIDDLE_CLUSTER_ORDER,
            "enable_phase2": self.config.enable_phase2,
            "stage_vocab_words": len(self._candidates),
            "bank_words": len(self._candidates),
            "cluster_20_count": len(self._by_c20),
            "c20_sizes": {k: len(v) for k, v in sorted(self._by_c20.items())},
            "cluster_100_count": len(self._by_c100),
        }

    # ── 内部：Rasch 拟合 ─────────────────────────────────────────────────

    def _mle_theta(self, d_logit: np.ndarray, y: np.ndarray) -> dict:
        """带 N(0,2) prior 的 θ MAP 估计（不依赖 scipy）。
        
        prior 可避免全对/全错 responses 产生极端 θ。"""
        PRIOR_VAR = 2.0  # N(0, 2)
        p_obs = float(np.mean(y))
        theta0 = _logit(max(0.01, min(0.99, p_obs)))

        best_theta = theta0
        best_nll = float("inf")

        for start in [theta0, 0.0, -2.0, 2.0, -5.0, 5.0]:
            theta = float(start)
            for _ in range(100):
                logits = theta - d_logit
                p = np.clip(_SIGMOID(logits), 1e-15, 1.0 - 1e-15)
                # Gradient 公式：Σ(σ - y) + θ/σ²（MAP prior）
                g = float(np.sum(p - y)) + theta / PRIOR_VAR
                # Hessian 公式：Σ(σ·(1-σ)) + 1/σ²
                h = float(np.sum(p * (1.0 - p))) + 1.0 / PRIOR_VAR
                if abs(h) < 1e-12:
                    break
                step = g / h
                theta = np.clip(theta - step, -10.0, 10.0)
                if abs(step) < 1e-8:
                    break

            final_logits = theta - d_logit
            final_p = np.clip(_SIGMOID(final_logits), 1e-15, 1.0 - 1e-15)
            nll = float(-np.sum(y * np.log(final_p) + (1.0 - y) * np.log(1.0 - final_p)))
            nll += 0.5 * theta**2 / PRIOR_VAR  # log prior 项

            if nll < best_nll:
                best_nll = nll
                best_theta = theta

        return {"theta": best_theta, "nll": best_nll}

    # ── 内部：辅助方法 ───────────────────────────────────────────────────

    def _phase1_balanced(self, rng: random.Random | None = None) -> list[dict]:
        """使用配置题量的非 adaptive Phase 1。"""
        _rng = rng or self.rng
        selected: list[dict] = []
        seen: set[str] = set()
        for c20, needed in self._phase1_class_counts().items():
            picks = self._pick_from_class(c20, needed, exclude=seen, strategy="balanced", rng=_rng)
            for p in picks:
                selected.append(p)
                seen.add(p["word"])
        return selected[: self.phase1_question_count]

    def _phase1_class_counts(self) -> dict[int, int]:
        """返回每个 cluster_20 需要抽取的 Phase 1 题数。"""
        counts: dict[int, int] = {}
        first_wave = min(self.phase1_question_count, len(STREAMING_CLUSTER_ORDER))
        for c20 in STREAMING_CLUSTER_ORDER[:first_wave]:
            counts[c20] = 1

        extra = max(0, self.phase1_question_count - len(STREAMING_CLUSTER_ORDER))
        for c20 in self._phase1_topup_order()[:extra]:
            counts[c20] = counts.get(c20, 0) + 1
        return counts

    def _phase1_topup_order(self) -> list[int]:
        """返回第二轮类别顺序。

        前 10 个补充名额指向中间类别（5-14），这是 30 题方案的最优选择。
        超过 30 题后继续使用剩余类别，以保留旧的 40 题 20×2 路径。
        """
        if self.phase1_question_count >= 40:
            return list(STREAMING_CLUSTER_ORDER)
        middle = list(MIDDLE_CLUSTER_ORDER)
        remaining = [c20 for c20 in STREAMING_CLUSTER_ORDER if c20 not in set(middle)]
        return middle + remaining

    def _pick_from_class(
        self,
        c20: int,
        n: int,
        exclude: set[str],
        strategy: str = "balanced",
        rng: random.Random | None = None,
    ) -> list[dict]:
        """从某个 cluster_20 类别中选择 n 个词。

        Args:
            rng: 每个请求独立的随机源（避免并发问题）。
        """
        _rng = rng or self.rng
        items = self._c20_sorted.get(c20, [])
        available = [i for i in items if i["word"] not in exclude]

        # 若词池太小，则从相邻类别借词
        if len(available) < 10:
            for offset in [1, -1, 2, -2, 3, -3]:
                neighbor = c20 + offset
                if neighbor < 0 or neighbor > 19:
                    continue
                neighbor_items = self._c20_sorted.get(neighbor, [])
                existing_words = {i["word"] for i in available}
                for i in neighbor_items:
                    if i["word"] not in exclude and i["word"] not in existing_words:
                        available.append(i)
                        existing_words.add(i["word"])
                        if len(available) >= 15:
                            break
                if len(available) >= 15:
                    break

        if not available:
            return []

        # 排序前先 shuffle，让相同 difficulty 的词随机化
        _rng.shuffle(available)
        available_sorted = sorted(available, key=lambda x: x["difficulty"])

        if strategy == "extremes":
            # 从两端选择
            picks = []
            if len(available_sorted) >= 2:
                picks = [available_sorted[0], available_sorted[-1]]
            elif available:
                picks = [available_sorted[0]]
            return picks[:n]

        if strategy == "mid":
            # 选择中位难度附近的词
            if not available_sorted:
                return []
            mid = len(available_sorted) // 2
            picks = available_sorted[mid:mid + n]
            return picks

        if strategy == "balanced":
            # 在难度范围内分散选择 n 个词，并加入随机性
            m = len(available_sorted)
            if m <= n:
                return available_sorted[:n]
            # 分成 n 段，并从每段随机选择一个词
            picked_indices = set()
            segment_size = m / n
            for i in range(n):
                seg_start = int(i * segment_size)
                seg_end = int((i + 1) * segment_size) - 1
                if seg_end >= m:
                    seg_end = m - 1
                if seg_start > seg_end:
                    seg_start = seg_end
                if seg_start == seg_end:
                    idx = seg_start
                else:
                    idx = _rng.randint(seg_start, seg_end)
                # 如果随机落到同一索引，则避免重复
                while idx in picked_indices and seg_end > seg_start:
                    idx = _rng.randint(seg_start, seg_end)
                picked_indices.add(idx)
            return [available_sorted[i] for i in sorted(picked_indices)][:n]

        if strategy == "informative":
            # 已按信息量排序，直接取前 n 个
            return available_sorted[:n]

        return available_sorted[:n]

    def _prepare_responses(self, responses: list[Response]) -> list[tuple[str, float, bool]]:
        """准备 (word, difficulty, known) 元组，并过滤未索引的词。"""
        result: list[tuple[str, float, bool]] = []
        for word, known in responses:
            w = word.strip().lower()
            info = self._word_to_stage.get(w)
            if info and "difficulty" in info:
                result.append((w, float(info["difficulty"]), bool(known)))
        return result

    def _identify_low_confidence(self, responses: list[Response]) -> list[int]:
        """识别用户恰好答对 1/2 的 cluster_20 类别。

        每类 2 题时：
          - 0/2 = 高置信未知
          - 2/2 = 高置信已知
          - 1/2 = 不确定 → 需要精细追问
        """
        from collections import defaultdict

        class_counts: dict[int, list[bool]] = defaultdict(list)
        for word, known in responses:
            info = self._word_to_stage.get(word.lower())
            if info and "cluster_20" in info:
                c20 = int(info["cluster_20"])
                class_counts[c20].append(bool(known))

        low_conf: list[int] = []
        for c20, vals in class_counts.items():
            if len(vals) == 2:
                correct = sum(vals)
                if correct == 1:
                    low_conf.append(c20)

        return sorted(low_conf)

    def _build_word_difficulties(self) -> dict[str, float]:
        """为词库中每个词构建 difficulty。

        对 stage_vocab 中的词：使用已存 difficulty。
        对仅在 bank 中的词：根据 wordfreq rank 估算。
        """
        difficulties: dict[str, float] = {}

        # 优先使用 stage_vocab 中的 difficulty
        for c in self._candidates:
            difficulties[c["word"]] = c["difficulty"]

        # 所有词都由 stage_vocab 覆盖，不需要 bank-only fallback

        return difficulties

    def _vocab_at_theta(self, theta: float) -> float:
        """对所有词库词求和 P(known | θ)，再应用版本校准。"""
        total = 0.0
        for word, difficulty in self._word_difficulties.items():
            d_logit = _logit(max(0.001, min(0.999, difficulty)))
            total += _sigmoid_scalar(theta - d_logit)
        return total * self._vocab_scale()

    @staticmethod
    def _detect_vocab_version(path: Path, raw: dict) -> str:
        """根据词库路径识别校准口径。"""
        if "v2" in path.name.lower():
            return "v2"
        if len(raw.get("word_to_stage", {})) > 15_000:
            return "v2"
        return "v1"

    def _is_v2_vocab(self) -> bool:
        return self.vocab_version == "v2"

    def _vocab_scale(self) -> float:
        if self._is_v2_vocab():
            return self._V2_VOCAB_SCALE
        return self._V1_VOCAB_SCALE

    def _theta_se_from_fit_ci(self, theta_ci: tuple[float, float]) -> float:
        ci_low, ci_high = theta_ci
        return (ci_high - ci_low) / (2.0 * self._FIT_ABILITY_CI_Z)

    def _vocab_ci_theta_bounds(self, theta: float, theta_ci: tuple[float, float]) -> tuple[float, float]:
        if not self._is_v2_vocab():
            return theta_ci
        se = self._theta_se_from_fit_ci(theta_ci)
        return (
            theta - self._V2_THETA_CI_Z * se,
            theta + self._V2_THETA_CI_Z * se,
        )

    @staticmethod
    def _item_information(theta: float, d_logit: float) -> float:
        """能力 θ 下单个题目的 Fisher information。

        ``I_i(θ) = σ(θ - d_i)·(1 - σ(θ - d_i))``
        当 θ = d_i 时达到最大值（σ = 0.5 → I = 0.25）。
        """
        p = _sigmoid_scalar(theta - d_logit)
        return p * (1.0 - p)

    def _calibrate(self, estimate: float) -> float:
        """应用与 VocabEstimator 相同的 tanh + piecewise 校准。"""
        if estimate <= 0:
            return estimate

        # Tanh 饱和
        max_v = float(self.config.calibration_native_max)
        k = self.config.calibration_k
        cal = max_v * math.tanh(k * estimate)

        # 分段校准
        if self.config.enable_piecewise_calibration:
            cal = self._piecewise_calibrate(cal)

        return float(cal)

    def _piecewise_calibrate(self, estimate: float) -> float:
        """分段线性校准。"""
        if estimate <= 0:
            return estimate
        knots = self.config.piecewise_knots
        prev_boundary = 0.0
        prev_value = 0.0
        for boundary, slope in knots:
            if estimate <= boundary:
                return float(prev_value + (estimate - prev_boundary) * slope)
            prev_value += (float(boundary) - prev_boundary) * slope
            prev_boundary = float(boundary)
        return float(prev_value + (estimate - prev_boundary) * knots[-1][1])

    def _map_level(self, estimate: float) -> str:
        """将词汇量估算映射到中国学习者等级。"""
        thresholds = []
        for name, low, high in self.config.levels:
            thresholds.append((name, low, high))

        margin = self.config.transition_margin
        for idx, (name, low, high) in enumerate(thresholds):
            if high is None:
                if estimate >= low:
                    if abs(estimate - low) <= margin and idx > 0:
                        return f"{thresholds[idx - 1][0]}/{name}过渡"
                    return name
                continue

            if abs(estimate - high) <= margin and idx + 1 < len(thresholds):
                return f"{name}/{thresholds[idx + 1][0]}过渡"
            if low <= estimate < high:
                return name

        if estimate < thresholds[0][1]:
            return "初中以下"
        return thresholds[-1][0]

    def _confidence_label(self, point: float, ci_low: float, ci_high: float) -> str:
        """将 CI 宽度 / estimate 映射为高、中或低。"""
        if point <= 0:
            return "低"
        ratio = (ci_high - ci_low) / point
        high_ratio = self.config.confidence_high_ratio
        mid_ratio = self.config.confidence_mid_ratio
        if self._is_v2_vocab():
            high_ratio = self._V2_CONFIDENCE_HIGH_RATIO
            mid_ratio = self._V2_CONFIDENCE_MID_RATIO
        if ratio < high_ratio:
            return "高"
        if ratio < mid_ratio:
            return "中"
        return "低"
