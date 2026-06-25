"""Two-stage stratified vocabulary quiz using Rasch model.

Uses ``stage_vocab.json`` (11950 words with difficulty / cluster_20 / cluster_100)
and ``VocabBank`` (21738 words with wordfreq rank) to implement:

1. Phase 1 — configurable questions across 20 difficulty classes (default 30)
2. Rasch MLE — fit user ability θ using scipy.optimize
3. Phase 2 — refined questions for low-confidence classes (where user scored 1/2)
4. Vocabulary estimate — sum P(known | θ) over all 21738 bank words

Key advantages over the existing bucket-based approach:
- Smooth probability curve (no hard rank thresholds)
- Self-consistent: easy-correct + hard-incorrect yields one θ
- Stratified sampling: balanced coverage across difficulty clusters
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
# VocabBank no longer needed — we use stage_vocab.json directly

# Type aliases
Response = tuple[str, bool]  # (word, known)
QuizItem = dict[str, Any]  # word, difficulty, cluster_20, cluster_100, source

STREAMING_CLUSTER_ORDER = [0, 19, 5, 15, 10, 2, 7, 12, 17, 4, 9, 14, 18, 1, 6, 11, 16, 3, 8, 13]
MIDDLE_CLUSTER_ORDER = [c20 for c20 in STREAMING_CLUSTER_ORDER if 5 <= c20 <= 14]

# ── Sigmoid (clamped for numerical stability) ─────────────────────────────────

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _logit(p: float) -> float:
    """Logit transform, clamped to avoid infinities."""
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))


def _sigmoid_scalar(x: float) -> float:
    """Scalar sigmoid, clamped."""
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# ── Project paths ─────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_STAGE_VOCAB_PATH = _PROJECT_ROOT / "data" / "stage_vocab.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Core class
# ═══════════════════════════════════════════════════════════════════════════════


class StratifiedQuiz:
    """Two-stage stratified vocabulary quiz with Rasch-model fitting.

    Usage::

        sq = StratifiedQuiz(vocab_bank)
        phase1 = sq.phase1_sample()
        # User answers → responses
        theta, ci = sq.fit_ability(responses)  # Rasch MLE
        phase2 = sq.phase2_sample(theta, None)
        # User answers more → all_responses
        theta2, ci2 = sq.fit_ability(all_responses)
        estimate = sq.estimate_vocab(theta2)
    """

    # ── Public API ────────────────────────────────────────────────────────

    def __init__(
        self,
        vocab_bank = None,  # deprecated, kept for API compatibility
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
        self.bank = None  # no longer loads wordfreq

        # Load stage_vocab
        sv_path = Path(stage_vocab_path or _STAGE_VOCAB_PATH)
        with open(sv_path, encoding="utf-8") as f:
            raw = json.load(f)

        self._word_to_stage: dict[str, dict] = raw["word_to_stage"]
        # Only keep words that have difficulty, cluster_20, cluster_100
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

        # Index candidates by cluster_20 and cluster_100
        self._by_c20: dict[int, list[dict]] = {}
        self._by_c100: dict[int, list[dict]] = {}
        for c in self._candidates:
            self._by_c20.setdefault(c["cluster_20"], []).append(c)
            self._by_c100.setdefault(c["cluster_100"], []).append(c)

        # Sorted difficulty for each cluster_20
        self._c20_sorted: dict[int, list[dict]] = {}
        for c20, items in self._by_c20.items():
            self._c20_sorted[c20] = sorted(items, key=lambda x: x["difficulty"])

        # Precompute logit-difficulty for all bank words (including those not in stage_vocab)
        self._word_difficulties: dict[str, float] = self._build_word_difficulties()

    def phase1_sample(self, adaptive: bool = True, rng: random.Random | None = None) -> list[dict]:
        """Generate Phase 1 questions in streaming-friendly order.

        The default 30-question path covers all 20 difficulty classes once,
        then adds one extra item from the 10 middle classes. The 40-question
        compatibility path keeps the previous 20 classes × 2 structure.

        Args:
            adaptive: If True, spread + per-class sampling; if False, balanced.
            rng: Per-request random source (prevents concurrency bugs).
        """
        _rng = rng or self.rng
        if not adaptive:
            return self._phase1_balanced(rng=_rng)

        return self._phase1_streaming(rng=_rng)

    def _phase1_streaming(self, rng: random.Random | None = None) -> list[dict]:
        """Configurable Phase 1 sampler ordered for useful prefixes."""
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
        """Fit Rasch ability θ from (word, known) responses via MLE.

        Returns:
            (theta, (ci_low, ci_high)) where ci is the 95% confidence interval.
        """
        prepared = self._prepare_responses(responses)
        if len(prepared) < 3:
            return (0.0, (-2.0, 2.0))

        difficulties = np.array([d for _, d, _ in prepared], dtype=float)
        y = np.array([y for _, _, y in prepared], dtype=float)

        # Logit-transform difficulties to match θ's scale
        logit_d = np.array([_logit(max(0.001, min(0.999, d))) for d in difficulties], dtype=float)

        # MLE via scipy
        result = self._mle_theta(logit_d, y)
        theta = float(result["theta"])

        # Fisher information
        p = _SIGMOID(theta - logit_d)
        fish = float(np.sum(p * (1.0 - p)))  # Fisher = Σ σ·(1-σ)
        se = 1.0 / math.sqrt(max(fish, 1e-10)) if fish > 0 else 5.0

        ci_low = theta - 1.96 * se
        ci_high = theta + 1.96 * se
        return (theta, (ci_low, ci_high))

    def phase2_sample(
        self,
        theta: float,
        low_confidence_classes: list[int] | None = None,
        responses: list[Response] | None = None,
        n_per_class: int = 4,
        exclude: set[str] | None = None,
    ) -> list[dict]:
        """Deprecated: generate Phase 2 refined questions for low-confidence classes.

        Experiments in ``scripts/explore_question_count.py`` showed Phase 2
        adds negligible accuracy over Phase 1-only estimates. New production
        flows keep this method for compatibility but do not call it by default.

        For each class where the user scored exactly 1/2, add
        ``n_per_class`` additional questions from the same
        cluster_20 class. These extra questions are the most
        informative (closest to current θ estimate).

        Phase 2 words will NOT overlap with those in ``exclude``
        (typically Phase 1 words).

        Args:
            theta: Current Rasch ability estimate.
            low_confidence_classes: Optional list of cluster_20 values.
                If None, computed from responses.
            responses: Phase 1 responses (needed if
                ``low_confidence_classes`` is None).
            n_per_class: Number of extra questions per uncertain class.
            exclude: Words to exclude (e.g. Phase 1 seen words).

        Returns:
            List of quiz items for Phase 2.
        """
        if low_confidence_classes is None:
            if responses is None:
                return []
            low_confidence_classes = self._identify_low_confidence(responses)

        selected: list[dict] = []
        seen_words: set[str] = set(exclude) if exclude else set()

        for c20 in low_confidence_classes:
            items = self._c20_sorted.get(c20, [])
            # Score by information content at current θ
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
        """Estimate total vocabulary size from Rasch θ.

        Returns dict with point_estimate, contributions per word source, etc.
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
        """Deprecated compatibility estimator: fit + vocab + CI.

        This method remains for old API callers. The current default quiz flow
        uses ``stream_estimate`` and does not run Phase 2, following the
        question-count experiment that found Phase 1-only estimates sufficient.

        Returns a dict compatible with the existing API format.
        """
        theta, (ci_low, ci_high) = self.fit_ability(responses)
        vocab_raw = self._vocab_at_theta(theta)
        vocab_low = self._vocab_at_theta(ci_low)
        vocab_high = self._vocab_at_theta(ci_high)

        prepared = self._prepare_responses(responses)
        sample_size = len(prepared)
        ignored = len(responses) - sample_size

        # Calibrate (same pipeline as existing VocabEstimator)
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
        """Estimate from all answered responses and suggest the next items.

        ``continue_available`` is based on the configured Phase 1 question
        count. Suggested items follow the same streaming cluster order and
        exclude already answered words.
        """
        theta, (ci_low, ci_high) = self.fit_ability(responses)
        se = (ci_high - ci_low) / (2.0 * 1.96)
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
        """Return metadata about the word pool structure."""
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

    # ── Internal: Rasch fitting ──────────────────────────────────────────

    def _mle_theta(self, d_logit: np.ndarray, y: np.ndarray) -> dict:
        """MAP for θ with N(0,2) prior (no scipy dependency).
        
        Prior prevents extreme θ for all-correct/all-wrong responses."""
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
                # Gradient: Σ(σ - y) + θ/σ² (MAP prior)
                g = float(np.sum(p - y)) + theta / PRIOR_VAR
                # Hessian: Σ(σ·(1-σ)) + 1/σ²
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
            nll += 0.5 * theta**2 / PRIOR_VAR  # log prior

            if nll < best_nll:
                best_nll = nll
                best_theta = theta

        return {"theta": best_theta, "nll": best_nll}

    # ── Internal: helpers ────────────────────────────────────────────────

    def _phase1_balanced(self, rng: random.Random | None = None) -> list[dict]:
        """Non-adaptive Phase 1 with the configured question count."""
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
        """Return how many Phase 1 items to draw from each cluster_20."""
        counts: dict[int, int] = {}
        first_wave = min(self.phase1_question_count, len(STREAMING_CLUSTER_ORDER))
        for c20 in STREAMING_CLUSTER_ORDER[:first_wave]:
            counts[c20] = 1

        extra = max(0, self.phase1_question_count - len(STREAMING_CLUSTER_ORDER))
        for c20 in self._phase1_topup_order()[:extra]:
            counts[c20] = counts.get(c20, 0) + 1
        return counts

    def _phase1_topup_order(self) -> list[int]:
        """Return the second-wave class order.

        The first 10 top-up slots target the middle classes (5-14), which is
        the 30-question optimum. Counts above 30 continue with the remaining
        classes to preserve the old 40-question 20×2 path.
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
        """Pick n words from a cluster_20 class.

        Args:
            rng: Per-request random source (prevents concurrency bugs).
        """
        _rng = rng or self.rng
        items = self._c20_sorted.get(c20, [])
        available = [i for i in items if i["word"] not in exclude]

        # Borrow from adjacent classes if pool is too small
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

        # Shuffle before sorting so identical difficulties get randomized
        _rng.shuffle(available)
        available_sorted = sorted(available, key=lambda x: x["difficulty"])

        if strategy == "extremes":
            # Pick from extremes
            picks = []
            if len(available_sorted) >= 2:
                picks = [available_sorted[0], available_sorted[-1]]
            elif available:
                picks = [available_sorted[0]]
            return picks[:n]

        if strategy == "mid":
            # Pick median difficulty words
            if not available_sorted:
                return []
            mid = len(available_sorted) // 2
            picks = available_sorted[mid:mid + n]
            return picks

        if strategy == "balanced":
            # Pick n words spread across the difficulty range, with randomness
            m = len(available_sorted)
            if m <= n:
                return available_sorted[:n]
            # Divide into n segments and pick one random word from each
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
                # Avoid duplicates if randomization lands on same index
                while idx in picked_indices and seg_end > seg_start:
                    idx = _rng.randint(seg_start, seg_end)
                picked_indices.add(idx)
            return [available_sorted[i] for i in sorted(picked_indices)][:n]

        if strategy == "informative":
            # Already sorted by information — just take top n
            return available_sorted[:n]

        return available_sorted[:n]

    def _prepare_responses(self, responses: list[Response]) -> list[tuple[str, float, bool]]:
        """Prepare (word, difficulty, known) tuples, filtering out unindexed words."""
        result: list[tuple[str, float, bool]] = []
        for word, known in responses:
            w = word.strip().lower()
            info = self._word_to_stage.get(w)
            if info and "difficulty" in info:
                result.append((w, float(info["difficulty"]), bool(known)))
        return result

    def _identify_low_confidence(self, responses: list[Response]) -> list[int]:
        """Identify cluster_20 classes where user got exactly 1/2 correct.

        With 2 questions per class:
          - 0/2 = confident unknown
          - 2/2 = confident known
          - 1/2 = uncertain → needs refinement
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
        """Build difficulty for every word in the bank.

        For words in stage_vocab: use stored difficulty.
        For bank-only words: estimate from wordfreq rank.
        """
        difficulties: dict[str, float] = {}

        # Prefer stage_vocab difficulties
        for c in self._candidates:
            difficulties[c["word"]] = c["difficulty"]

        # All words covered by stage_vocab — no bank-only fallback needed

        return difficulties

    def _vocab_at_theta(self, theta: float) -> float:
        """Compute sum of P(known | θ) over all bank words, then apply 0.8 calibration."""
        total = 0.0
        for word, difficulty in self._word_difficulties.items():
            d_logit = _logit(max(0.001, min(0.999, difficulty)))
            total += _sigmoid_scalar(theta - d_logit)
        # Bug 1: empirical calibration — raw estimates are ~25% high
        return total * 0.8

    @staticmethod
    def _item_information(theta: float, d_logit: float) -> float:
        """Fisher information of one item at ability θ.

        ``I_i(θ) = σ(θ - d_i)·(1 - σ(θ - d_i))``
        This is maximised when θ = d_i (σ = 0.5 → I = 0.25).
        """
        p = _sigmoid_scalar(theta - d_logit)
        return p * (1.0 - p)

    def _calibrate(self, estimate: float) -> float:
        """Apply the same tanh + piecewise calibration as VocabEstimator."""
        if estimate <= 0:
            return estimate

        # Tanh saturation
        max_v = float(self.config.calibration_native_max)
        k = self.config.calibration_k
        cal = max_v * math.tanh(k * estimate)

        # Piecewise calibration
        if self.config.enable_piecewise_calibration:
            cal = self._piecewise_calibrate(cal)

        return float(cal)

    def _piecewise_calibrate(self, estimate: float) -> float:
        """Piecewise linear calibration."""
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
        """Map a vocabulary-size estimate to a Chinese learner level."""
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
        """Map CI width / estimate to 高, 中 or 低."""
        if point <= 0:
            return "低"
        ratio = (ci_high - ci_low) / point
        if ratio < self.config.confidence_high_ratio:
            return "高"
        if ratio < self.config.confidence_mid_ratio:
            return "中"
        return "低"
