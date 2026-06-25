#!/usr/bin/env python3
"""Synthetic-user simulation for the stratified Rasch vocabulary estimator.

The simulator builds users with a true Rasch ability ``theta``, generates
responses from the same ``P(known | theta)`` model used by the estimator, runs
phase 1 plus phase 2 refinement, then evaluates estimate quality against the
synthetic expected vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vocab_estimator.stratified_quiz import StratifiedQuiz


DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "simulation_results_v2.json"
DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"


@dataclass(frozen=True)
class VocabWord:
    """Minimal vocabulary-bank entry needed by the simulator."""

    word: str
    difficulty: float


@dataclass(frozen=True)
class SyntheticUser:
    """Synthetic respondent parameterized by true Rasch ability."""

    user_id: int
    true_theta: float
    true_vocab: int
    expected_vocab_raw: int
    seed: int


def sigmoid(x: float) -> float:
    """Numerically stable scalar sigmoid."""
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    """Logit transform, clamped to match the estimator's item scale."""
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))


def load_vocab_bank(stage_vocab_path: str | Path = DEFAULT_STAGE_VOCAB) -> list[VocabWord]:
    """Load words with difficulty scores from ``stage_vocab.json``."""
    path = Path(stage_vocab_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    vocab: list[VocabWord] = []
    for word, info in data["word_to_stage"].items():
        difficulty = info.get("difficulty")
        if difficulty is None:
            continue
        vocab.append(VocabWord(word=word.lower(), difficulty=float(difficulty)))

    if not vocab:
        raise ValueError(f"No difficulty-scored words found in {path}")
    return vocab


def _difficulty_logits(vocab_bank: Sequence[VocabWord]) -> np.ndarray:
    """Return estimator-scale item difficulties for a vocabulary bank."""
    return np.array(
        [_logit(max(0.001, min(0.999, item.difficulty))) for item in vocab_bank],
        dtype=float,
    )


def _expected_vocab_from_logits(theta: float, difficulty_logits: np.ndarray) -> float:
    """Expected known-word count under the Rasch response model."""
    return float(np.sum(1.0 / (1.0 + np.exp(-np.clip(theta - difficulty_logits, -40.0, 40.0)))))


def _expected_vocab(theta: float, vocab_bank: Sequence[VocabWord]) -> float:
    """Expected vocabulary from ``sum(sigmoid(theta - logit(difficulty)))``."""
    return _expected_vocab_from_logits(theta, _difficulty_logits(vocab_bank))


def _theta_for_expected_vocab(target_vocab: int, difficulty_logits: np.ndarray) -> float:
    """Find true theta whose expected vocabulary is close to ``target_vocab``."""
    max_expected = len(difficulty_logits)
    target = max(0.0, min(float(target_vocab), float(max_expected)))
    if target <= 0:
        return -50.0
    if target >= max_expected:
        return 50.0

    lo = float(np.min(difficulty_logits) - 10.0)
    hi = float(np.max(difficulty_logits) + 10.0)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        expected = _expected_vocab_from_logits(mid, difficulty_logits)
        if expected < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_synthetic_users(
    n_users: int = 2000,
    vocab_bank: Sequence[VocabWord] | None = None,
    true_min: int = 1000,
    true_max: int = 15000,
    seed: int = 42,
) -> Iterable[SyntheticUser]:
    """Generate synthetic users with theta-derived expected vocabularies."""
    if n_users < 0:
        raise ValueError("n_users must be non-negative")

    bank = list(vocab_bank) if vocab_bank is not None else load_vocab_bank()
    if not bank:
        raise ValueError("vocab_bank must not be empty")

    difficulty_logits = _difficulty_logits(bank)
    effective_max = min(int(true_max), len(difficulty_logits))
    effective_min = max(0, min(int(true_min), effective_max))
    py_rng = random.Random(seed)

    for user_id in range(n_users):
        target_vocab = py_rng.randint(effective_min, effective_max)
        true_theta = _theta_for_expected_vocab(target_vocab, difficulty_logits)
        expected_vocab = int(round(_expected_vocab_from_logits(true_theta, difficulty_logits)))
        user_seed = py_rng.randrange(0, 2**32)
        yield SyntheticUser(
            user_id=user_id,
            true_theta=true_theta,
            true_vocab=expected_vocab,
            expected_vocab_raw=expected_vocab,
            seed=user_seed,
        )


def simulate_quiz(
    user: SyntheticUser,
    quiz_sampler: StratifiedQuiz,
    *,
    adaptive: bool = True,
    phase2_n_per_class: int = 8,
) -> dict[str, Any]:
    """Run phase 1 + phase 2 and generate Rasch-model responses."""
    sample_rng = random.Random(user.seed)
    response_rng = random.Random(user.seed ^ 0x9E3779B9)

    def answer(item: dict[str, Any]) -> tuple[str, bool]:
        d_logit = _logit(max(0.001, min(0.999, float(item["difficulty"]))))
        p_known = sigmoid(user.true_theta - d_logit)
        return item["word"], response_rng.random() < p_known

    phase1_items = quiz_sampler.phase1_sample(adaptive=adaptive, rng=sample_rng)
    phase1_responses = [answer(item) for item in phase1_items]

    theta1, _ = quiz_sampler.fit_ability(phase1_responses)
    low_confidence_classes = quiz_sampler._identify_low_confidence(phase1_responses)
    phase1_words = {word.lower() for word, _ in phase1_responses}
    phase2_items = quiz_sampler.phase2_sample(
        theta1,
        low_confidence_classes=low_confidence_classes,
        responses=phase1_responses,
        n_per_class=phase2_n_per_class,
        exclude=phase1_words,
    )
    phase2_responses = [answer(item) for item in phase2_items]

    responses = phase1_responses + phase2_responses
    return {
        "responses": responses,
        "phase1_total": len(phase1_responses),
        "phase1_correct": int(sum(known for _, known in phase1_responses)),
        "phase2_total": len(phase2_responses),
        "phase2_correct": int(sum(known for _, known in phase2_responses)),
        "low_confidence_classes": low_confidence_classes,
    }


def _bucket_name(true_vocab: int) -> str:
    if 1000 <= true_vocab < 3000:
        return "low_1k_3k"
    if 3000 <= true_vocab < 8000:
        return "mid_3k_8k"
    if 8000 <= true_vocab <= 15000:
        return "high_8k_15k"
    return "out_of_range"


def _metrics(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "correlation": 0.0,
            "r2": 0.0,
            "mean_bias": 0.0,
        }

    true_vals = np.array([r["true_vocab"] for r in records], dtype=float)
    est_vals = np.array([r["estimated_vocab"] for r in records], dtype=float)
    errors = est_vals - true_vals

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_bias = float(np.mean(errors))

    if len(records) < 2 or np.std(true_vals) == 0 or np.std(est_vals) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(true_vals, est_vals)[0, 1])

    sst = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    sse = float(np.sum(errors**2))
    r2 = 0.0 if sst == 0 else 1.0 - sse / sst

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "correlation": round(corr, 6),
        "r2": round(float(r2), 6),
        "mean_bias": round(mean_bias, 3),
    }


def _bucket_errors(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets = ["low_1k_3k", "mid_3k_8k", "high_8k_15k", "out_of_range"]
    result: list[dict[str, Any]] = []
    for bucket in buckets:
        bucket_records = [r for r in records if r["bucket"] == bucket]
        if not bucket_records:
            continue
        metrics = _metrics(bucket_records)
        result.append(
            {
                "bucket": bucket,
                "n": len(bucket_records),
                **metrics,
            }
        )
    return result


def _histogram(values: Sequence[int], bin_edges: Sequence[int]) -> dict[str, Any]:
    counts, edges = np.histogram(np.array(values, dtype=float), bins=np.array(bin_edges, dtype=float))
    return {
        "bin_edges": [int(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
    }


def _distribution_data(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_seen = max(
        [15000]
        + [int(r["true_vocab"]) for r in records]
        + [int(r["estimated_vocab"]) for r in records]
    )
    upper = int(math.ceil(max_seen / 1000.0) * 1000)
    bin_edges = list(range(0, upper + 1000, 1000))

    return {
        "histogram_bins": bin_edges,
        "true_vocab_histogram": _histogram([r["true_vocab"] for r in records], bin_edges),
        "estimated_vocab_histogram": _histogram([r["estimated_vocab"] for r in records], bin_edges),
        "scatter": [
            {
                "true_vocab": r["true_vocab"],
                "estimated_vocab": r["estimated_vocab"],
                "bias": r["bias"],
            }
            for r in records
        ],
    }


def run_evaluation(
    n_users: int = 2000,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    seed: int = 42,
    true_min: int = 1000,
    true_max: int = 15000,
    stage_vocab_path: str | Path = DEFAULT_STAGE_VOCAB,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full simulation and write JSON results."""
    vocab_bank = load_vocab_bank(stage_vocab_path)
    quiz = StratifiedQuiz(stage_vocab_path=stage_vocab_path)

    records: list[dict[str, Any]] = []
    users = generate_synthetic_users(
        n_users=n_users,
        vocab_bank=vocab_bank,
        true_min=true_min,
        true_max=true_max,
        seed=seed,
    )
    difficulty_logits = _difficulty_logits(vocab_bank)

    for idx, user in enumerate(users, start=1):
        simulation = simulate_quiz(user, quiz)
        responses = simulation["responses"]
        theta_hat, theta_ci = quiz.fit_ability(responses)
        estimate = quiz.estimate_with_ci(responses)
        estimated_vocab = int(round(_expected_vocab_from_logits(theta_hat, difficulty_logits)))
        correct_count = int(sum(known for _, known in responses))
        total_questions = int(len(responses))
        bias = int(estimated_vocab - user.true_vocab)

        records.append(
            {
                "user_id": user.user_id,
                "true_theta": round(user.true_theta, 6),
                "true_vocab": int(user.true_vocab),
                "expected_vocab_raw": int(user.expected_vocab_raw),
                "estimated_vocab": estimated_vocab,
                "point_estimate": int(estimate["point_estimate"]),
                "raw_vocab_estimate": int(estimate["raw_vocab_estimate"]),
                "correct_count": correct_count,
                "total_questions": total_questions,
                "phase1_correct": simulation["phase1_correct"],
                "phase1_total": simulation["phase1_total"],
                "phase2_correct": simulation["phase2_correct"],
                "phase2_total": simulation["phase2_total"],
                "low_confidence_classes": simulation["low_confidence_classes"],
                "bias": bias,
                "bucket": _bucket_name(user.true_vocab),
                "theta": round(theta_hat, 4),
                "theta_ci_95": [round(v, 4) for v in theta_ci],
                "api_theta": estimate["theta"],
                "vocabulary_range": estimate["vocabulary_range"],
                "confidence": estimate["confidence"],
            }
        )

        if not quiet and (idx == n_users or idx % 100 == 0):
            print(f"simulated {idx}/{n_users} users", file=sys.stderr)

    summary = {
        **_metrics(records),
        "n_users": len(records),
        "vocab_bank_size": len(vocab_bank),
        "true_vocab_min": min((r["true_vocab"] for r in records), default=0),
        "true_vocab_max": max((r["true_vocab"] for r in records), default=0),
        "mean_phase1_questions": round(float(np.mean([r["phase1_total"] for r in records])), 3) if records else 0.0,
        "mean_phase2_questions": round(float(np.mean([r["phase2_total"] for r in records])), 3) if records else 0.0,
        "bucket_errors": _bucket_errors(records),
    }
    result = {
        "metadata": {
            "simulation_model": "rasch_theta",
            "response_probability": "sigmoid(true_theta - logit(difficulty))",
            "estimated_vocab_metric": "sum(sigmoid(theta_hat - logit(difficulty)))",
            "phase2_n_per_low_confidence_class": 8,
        },
        "summary": summary,
        "records": records,
        "distribution_data": _distribution_data(records),
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if not quiet:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"wrote {output_path}")

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run synthetic evaluation for the stratified Rasch vocabulary estimator."
    )
    parser.add_argument("--n-users", type=int, default=2000, help="Number of synthetic users.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--true-min", type=int, default=1000, help="Minimum true vocabulary size.")
    parser.add_argument("--true-max", type=int, default=15000, help="Maximum true vocabulary size.")
    parser.add_argument(
        "--stage-vocab",
        type=Path,
        default=DEFAULT_STAGE_VOCAB,
        help="Path to data/stage_vocab.json.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress and summary prints.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_evaluation(
        n_users=args.n_users,
        output=args.output,
        seed=args.seed,
        true_min=args.true_min,
        true_max=args.true_max,
        stage_vocab_path=args.stage_vocab,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
