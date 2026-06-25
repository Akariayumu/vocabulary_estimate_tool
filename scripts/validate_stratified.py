#!/usr/bin/env python3
"""Validate the stratified v2 quiz with 3 simulated user profiles.

Users are simulated using the Rasch model itself:
  P(word known) = sigmoid(θ_user - logit(difficulty))

This creates realistic graded responses (not hard thresholds).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.stratified_quiz import StratifiedQuiz

import numpy as np


def _logit(p):
    p = max(1e-10, min(1 - 1e-10, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x):
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def simulate_rasch_user(
    name: str,
    true_theta: float,
    seed: int = 42,
) -> dict:
    """Run the full v2 pipeline for a Rasch-simulated user.

    The user's responses are generated as:
        P(known | word_j) = σ(true_theta - logit(difficulty_j))

    Args:
        name: User label.
        true_theta: The "true" ability θ used to generate responses.
        seed: Random seed for reproducibility.

    Returns:
        dict with theta estimates, vocabulary estimates, and diagnostics.
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    bank = VocabBank(DEFAULT_CONFIG)
    sq = StratifiedQuiz(bank)

    # ── Phase 1 ──────────────────────────────────────────────────────────
    phase1 = sq.phase1_sample(adaptive=True)
    responses_phase1: list[tuple[str, bool]] = []
    for item in phase1:
        d_logit = _logit(max(0.001, min(0.999, item["difficulty"])))
        p = _sigmoid(true_theta - d_logit)
        known = rng.random() < p
        responses_phase1.append((item["word"], known))

    phase1_correct = sum(r[1] for r in responses_phase1)
    phase1_rate = phase1_correct / len(phase1)

    # ── Fit θ from Phase 1 ──────────────────────────────────────────────
    theta1, ci1 = sq.fit_ability(responses_phase1)
    est1 = sq.estimate_with_ci(responses_phase1)

    # ── Phase 2 ──────────────────────────────────────────────────────────
    low_conf = sq._identify_low_confidence(responses_phase1)
    phase2 = sq.phase2_sample(theta1, low_confidence_classes=low_conf, responses=responses_phase1)
    responses_phase2: list[tuple[str, bool]] = []
    for item in phase2:
        d_logit = _logit(max(0.001, min(0.999, item["difficulty"])))
        p = _sigmoid(true_theta - d_logit)
        known = rng.random() < p
        responses_phase2.append((item["word"], known))

    phase2_correct = sum(r[1] for r in responses_phase2)

    # ── Combined fit ────────────────────────────────────────────────────
    all_responses = responses_phase1 + responses_phase2
    theta2, ci2 = sq.fit_ability(all_responses)
    est2 = sq.estimate_with_ci(all_responses)

    # ── Expected vocabulary (theoretical) ───────────────────────────────
    expected_vocab = sum(
        _sigmoid(true_theta - _logit(max(0.001, min(0.999, d))))
        for d in sq._word_difficulties.values()
    )

    # ── Per-class stats ─────────────────────────────────────────────────
    from collections import defaultdict
    class_stats: dict[int, dict] = {}
    class_counts: dict[int, list[bool]] = defaultdict(list)
    for word, known in all_responses:
        info = sq._word_to_stage.get(word)
        if info and "cluster_20" in info:
            c20 = int(info["cluster_20"])
            class_counts[c20].append(known)
    for c20, vals in sorted(class_counts.items()):
        class_stats[c20] = {
            "asked": len(vals),
            "correct": sum(vals),
            "rate": round(sum(vals) / len(vals), 3),
        }

    return {
        "user": name,
        "true_theta": true_theta,
        "theta_phase1": round(theta1, 4),
        "theta_phase2": round(theta2, 4),
        "theta_ci_phase1": [round(v, 4) for v in ci1],
        "theta_ci_phase2": [round(v, 4) for v in ci2],
        "phase1_correct": phase1_correct,
        "phase1_total": len(phase1),
        "phase1_rate": round(phase1_rate, 3),
        "phase2_correct": phase2_correct,
        "phase2_total": len(phase2),
        "low_confidence_classes": low_conf,
        "estimate_phase1": est1["point_estimate"],
        "estimate_phase2": est2["point_estimate"],
        "expected_vocab": round(expected_vocab),
        "range_phase1": est1["vocabulary_range"],
        "range_phase2": est2["vocabulary_range"],
        "confidence_phase1": est1["confidence"],
        "confidence_phase2": est2["confidence"],
        "per_class": class_stats,
    }


def main():
    print("=" * 80)
    print("  Stratified v2 Quiz — Rasch-based Simulation")
    print("=" * 80)

    # Three user profiles using Rasch θ
    # θ ≈ -2 → ~初中 level; θ ≈ 0 → ~CET-4; θ ≈ 2 → ~CET-6/proficient
    users = [
        ("初中水平  (θ=-1.5)", -1.5, 101),
        ("四级水平    (θ=0.0)", 0.0, 102),
        ("六级水平  (θ=1.5)", 1.5, 103),
    ]

    for name, true_theta, seed in users:
        print(f"\n{'─' * 80}")
        print(f"  📚 {name}")
        print(f"{'─' * 80}")
        result = simulate_rasch_user(name, true_theta, seed)

        print(f"  Phase 1 : {result['phase1_correct']}/{result['phase1_total']} correct "
              f"({result['phase1_rate']*100:.1f}%)")
        print(f"    Est. θ    : {result['theta_phase1']}  "
              f"CI [{result['theta_ci_phase1'][0]}, {result['theta_ci_phase1'][1]}]")
        print(f"    True θ    : {result['true_theta']}")
        print(f"    Vocab     : {result['estimate_phase1']}  "
              f"({result['range_phase1'][0]}–{result['range_phase1'][1]}) "
              f"[{result['confidence_phase1']}]")

        print(f"  Low-confidence classes: {result['low_confidence_classes']}")
        print(f"  Phase 2 : {result['phase2_correct']}/{result['phase2_total']} correct")

        print(f"  Combined Results:")
        print(f"    Est. θ    : {result['theta_phase2']}  "
              f"CI [{result['theta_ci_phase2'][0]}, {result['theta_ci_phase2'][1]}]")
        print(f"    Vocab     : {result['estimate_phase2']}  "
              f"({result['range_phase2'][0]}–{result['range_phase2'][1]}) "
              f"[{result['confidence_phase2']}]")
        print(f"    Expected  : {result['expected_vocab']}  "
              f"(theoretical vocab for true θ={result['true_theta']})")

        # Diagnostic
        theta_err = abs(result['theta_phase2'] - result['true_theta'])
        vocab_err = abs(result['estimate_phase2'] - result['expected_vocab'])
        print(f"    θ error   : {theta_err:.3f}  "
              f"{'✓' if theta_err < 1.0 else '⚠️ large'}")
        print(f"    Vocab err : {vocab_err}  "
              f"{'✓' if vocab_err < 3000 else '⚠️ large'}")

        # Per-class breakdown
        n_p = len([c for c, s in result["per_class"].items() if s["rate"] >= 0.8])
        n_m = len([c for c, s in result["per_class"].items() if 0.2 < s["rate"] < 0.8])
        n_u = len([c for c, s in result["per_class"].items() if s["rate"] <= 0.2])
        print(f"    Classes: {n_p} known, {n_m} mixed, {n_u} unknown (out of 20)")

    print(f"\n{'=' * 80}")
    print("  Validation Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
