#!/usr/bin/env python3
"""Compare hybrid bisection + CAT vs existing stratified quiz.

Two schemes on the same synthetic users:
  A) Hybrid: 6× quantile bisection + 34× Fisher-information CAT
  B) Control: existing 40× stratified + ~8× per low-conf class refine
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vocab_estimator.stratified_quiz import StratifiedQuiz

# ── Reuse from tests/simulation_eval.py ──────────────────────────────────
from tests.simulation_eval import (
    VocabWord,
    SyntheticUser,
    sigmoid,
    _logit,
    load_vocab_bank,
    _difficulty_logits,
    _expected_vocab_from_logits,
    _theta_for_expected_vocab,
    generate_synthetic_users,
    _bucket_name,
    _metrics,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "hybrid_bisection_validation.json"
DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"


# ── Quantile anchors (derived from stage_vocab.json) ────────────────────
# difficulty distribution is strongly right-skewed (mean 0.81, median 0.857)
# Binary search operates on quantile indices, not raw difficulty
QUANTILE_LABELS = ["P5", "P10", "P25", "P50", "P75", "P90", "P95"]
QUANTILE_DIFFS = [0.4323, 0.5648, 0.7378, 0.8572, 0.9447, 0.9753, 0.9873]


def _logit_diffs(arr: list[float]) -> np.ndarray:
    return np.array([_logit(max(0.001, min(0.999, d))) for d in arr], dtype=float)


QUANTILE_LOGITS = _logit_diffs(QUANTILE_DIFFS)


def _nearest_word(items: list[dict], target_diff: float, exclude: set[str]) -> dict | None:
    """Pick word with difficulty closest to target, excluding already-seen words."""
    best = None
    best_err = float("inf")
    for item in items:
        w = item["word"]
        if w in exclude:
            continue
        err = abs(item["difficulty"] - target_diff)
        if err < best_err:
            best_err = err
            best = item
    return best


# ── Scheme A: Hybrid bisection + CAT ─────────────────────────────────────

def run_hybrid_scheme(
    user: SyntheticUser,
    quiz: StratifiedQuiz,
    response_rng: random.Random,
    all_candidates: list[dict],
    total_questions: int = 40,
    bisection_steps: int = 6,
) -> tuple[list[tuple[str, bool]], int, int]:
    """Run scheme A and return (responses, phase1_correct, phase1_total)."""
    responses: list[tuple[str, bool]] = []
    seen: set[str] = set()
    response_rng = random.Random(response_rng.randrange(0, 2**32))

    def answer_item(item: dict) -> bool:
        d_logit = _logit(max(0.001, min(0.999, float(item["difficulty"]))))
        p_known = sigmoid(user.true_theta - d_logit)
        return response_rng.random() < p_known

    # ── Phase 1: quantile bisection (bisection_steps questions) ────────
    lo_idx = 0   # P5 index
    hi_idx = 6   # P95 index
    phase1_correct = 0

    for step in range(bisection_steps):
        mid_idx = (lo_idx + hi_idx) // 2
        target_diff = QUANTILE_DIFFS[mid_idx]
        item = _nearest_word(all_candidates, target_diff, seen)
        if item is None:
            # Fallback: pick any unseen word near mid
            remaining = [c for c in all_candidates if c["word"] not in seen]
            if not remaining:
                break
            remaining.sort(key=lambda x: abs(x["difficulty"] - target_diff))
            item = remaining[0]

        seen.add(item["word"])
        known = answer_item(item)
        responses.append((item["word"], known))
        if known:
            phase1_correct += 1
            lo_idx = mid_idx
        else:
            hi_idx = mid_idx

    phase1_total = len(responses)

    # ── Phase 2: Fisher-information CAT ─────────────────────────────────
    n_remaining = total_questions - bisection_steps
    theta, _ = quiz.fit_ability(responses)

    for _ in range(n_remaining):
        # Score all candidates by Fisher info at current θ
        best_item = None
        best_info = -1.0
        for c in all_candidates:
            w = c["word"]
            if w in seen:
                continue
            d_logit = _logit(max(0.001, min(0.999, float(c["difficulty"]))))
            info_val = quiz._item_information(theta, d_logit)
            if info_val > best_info:
                best_info = info_val
                best_item = c

        if best_item is None:
            break  # ran out of words

        seen.add(best_item["word"])
        known = answer_item(best_item)
        responses.append((best_item["word"], known))
        theta, _ = quiz.fit_ability(responses)

    return responses, phase1_correct, phase1_total


# ── Scheme B: Existing stratified quiz pipeline ──────────────────────────

def run_control_scheme(
    user: SyntheticUser,
    quiz: StratifiedQuiz,
) -> tuple[list[tuple[str, bool]], int, int]:
    """Run the existing stratified quiz pipeline."""
    sample_rng = random.Random(user.seed)
    response_rng = random.Random(user.seed ^ 0x9E3779B9)

    def answer(item: dict) -> tuple[str, bool]:
        d_logit = _logit(max(0.001, min(0.999, float(item["difficulty"]))))
        p_known = sigmoid(user.true_theta - d_logit)
        return item["word"], response_rng.random() < p_known

    phase1_items = quiz.phase1_sample(adaptive=True, rng=sample_rng)
    phase1_responses = [answer(item) for item in phase1_items]
    phase1_correct = int(sum(known for _, known in phase1_responses))

    theta1, _ = quiz.fit_ability(phase1_responses)
    low_confidence = quiz._identify_low_confidence(phase1_responses)
    phase1_words = {word.lower() for word, _ in phase1_responses}
    phase2_items = quiz.phase2_sample(
        theta1,
        low_confidence_classes=low_confidence,
        responses=phase1_responses,
        n_per_class=8,
        exclude=phase1_words,
    )
    phase2_responses = [answer(item) for item in phase2_items]

    responses = phase1_responses + phase2_responses
    return responses, phase1_correct, len(phase1_responses)


# ── Main evaluation ──────────────────────────────────────────────────────

def run_evaluation(
    n_users: int = 500,
    *,
    seed: int = 42,
    output: str | Path = DEFAULT_OUTPUT,
    true_min: int = 1000,
    true_max: int = 15000,
    quiet: bool = False,
) -> dict[str, Any]:
    t0 = time.time()

    # Load data
    vocab_bank = load_vocab_bank()
    all_candidates: list[dict] = []
    wts = json.load(open(DEFAULT_STAGE_VOCAB, encoding="utf-8"))["word_to_stage"]
    for word, info in wts.items():
        d = info.get("difficulty")
        c20 = info.get("cluster_20")
        c100 = info.get("cluster_100")
        if d is not None and c20 is not None and c100 is not None:
            all_candidates.append({
                "word": word, "difficulty": float(d),
                "cluster_20": int(c20), "cluster_100": int(c100),
            })
    if not quiet:
        print(f"Loaded {len(all_candidates)} stage-vocab candidates", file=sys.stderr)

    quiz = StratifiedQuiz(stage_vocab_path=str(DEFAULT_STAGE_VOCAB))
    difficulty_logits = _difficulty_logits(vocab_bank)

    users = list(generate_synthetic_users(
        n_users=n_users, vocab_bank=vocab_bank,
        true_min=true_min, true_max=true_max, seed=seed,
    ))

    records_a: list[dict] = []
    records_b: list[dict] = []

    for idx, user in enumerate(users, start=1):
        # ── Scheme A ────────────────────────────────────────────────────
        response_rng_a = random.Random(user.seed ^ 0xDEADBEEF)
        resp_a, p1corr_a, p1tot_a = run_hybrid_scheme(
            user, quiz, response_rng_a, all_candidates,
            total_questions=40, bisection_steps=6,
        )
        theta_a, ci_a = quiz.fit_ability(resp_a)
        est_a = int(round(_expected_vocab_from_logits(theta_a, difficulty_logits)))

        # Phase 1 only estimate
        resp_a_p1 = resp_a[:6]
        theta_a_p1, _ = quiz.fit_ability(resp_a_p1)
        est_a_p1 = int(round(_expected_vocab_from_logits(theta_a_p1, difficulty_logits)))

        records_a.append({
            "user_id": user.user_id,
            "true_theta": user.true_theta,
            "true_vocab": user.true_vocab,
            "estimated_vocab": est_a,
            "estimated_vocab_p1": est_a_p1,
            "theta": round(theta_a, 4),
            "theta_ci": [round(v, 4) for v in ci_a],
            "correct_total": int(sum(known for _, known in resp_a)),
            "total_questions": len(resp_a),
            "phase1_correct": p1corr_a,
            "phase1_total": p1tot_a,
            "bias": est_a - user.true_vocab,
            "bucket": _bucket_name(user.true_vocab),
        })

        # ── Scheme B ────────────────────────────────────────────────────
        resp_b, p1corr_b, p1tot_b = run_control_scheme(user, quiz)
        theta_b, ci_b = quiz.fit_ability(resp_b)
        est_b = int(round(_expected_vocab_from_logits(theta_b, difficulty_logits)))

        # Phase 1 only
        resp_b_p1_prepared = quiz._prepare_responses(resp_b[:p1tot_b])
        if len(resp_b_p1_prepared) >= 3:
            theta_b_p1, _ = quiz.fit_ability(resp_b[:p1tot_b])
        else:
            theta_b_p1 = theta_b
        est_b_p1 = int(round(_expected_vocab_from_logits(theta_b_p1, difficulty_logits)))

        records_b.append({
            "user_id": user.user_id,
            "true_theta": user.true_theta,
            "true_vocab": user.true_vocab,
            "estimated_vocab": est_b,
            "estimated_vocab_p1": est_b_p1,
            "theta": round(theta_b, 4),
            "theta_ci": [round(v, 4) for v in ci_b],
            "correct_total": int(sum(known for _, known in resp_b)),
            "total_questions": len(resp_b),
            "phase1_correct": p1corr_b,
            "phase1_total": p1tot_b,
            "bias": est_b - user.true_vocab,
            "bucket": _bucket_name(user.true_vocab),
        })

        if not quiet and (idx % 100 == 0 or idx == n_users):
            elapsed = time.time() - t0
            print(f"  [{idx}/{n_users}] {elapsed:.0f}s elapsed", file=sys.stderr)

    # ── Compute metrics ─────────────────────────────────────────────────
    def bucket_results(records):
        buckets = ["low_1k_3k", "mid_3k_8k", "high_8k_15k"]
        out = []
        for b in buckets:
            rs = [r for r in records if r["bucket"] == b]
            if not rs:
                continue
            m = _metrics(rs)
            out.append({**m, "bucket": b, "n": len(rs)})
        return out

    def scheme_metrics(records, key="estimated_vocab"):
        true_vals = np.array([r["true_vocab"] for r in records], dtype=float)
        est_vals = np.array([r[key] for r in records], dtype=float)
        return _metrics(records)

    summary = {
        "n_users": n_users,
        "seed": seed,
        "vocab_bank_size": len(vocab_bank),
        "scheme_a": {
            "name": "Hybrid bisection + CAT (6 bisection + 34 FI-CAT)",
            "overall": scheme_metrics(records_a),
            "overall_p1": scheme_metrics(records_a, "estimated_vocab_p1"),
            "bucket_errors": bucket_results(records_a),
            "mean_total_questions": np.mean([r["total_questions"] for r in records_a]).round(1),
            "mean_phase1_questions": np.mean([r["phase1_total"] for r in records_a]).round(1),
        },
        "scheme_b": {
            "name": "Existing stratified quiz (40 stratified + refine)",
            "overall": scheme_metrics(records_b),
            "overall_p1": scheme_metrics(records_b, "estimated_vocab_p1"),
            "bucket_errors": bucket_results(records_b),
            "mean_total_questions": np.mean([r["total_questions"] for r in records_b]).round(1),
            "mean_phase1_questions": np.mean([r["phase1_total"] for r in records_b]).round(1),
        },
    }

    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Write output
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "records_a": records_a,
            "records_b": records_b,
        }, f, ensure_ascii=False, indent=2)

    return summary


def print_comparison(summary: dict):
    """Pretty-print the comparison."""
    sa = summary["scheme_a"]
    sb = summary["scheme_b"]

    print("\n" + "=" * 76)
    print("  Hybrid Bisection + CAT  vs  Existing Stratified Quiz")
    print("=" * 76)
    print(f"  N={summary['n_users']} synthetic users, seed={summary['seed']}")
    print(f"  Elapsed: {summary['elapsed_seconds']}s")
    print()

    headers = ["Metric", "Scheme A\n(bisect+CAT)", "Scheme B\n(stratified)", "Δ"]
    rows = [
        ["MAE", str(sa["overall"]["mae"]), str(sb["overall"]["mae"]),
         str(round(sa["overall"]["mae"] - sb["overall"]["mae"], 1))],
        ["RMSE", str(sa["overall"]["rmse"]), str(sb["overall"]["rmse"]),
         str(round(sa["overall"]["rmse"] - sb["overall"]["rmse"], 1))],
        ["R²", str(sa["overall"]["r2"]), str(sb["overall"]["r2"]),
         str(round(sa["overall"]["r2"] - sb["overall"]["r2"], 4))],
        ["Corr", str(sa["overall"]["correlation"]), str(sb["overall"]["correlation"]),
         str(round(sa["overall"]["correlation"] - sb["overall"]["correlation"], 4))],
        ["Mean Bias", str(sa["overall"]["mean_bias"]), str(sb["overall"]["mean_bias"]),
         str(round(sa["overall"]["mean_bias"] - sb["overall"]["mean_bias"], 1))],
    ]

    col_widths = [18, 14, 14, 12]
    def fmt_row(cells):
        return "  " + "  ".join(c.rjust(w) for c, w in zip(cells, col_widths))

    print(fmt_row(headers))
    print("  " + "-" * 60)
    for row in rows:
        print(fmt_row(row))

    print()
    print("  Bucket errors (MAE):")
    print("  " + f"{'Bucket':>14s}  {'A (MAE)':>8s}  {'B (MAE)':>8s}  {'n':>5s}")
    all_buckets = set()
    for b in sa["bucket_errors"]:
        all_buckets.add(b["bucket"])
    for b in sb["bucket_errors"]:
        all_buckets.add(b["bucket"])
    for bucket in sorted(all_buckets):
        a_b = next((x for x in sa["bucket_errors"] if x["bucket"] == bucket), None)
        b_b = next((x for x in sb["bucket_errors"] if x["bucket"] == bucket), None)
        a_mae = str(a_b["mae"]) if a_b else "—"
        b_mae = str(b_b["mae"]) if b_b else "—"
        n = a_b["n"] if a_b else (b_b["n"] if b_b else 0)
        print(f"  {bucket:>14s}  {a_mae:>8s}  {b_mae:>8s}  {n:>5d}")

    print()
    print("  Phase-1-only comparison (Δ = A - B):")
    print(f"  {'Metric':>14s}  {'A (MAE)':>10s}  {'B (MAE)':>10s}  {'Δ':>10s}")
    p1_a = sa["overall_p1"]["mae"]
    p1_b = sb["overall_p1"]["mae"]
    print(f"  {'MAE':>14s}  {p1_a:>10.1f}  {p1_b:>10.1f}  {p1_a - p1_b:>+10.1f}")
    p1_a_r2 = sa["overall_p1"]["r2"]
    p1_b_r2 = sb["overall_p1"]["r2"]
    print(f"  {'R²':>14s}  {p1_a_r2:>10.4f}  {p1_b_r2:>10.4f}  {p1_a_r2 - p1_b_r2:>+10.4f}")

    # Conclusion
    print("\n" + "-" * 76)
    a_better = sa["overall"]["r2"] > sb["overall"]["r2"]
    r2_delta = abs(sa["overall"]["r2"] - sb["overall"]["r2"])
    mae_delta = abs(sa["overall"]["mae"] - sb["overall"]["mae"])

    if r2_delta < 0.005:
        print(f"  🏁 两方案在最终精度上无显著差异 (R²差={r2_delta:.4f})")
    elif a_better:
        print(f"  🏆 Hybrid方案在R²上领先{r2_delta:.4f}")
    else:
        print(f"  🏆 现有方案在R²上领先{r2_delta:.4f}")

    print(f"  📊 A Phase1(6题) MAE={p1_a} vs B Phase1(40题) MAE={p1_b}")
    if p1_a < p1_b * 0.8:
        print(f"  ✅ Hybrid的6题二分定位效率很高：仅6题就达到现有40题∼{int(100*p1_a/p1_b)}%的MAE水平")
    elif p1_a < p1_b:
        print(f"  ✅ Hybrid的6题二分定位效率更好：6题 vs 40题")
    else:
        print(f"  ⚠️  Hybrid的6题二分定位不如现有40题的Phase 1精度")

    print(f"  📐 方案A总题量: ~{sa['mean_total_questions']} 题 | 方案B总题量: ~{sb['mean_total_questions']} 题")
    print("-" * 76)


def main():
    summary = run_evaluation(n_users=500, quiet=False)
    print_comparison(summary)
    print(f"\nResults written to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
