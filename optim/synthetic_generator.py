"""Synthetic test-response generator for calibration pipeline validation.

Generates a population of synthetic test-takers with known (α, β) parameters,
then renders binary responses on a stratified word list.  The resulting dataset
can be fed to ``calibration_trainer`` to verify that the original global
parameters (β, k, piecewise_knots) can be recovered from the data.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.vocab_model import VocabEstimator


# ---------------------------------------------------------------------------
# Known "ground truth" parameters
# ---------------------------------------------------------------------------

TRUE_BETA = -0.285
TRUE_K = 0.0000792
TRUE_KNOTS = [(3000, 1.0), (8000, 0.42), (22000, 1.35)]
TRUE_MAX_V = 20000.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def piecewise_calibrate(x: float, knots: list[tuple[int, float]]) -> float:
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


def calibrate_vocab(raw: float) -> float:
    """Apply TRUE calibration parameters."""
    cal = TRUE_MAX_V * math.tanh(TRUE_K * raw)
    return piecewise_calibrate(cal, TRUE_KNOTS)


# ---------------------------------------------------------------------------
# Synthetic user generation
# ---------------------------------------------------------------------------

def generate_user(
    user_id: int,
    alpha: float,
    bank: VocabBank,
    sampler: VocabularySampler,
    questions_per_user: int = 80,
    noise: float = 0.02,
    rng: random.Random | None = None,
) -> list[tuple[str, bool]]:
    """Generate a synthetic user's responses.

    Args:
        alpha: Per-user intercept (ability parameter).
        bank: Vocabulary bank.
        sampler: For stratified word selection.
        questions_per_user: Number of test questions for this user.
        noise: Label-flip probability (simulates careless mistakes).
        rng: Random state for reproducibility.

    Returns:
        [(word, known), ...]
    """
    if rng is None:
        rng = random.Random(42 + user_id)

    # Sample a stratified word list for this user
    items = sampler.balanced_sample(per_bucket=4)
    # Get more words if needed
    while len(items) < questions_per_user:
        extra = sampler.balanced_sample(per_bucket=2)
        items.extend(extra)

    # Shuffle and trim
    rng.shuffle(items)
    items = items[:questions_per_user]

    responses: list[tuple[str, bool]] = []
    for word, rank, bucket in items:
        log_rank = math.log(max(rank, 1))
        logit = alpha + TRUE_BETA * log_rank
        prob = 1.0 / (1.0 + math.exp(-logit))
        # Apply noise
        prob = (1.0 - noise) * prob + noise * 0.5
        known = rng.random() < prob
        responses.append((word, known))

    return responses


def generate_population(
    bank: VocabBank,
    n_users: int = 100,
    questions_per_user: int = 80,
    seed: int = 42,
) -> dict[int, list[tuple[str, bool]]]:
    """Generate a population of synthetic test-takers.

    α values are evenly spaced from -5 (low ability, ~1500 vocab)
    to +3 (high ability, ~15000 vocab).  With TRUE_BETA = -0.285,
    this translates to raw estimates from ~2500 to ~21000.

    Distribution:
        20% low ability    (α ∈ [-5.0, -3.0])
        40% medium ability (α ∈ [-3.0,  0.0])
        30% high ability   (α ∈ [ 0.0,  2.0])
        10% very high      (α ∈ [ 2.0,  3.5])
    """
    rng = random.Random(seed)
    sampler = VocabularySampler(bank, DEFAULT_CONFIG, seed=seed)

    population: dict[int, list[tuple[str, bool]]] = {}

    # Stratified α generation
    alpha_ranges = [
        (-5.0, -3.0, int(n_users * 0.20)),
        (-3.0, 0.0, int(n_users * 0.40)),
        (0.0, 2.0, int(n_users * 0.30)),
        (2.0, 3.5, n_users - int(n_users * 0.90)),
    ]

    uid = 0
    for low, high, count in alpha_ranges:
        for _ in range(count):
            if uid >= n_users:
                break
            alpha = low + (high - low) * rng.random()
            responses = generate_user(
                uid, alpha, bank, sampler,
                questions_per_user=questions_per_user,
                noise=0.02,
                rng=random.Random(seed + uid),
            )
            population[uid] = responses
            uid += 1

    return population


# ---------------------------------------------------------------------------
# Validation runner
# ---------------------------------------------------------------------------

def synthetic_validation(
    bank: VocabBank,
    bucket_labels: list[str],
    n_users: int = 50,
) -> dict[str, Any]:
    """Run synthetic validation: generate, train, check recovery accuracy.

    Returns summary metrics including the recovered parameters.
    """
    from .calibration_trainer import train_torch, train_numpy, load_responses

    print(f"Generating {n_users} synthetic users...")
    population = generate_population(bank, n_users=n_users, questions_per_user=80)

    # Save to a temp file, then load back (clean API boundary)
    tmp_path = Path("/tmp/synthetic_calibration_data.json")
    json_data = {
        "users": [
            {"user_id": uid, "responses": [{"word": w, "known": k} for w, k in resp]}
            for uid, resp in population.items()
        ]
    }
    tmp_path.write_text(json.dumps(json_data, ensure_ascii=False))

    print("Training...")
    params = train_numpy(load_responses(str(tmp_path)), bank, bucket_labels,
                         n_epochs=500, lr=0.001)

    print("\n=== Validation Results ===")
    print(f"  True β:         {TRUE_BETA:.6f}")
    print(f"  Recovered β:    {params.beta:.6f}")
    print(f"  β error:        {abs(params.beta - TRUE_BETA):.6f}")
    print(f"  True k:         {TRUE_K:.8f}")
    print(f"  Recovered k:    {params.calibration_k:.8f}")
    print(f"  k error:        {abs(params.calibration_k - TRUE_K):.8f}")
    print(f"  True knots:     {TRUE_KNOTS}")
    print(f"  Recovered knots:{params.piecewise_knots}")
    print(f"  Training loss:  {params.training_loss:.6f}")

    beta_ok = abs(params.beta - TRUE_BETA) < 0.05
    k_ok = abs(params.calibration_k - TRUE_K) < 1e-5
    slope_errors = [
        abs(params.piecewise_knots[i][1] - TRUE_KNOTS[i][1]) < 0.1
        for i in range(len(TRUE_KNOTS))
    ]
    knots_ok = all(slope_errors)

    print(f"\n  β recovery:  {'✓' if beta_ok else '✗'}")
    print(f"  k recovery:  {'✓' if k_ok else '✗'}")
    print(f"  knots slope: {'✓' if knots_ok else '✗'}")
    print(f"  OVERALL:     {'✓ PASS' if all([beta_ok, k_ok, knots_ok]) else '✗ FAIL'}")

    tmp_path.unlink(missing_ok=True)

    return {
        "true_beta": TRUE_BETA,
        "recovered_beta": params.beta,
        "true_k": TRUE_K,
        "recovered_k": params.calibration_k,
        "true_knots": TRUE_KNOTS,
        "recovered_knots": params.piecewise_knots,
        "training_loss": params.training_loss,
        "beta_ok": beta_ok,
        "k_ok": k_ok,
        "knots_ok": knots_ok,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic calibration data.")
    parser.add_argument("--n-users", type=int, default=100, help="Number of synthetic users")
    parser.add_argument("--output", default="synthetic_dataset.json", help="Output JSON path")
    parser.add_argument("--validate", action="store_true", help="Run train-and-recover validation")
    args = parser.parse_args()

    bank = VocabBank(DEFAULT_CONFIG)

    if args.validate:
        bucket_labels = list(bank.words_by_bucket.keys())
        synthetic_validation(bank, bucket_labels, n_users=args.n_users)
        return

    population = generate_population(bank, n_users=args.n_users, questions_per_user=80)
    json_data = {
        "dataset_meta": {
            "synthetic": True,
            "n_users": args.n_users,
            "true_beta": TRUE_BETA,
            "true_k": TRUE_K,
            "true_knots": [(b, s) for b, s in TRUE_KNOTS],
        },
        "users": [
            {"user_id": uid, "responses": [{"word": w, "known": k} for w, k in resp]}
            for uid, resp in population.items()
        ],
    }
    Path(args.output).write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
    print(f"Generated {args.n_users} synthetic users → {args.output}")


# ---------------------------------------------------------------------------
# NEW: Synthetic training data generator (frequency-weighted sampling)
# ---------------------------------------------------------------------------


def build_known_vocab(
    bank: VocabBank,
    vocab_size: int,
) -> set[str]:
    """Build a "known word set" for a virtual test-taker with a given ``vocab_size``.

    Words are accumulated from high-frequency buckets first. When a bucket is
    partially needed, a random subset of its words is taken.

    Args:
        bank: The vocabulary bank.
        vocab_size: Target vocabulary size for this virtual user.

    Returns:
        Set of known words.
    """
    bucket_labels_in_order = [
        "1k", "2k", "3k", "5k", "8k", "10k", "15k", "20k", "30k",
    ]
    known: set[str] = set()
    remaining = vocab_size

    for label in bucket_labels_in_order:
        items = list(bank.words_by_bucket.get(label, []))
        if not items:
            continue
        if remaining >= len(items):
            # Take the whole bucket
            for item in items:
                known.add(item.word)
            remaining -= len(items)
        else:
            # Take a random subset. Use a deterministic seed derived from the
            # target vocab_size so subsets are reproducible yet decorrelated
            # from the test-question sampling rng.
            inner_rng = random.Random(vocab_size * 12345)
            chosen = inner_rng.sample(items, remaining)
            for item in chosen:
                known.add(item.word)
            remaining = 0
            break

    return known


SYNTHETIC_VOCAB_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                         10000, 12000, 15000]


def generate_synthetic_data(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions: int = 100,
    power: float = 0.5,
    seed: int = 42,
) -> dict[int, dict]:
    """Generate synthetic training data for calibration.

    For each ``vocab_size``:
      1. Build a "known word set" (fill from high-frequency buckets up to V)
      2. Sample N test questions using frequency-weighted distribution
         (high-frequency words appear more often, like real exams)
      3. Label each question as known=True/False based on the known set

    Args:
        bank: Vocabulary bank.
        vocab_sizes: List of vocabulary sizes to simulate. Defaults to
                     ``SYNTHETIC_VOCAB_SIZES``.
        n_questions: Number of test questions per synthetic user.
        power: Power-law exponent for test question sampling.
        seed: Random seed.

    Returns:
        {vocab_size: {"responses": [(word, known), ...],
                      "known_set_size": int,
                      "known_rate": float}, ...}
    """
    if vocab_sizes is None:
        vocab_sizes = list(SYNTHETIC_VOCAB_SIZES)

    result: dict[int, dict] = {}

    for vsize in sorted(vocab_sizes):
        # Step 1: Build known word set
        known_words = build_known_vocab(bank, vsize)

        # Step 2: Sample test questions (frequency-weighted)
        from .interval_sampler import sample_test_questions
        sampled = sample_test_questions(bank, n=n_questions, power=power, seed=seed)

        # Step 3: Label each question
        responses: list[tuple[str, bool]] = []
        for sw in sampled:
            known = sw.word in known_words
            responses.append((sw.word, known))

        known_count = sum(1 for _, k in responses if k)
        known_rate = known_count / len(responses) if responses else 0.0

        result[vsize] = {
            "responses": responses,
            "known_set_size": len(known_words),
            "known_rate": known_rate,
            "n_questions": len(responses),
        }

    return result


def describe_synthetic_dataset(
    data: dict[int, dict],
) -> str:
    """Return a human-readable report of a synthetic dataset."""
    lines = []
    lines.append(f"Synthetic training data ({len(data)} vocabulary sizes)\n")
    lines.append(f"  {'Vocab':>7s}  {'KnownSet':>9s}  {'Q':>4s}  {'Known':>7s}  {'Known%':>7s}")
    lines.append(f"  {'------':>7s}  {'--------':>9s}  {'---':>4s}  {'------':>7s}  {'-------':>7s}")
    for vsize in sorted(data):
        d = data[vsize]
        lines.append(
            f"  {vsize:>7d}  {d['known_set_size']:>9d}"
            f"  {d['n_questions']:>4d}  {d['known_rate']*d['n_questions']:>5.0f}"
            f"  {d['known_rate']:>6.1%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Synthetic calibration training loop
# ---------------------------------------------------------------------------


def run_synthetic_training(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions: int = 100,
    power: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Run a synthetic training validation.

    For each ``vocab_size``, generates synthetic test-taker responses and
    feeds them to the ``VocabEstimator`` to get a prediction, then reports
    prediction vs actual vocab size.

    Returns:
        List of dicts, one per vocab size, with prediction details.
    """
    from vocab_estimator.config import DEFAULT_CONFIG
    from vocab_estimator.vocab_model import VocabEstimator

    if vocab_sizes is None:
        vocab_sizes = [2000, 5000, 8000, 10000]

    data = generate_synthetic_data(
        bank, vocab_sizes=vocab_sizes,
        n_questions=n_questions, power=power, seed=seed,
    )

    estimator = VocabEstimator(bank, DEFAULT_CONFIG, seed=seed)
    results: list[dict] = []

    for vsize in sorted(data):
        responses = data[vsize]["responses"]
        known_rate = data[vsize]["known_rate"]

        pred = estimator.estimate_single(responses)
        pred_v = pred["point_estimate"]
        loss = (pred_v - vsize) ** 2

        results.append({
            "vocab_size": vsize,
            "predicted": pred_v,
            "loss": loss,
            "known_rate": known_rate,
            "n_questions": len(responses),
            "raw_estimate": pred.get("raw_estimate"),
            "logistic_estimate": pred.get("logistic_estimate"),
        })

    return results


if __name__ == "__main__":
    main()
