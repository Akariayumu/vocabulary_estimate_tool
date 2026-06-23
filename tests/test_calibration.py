"""Validate the combined calibration (方案 A + 方案 B)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import random

from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.config import EstimatorConfig
from vocab_estimator.vocab_model import VocabEstimator, Response


def make_responses(
    bank: VocabBank,
    max_rank: int = 30000,
    known_high: float = 0.5,
    known_mid: float = 0.3,
    known_low: float = 0.1,
    seed: int = 42,
    words_per_band: int = 30,
) -> list[Response]:
    """Synthesize test responses with controlled knowledge rates."""

    rng = random.Random(seed)

    all_items = [item for item in bank.items if item.rank <= max_rank]
    high_items = [it for it in all_items if it.rank <= 5000]
    mid_items = [it for it in all_items if 5000 < it.rank <= 12000]
    low_items = [it for it in all_items if it.rank > 12000]

    responses: list[Response] = []
    band_results = {}

    for label, items, known_rate in [
        ("high", high_items, known_high),
        ("mid", mid_items, known_mid),
        ("low", low_items, known_low),
    ]:
        sample = rng.sample(items, min(words_per_band, len(items)))
        n_known = 0
        for item in sample:
            known = rng.random() < known_rate
            if known:
                n_known += 1
            responses.append((item.word, known))
        band_results[label] = (len(sample), n_known)

    rng.shuffle(responses)
    return responses, band_results


def describe_rates(band_results: dict, label: str) -> None:
    total_known = 0
    total_all = 0
    for band, (n, k) in band_results.items():
        print(f"    {band}: {k}/{n} correct ({100*k//n}%)")
        total_known += k
        total_all += n
    if total_all > 0:
        print(f"    overall: {total_known}/{total_all} correct ({100*total_known//total_all}%)")


def main():
    config = EstimatorConfig()
    bank = VocabBank(config=config)
    print(f"VocabBank size: {len(bank)} words")
    print(f"Fallback mode: {bank.used_fallback}")
    print()

    # All three scenarios use the SAME model instance for reproducibility
    model = VocabEstimator(bank, config)

    # ---- scenario 1: LOW ----
    print("=" * 65)
    print("SCENARIO 1 — LOW: 高频50%正确, 中频20%, 低频0%")
    r1, br1 = make_responses(bank, known_high=0.50, known_mid=0.20, known_low=0.00, words_per_band=25)
    describe_rates(br1, "")
    e1 = model.estimate_single(r1)
    print(f"  raw estimate:      {e1['raw_estimate']}")
    print(f"  logistic estimate: {e1['logistic_estimate']}")
    print(f"  final point:       {e1['point_estimate']}")
    print(f"  level:             {e1['level']}")
    print(f"  90% CI:            {e1['vocabulary_range']}")
    print()

    # ---- scenario 2: MID ----
    print("=" * 65)
    print("SCENARIO 2 — MID:  高频80%正确, 中频50%, 低频20%")
    r2, br2 = make_responses(bank, known_high=0.80, known_mid=0.50, known_low=0.20, words_per_band=30)
    describe_rates(br2, "")
    e2 = model.estimate_single(r2)
    print(f"  raw estimate:      {e2['raw_estimate']}")
    print(f"  logistic estimate: {e2['logistic_estimate']}")
    print(f"  final point:       {e2['point_estimate']}")
    print(f"  level:             {e2['level']}")
    print(f"  90% CI:            {e2['vocabulary_range']}")
    print()

    # ---- scenario 3: HIGH ----
    print("=" * 65)
    print("SCENARIO 3 — HIGH: 高频100%正确, 中频80%, 低频30%")
    r3, br3 = make_responses(bank, known_high=1.00, known_mid=0.80, known_low=0.30, words_per_band=30)
    describe_rates(br3, "")
    e3 = model.estimate_single(r3)
    print(f"  raw estimate:      {e3['raw_estimate']}")
    print(f"  logistic estimate: {e3['logistic_estimate']}")
    print(f"  final point:       {e3['point_estimate']}")
    print(f"  level:             {e3['level']}")
    print(f"  90% CI:            {e3['vocabulary_range']}")
    print()

    # ---- cross-check: piecewise_calibrate isolated ----
    print("=" * 65)
    print("PIECEWISE CALIBRATION (isolated) — show compression curve")
    for raw_in in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                   10000, 12000, 15000, 18000, 20000, 22000, 25000]:
        cal = model.piecewise_calibrate(float(raw_in))
        print(f"  raw {raw_in:>6} → piecewise {cal:>8.1f}")

    # ---- weight function values ----
    print()
    print("=" * 65)
    print("WEIGHT FUNCTION (isolated)")
    for r in [1, 10, 100, 500, 1000, 3000, 5000, 10000, 15000, 20000, 25000]:
        w = VocabEstimator._compute_weight(float(r))
        print(f"  rank {r:>6} → weight {w:.6f}")

    print()
    print("=" * 65)
    print("VALIDATION SUMMARY")
    print()
    ok_low = 4000 <= e1['point_estimate'] <= 7500
    ok_mid = 5500 <= e2['point_estimate'] <= 9500
    ok_high = 7000 <= e3['point_estimate'] <= 15000
    print(f"  LOW  scenario  → got {e1['point_estimate']:>6}  range [4000, 7500]  {'✓' if ok_low else '✗'}")
    print(f"  MID  scenario  → got {e2['point_estimate']:>6}  range [5500, 9500]  {'✓' if ok_mid else '✗'}")
    print(f"  HIGH scenario  → got {e3['point_estimate']:>6}  range [7000,15000]  {'✓' if ok_high else '✗'}")

    ok = all([ok_low, ok_mid, ok_high])
    print()
    print(f"Overall: {'✓ PASS' if ok else '✗ FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
