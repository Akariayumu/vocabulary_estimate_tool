#!/usr/bin/env python3
"""Fast calibration training: pre-compute raw estimates, then fit α/β/k/knots."""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocab_estimator.config import EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator
import numpy as np

bank = VocabBank(EstimatorConfig())
estimator = VocabEstimator(bank, EstimatorConfig())

# ── Load data ──
with open("test_samples_trainer.json") as f:
    raw = json.load(f)
users = raw["users"]
print(f"Loaded {len(users)} synthetic users")

# ── Test with 20 representative levels ──
levels = list(range(500, 20001, 1000))  # 20 levels
print(f"Testing {len(levels)} levels (1 user each):")

results = []
for v in levels:
    # Pick first user at this vocab level
    vu = [u for u in users if u["vocab_size"] == v]
    if not vu:
        continue
    u = vu[0]
    resp = [(r["word"], r["known"]) for r in u["responses"]]
    r = estimator.estimate_single(resp)
    results.append((v, r["point_estimate"], r["raw_estimate"], r["logistic_estimate"]))
    print(f"  v={v:>5}: pred={r['point_estimate']:>5} raw={r['raw_estimate']:>5} log={r['logistic_estimate']:>5}")

# ── Calibrate function with tunable params ──
def calibrate_simple(raw_est, k, k1, k2, k3):
    """Same as VocabEstimator.calibrate with tunable piecewise slopes."""
    cal = 20000.0 * math.tanh(k * raw_est)
    b = [3000, 8000, 22000]
    ks = [k1, k2, k3]
    prev_v, prev_b = 0.0, 0.0
    for i, boundary in enumerate(b):
        if cal <= boundary:
            return prev_v + (cal - prev_b) * ks[i]
        prev_v += (boundary - prev_b) * ks[i]
        prev_b = boundary
    return prev_v + (cal - prev_b) * ks[-1]

# ── Loss over all users ──

def fast_estimate(responses):
    """Fast point estimate without bootstrap."""
    return estimator.logistic_estimate(responses)["estimate"]

def full_loss(k, k1, k2, k3):
    total = 0.0
    n = 0
    for u in users[:200]:  # subset for speed
        resp = [(r["word"], r["known"]) for r in u["responses"]]
        log_est = estimator.logistic_estimate(resp)["estimate"]
        pred = calibrate_simple(log_est, k, k1, k2, k3)
        err = pred - u["vocab_size"]
        total += err * err
        n += 1
    return total / n

# ── Evaluate current params ──
k_current = 0.0000691
k1, k2, k3 = 1.0, 0.45, 1.28
l = full_loss(k_current, k1, k2, k3)
print(f"\n{'='*50}")
print(f"Current loss (200 users): {l:.0f}")

# ── Test adjustments ──
print(f"\nSensitivity analysis:")
for k_test in [0.00003, 0.00005, 0.0000691, 0.00008, 0.00010]:
    l = full_loss(k_test, k1, k2, k3)
    print(f"  k={k_test:.7f}: loss={l:.0f}")
for k1_test in [0.7, 0.85, 1.0, 1.15, 1.30]:
    l = full_loss(k_current, k1_test, k2, k3)
    print(f"  k1={k1_test:.2f}:    loss={l:.0f}")
for k2_test in [0.30, 0.45, 0.60, 0.75, 0.90]:
    l = full_loss(k_current, k1, k2_test, k3)
    print(f"  k2={k2_test:.2f}:    loss={l:.0f}")
for k3_test in [1.0, 1.10, 1.20, 1.28, 1.40]:
    l = full_loss(k_current, k1, k2, k3_test)
    print(f"  k3={k3_test:.3f}:   loss={l:.0f}")

print(f"\n{'='*50}")
print("Best combo found via grid search:")
best_loss = float('inf')
best = None
for tk in [0.00005, 0.0000691, 0.00008]:
    for tk1 in [0.85, 1.0, 1.10]:
        for tk2 in [0.45, 0.55, 0.65]:
            for tk3 in [1.10, 1.20, 1.28, 1.35]:
                l = full_loss(tk, tk1, tk2, tk3)
                if l < best_loss:
                    best_loss = l
                    best = (tk, tk1, tk2, tk3)
if best:
    print(f"  k={best[0]:.7f}, k1={best[1]:.2f}, k2={best[2]:.2f}, k3={best[3]:.2f}")
    print(f"  loss: {best_loss:.0f} (vs current {full_loss(k_current, k1, k2, k3):.0f})")
    
    # Show improvement
    print(f"\n{'='*50}")
    print(f"Before vs After:")
    print(f"{'Vocab':>8} {'Before':>8} {'After':>8} {'Bias_B':>8} {'Bias_A':>8}")
    for v in levels[:10]:
        vu = [u for u in users if u["vocab_size"] == v]
        if not vu: continue
        resp = [(r["word"], r["known"]) for r in vu[0]["responses"]]
        log_est = estimator.logistic_estimate(resp)["estimate"]
        before = calibrate_simple(log_est, k_current, k1, k2, k3)
        after = calibrate_simple(log_est, best[0], best[1], best[2], best[3])
        print(f"  {v:>5}: {before:>6.0f} {after:>6.0f} {before-v:>+6.0f} {after-v:>+6.0f}")
