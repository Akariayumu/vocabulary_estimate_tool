#!/usr/bin/env python3
"""Train calibration: analyze model fit vs synthetic data."""
import sys, json, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vocab_estimator.config import EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator

bank = VocabBank(EstimatorConfig())

# ── Load data ──
with open("test_samples_trainer.json") as f:
    raw = json.load(f)
users = raw["users"]

# Aggregate by vocab level
by_level = {}
for u in users:
    v = u["vocab_size"]
    if v not in by_level:
        by_level[v] = []
    by_level[v].append(u["responses"])

levels = sorted(by_level.keys())
print(f"{len(levels)} vocab levels, sampling ~{len(levels[::10])} for test:")
print(f"{'Vocab':>8} {'RawLog':>8} {'Pred':>8} {'Err':>8}")
print("-" * 35)

estimator = VocabEstimator(bank, EstimatorConfig())
for v in levels[::10]:  # every 10th level
    resp = [(r["word"], r["known"]) for r in by_level[v][0]]
    r = estimator.estimate_single(resp)
    print(f"{v:>8} {r['raw_estimate']:>8} {r['point_estimate']:>8} {r['point_estimate']-v:>+8}")

print(f"\n{'='*50}")
print("Insight: synthetic data has 'perfect cutoff' (know top N, nothing more)")
print("Real learners have SMOOTH decay across buckets.")
print("The logistic model expects smooth decay - hence misfit.")
print(f"{'='*50}")

# ── Better synthetic data: use logistic model itself ──
print(f"\nBetter approach: generate data USING the model:")
print(f"1. Pick true_vocab_size V")
print(f"2. Use current model to estimate P(known|rank) for each rank")
print(f"3. At each bucket, sample words with P(known) = model's prediction")
print(f"4. This creates realistic data that the model can learn from")
print(f"\nThis is effectively: use synthetic data to find α,β,k,knots")
print(f"that make raw_logistic more linear with vocabulary size.")
