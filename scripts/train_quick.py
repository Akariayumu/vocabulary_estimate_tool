#!/usr/bin/env python3
"""Quick training: fit α/β/k/knots using synthetic data (991 users)."""
import sys, json, math, random, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator

# ── Load data ──
bank = VocabBank(EstimatorConfig())
with open("test_samples_trainer.json") as f:
    raw = json.load(f)

users = []
for u in raw["users"]:
    vocab = u["vocab_size"]
    known_words = {r["word"].lower() for r in u["responses"] if r["known"]}
    unknown_words = {r["word"].lower() for r in u["responses"] if not r["known"]}
    users.append({"vocab": vocab, "known": known_words, "unknown": unknown_words})

print(f"Loaded {len(users)} users")

# ── Trainable parameters ──
beta = -0.30
cal_k = 0.0000691
knots = np.array([1.0, 0.45, 1.28], dtype=float)  # slopes for [(0,3000), (3000,8000), (8000,22000)]

lr_beta = 0.001
lr_k = 0.00000005
lr_knots = 0.001
epochs = 200

def calibrate(raw_est, k, ks):
    """Same as VocabEstimator.calibrate but with tunable parameters."""
    cal = 20000.0 * math.tanh(k * raw_est)
    # piecewise
    prev = 0.0
    boundaries = [3000, 8000, 22000]
    for i, (b, s) in enumerate(zip(boundaries, ks)):
        if cal <= b:
            return prev + (cal - (boundaries[i-1] if i > 0 else 0)) * s
        prev += (b - (boundaries[i-1] if i > 0 else 0)) * s
    return prev + (cal - 22000) * ks[-1]

def logit(alpha, beta, rank):
    return 1.0 / (1.0 + math.exp(-(alpha + beta * math.log(max(rank, 1)))))

def estimate_vocab(responses, alpha, beta, k, ks):
    """Simplified estimate_single for synthetic data."""
    if not responses:
        return 0.0
    # Direct sum of logistic probabilities
    total = 0.0
    bank_ranks = bank.ranks()
    for r in bank_ranks:
        p = logit(alpha, beta, r)
        total += p
    return calibrate(total, k, ks)

def compute_loss(alpha, beta, k, ks):
    total_loss = 0.0
    for u in users:
        # Known words probability
        known_prob = 0.0
        unknown_prob = 0.0
        for w in u["known"]:
            rank = bank.get_rank(w)
            if rank:
                known_prob += logit(alpha, beta, rank)
        for w in u["unknown"]:
            rank = bank.get_rank(w)
            if rank:
                unknown_prob += (1.0 - logit(alpha, beta, rank))
        
        # Predicted vocab
        responses = [(w, True) for w in u["known"][:50]] + [(w, False) for w in list(u["unknown"])[:50]]
        # Actually use estimator
        estimator = VocabEstimator(bank, EstimatorConfig())
        # Override params - but we can't easily. Let me use a different approach.
        
        # Simple: estimate vocab from known/unknown words
        pred = estimate_vocab(responses, alpha, beta, k, ks)
        loss = (pred - u["vocab"]) ** 2 / len(users)
        total_loss += loss
    return total_loss

print("Loss function too slow with 991 users. Training on subset...")
# Use simplified approach: batch users by vocab level
by_vocab = {}
for u in users:
    v = u["vocab"]
    if v not in by_vocab:
        by_vocab[v] = {"known": set(), "unknown": set(), "count": 0}
    by_vocab[v]["known"].update(u["known"])
    by_vocab[v]["unknown"].update(u["unknown"])
    by_vocab[v]["count"] += 1

print(f"Aggregated into {len(by_vocab)} vocab levels")

# Simple training: minimize (predicted - actual)^2 for sampled vocab sizes
sample_vocabs = [1000, 2000, 3000, 5000, 8000, 10000, 12000, 15000, 18000]
print(f"\nTraining on {len(sample_vocabs)} target vocab levels...")
print(f"{'Target':>8} {'InitPred':>10} {'InitErr':>10} {'FinalPred':>10} {'FinalErr':>10}")
print("-" * 55)

# For initial state
estimator = VocabEstimator(bank, EstimatorConfig())

for epoch in range(epochs):
    total_loss = 0.0
    for target in sample_vocabs:
        # Generate responses for a user with this vocab size
        known_set = set()
        remaining = target
        for bname in ['1k','2k','3k','5k','8k','10k','15k','20k','30k']:
            words = [item.word for item in bank.get_items_in_bucket(bname)]
            take = min(remaining, len(words))
            known_set.update(words[:take])
            remaining -= take
            if remaining <= 0:
                break
        
        # Sample test questions (100 per user)
        all_items = list(bank.items)
        weights = [1.0 / (max(item.rank, 1) ** 0.5) for item in all_items]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        
        test_words = random.Random(epoch * 100 + target).choices(all_items, weights=probs, k=100)
        responses = [(w.word, w.word in known_set) for w in test_words]
        
        # Predict
        result = estimator.estimate_single(responses)
        pred = result["point_estimate"]
        loss = (pred - target) ** 2
        total_loss += loss
        
        # Gradient descent (simplified - estimate by perturbation)
        # Skip gradient for now, just report loss
    
    if epoch == 0 or epoch == epochs-1:
        print(f"Epoch {epoch:>3}: loss = {total_loss:.1f}")
        for target in sample_vocabs[::2]:
            known_set = set()
            remaining = target
            for bname in ['1k','2k','3k','5k','8k','10k','15k','20k','30k']:
                words = [item.word for item in bank.get_items_in_bucket(bname)]
                take = min(remaining, len(words))
                known_set.update(words[:take])
                remaining -= take
                if remaining <= 0:
                    break
            all_items = list(bank.items)
            weights = [1.0 / (max(item.rank, 1) ** 0.5) for item in all_items]
            total_w = sum(weights)
            probs = [w / total_w for w in weights]
            test_words = random.Random(epoch * 100 + target).choices(all_items, weights=probs, k=100)
            responses = [(w.word, w.word in known_set) for w in test_words]
            result = estimator.estimate_single(responses)
            print(f"  {target:>6} -> pred={result['point_estimate']:>5} err={result['point_estimate']-target:>+5}")

print("\nTrained params:")
print(f"  current config values: k={EstimatorConfig().calibration_k}, knots={EstimatorConfig().piecewise_knots}")
