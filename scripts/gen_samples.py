#!/usr/bin/env python3
"""为校准训练生成合成测试样本。"""
import sys, json, random, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank

bank = VocabBank(DEFAULT_CONFIG)
rng = random.Random(42)

N_SAMPLES = 1000
N_USERS = 1000
TEST_QS = 100
POWER = 0.5

# 要模拟的词汇量等级（从 500 到 20000 均匀分布）
vocab_levels = [int(x) for x in range(200, 20001, 20)]  # 约 1000 个等级

def build_known_set(vocab_size):
    """从最高频 buckets 开始填充，构建已知词集合。"""
    known = set()
    remaining = vocab_size
    for bucket_name in ['1k','2k','3k','5k','8k','10k','15k','20k','30k']:
        words = [item.word for item in bank.get_items_in_bucket(bucket_name)]
        take = min(remaining, len(words))
        known.update(words[:take])
        remaining -= take
        if remaining <= 0:
            break
    return known

def sample_test_questions(n=100, power=0.5):
    """带高频偏置地抽样测试题（类似真实考试）。"""
    all_items = list(bank.items)
    # 权重与 1/rank^power 成正比
    weights = [1.0 / (max(item.rank, 1) ** power) for item in all_items]
    total = sum(weights)
    probs = [w / total for w in weights]
    chosen = rng.choices(all_items, weights=probs, k=n)
    return [c.word for c in chosen]

# 生成数据
dataset = {}
for vid, target_size in enumerate(vocab_levels):
    known_set = build_known_set(target_size)
    test_words = sample_test_questions(TEST_QS, POWER)
    responses = []
    for w in test_words:
        lemma = bank.lemmatizer.normalize(w).lower()
        known = lemma in known_set or w.lower() in known_set
        responses.append({"word": w, "known": known})
    dataset[str(vid)] = {
        "vocab_size": target_size,
        "responses": responses,
        "n_known": sum(1 for r in responses if r["known"]),
    }

# 统计
print(f"Generated {len(dataset)} synthetic users")
print(f"Vocabulary levels: {vocab_levels[0]} - {vocab_levels[-1]}")
print(f"Test questions per user: {TEST_QS}")
print(f"Power: {POWER}")
print()
print(f"{'Vocab':>8} {'Known%':>8} {'Known':>6}/{TEST_QS}")
print("-" * 30)
known_rates = []
for vid in sorted(dataset.keys(), key=lambda k: dataset[k]['vocab_size']):
    d = dataset[vid]
    kr = d['n_known'] / TEST_QS * 100
    known_rates.append(kr)
    if int(vid) % 10 == 0 or int(vid) == len(dataset)-1:
        print(f"{d['vocab_size']:>8} {kr:>7.1f}% {d['n_known']:>3}/{TEST_QS}")

# 保存
output = Path("test_samples.json")
with open(output, 'w') as f:
    json.dump({"dataset": dataset, "meta": {
        "n_users": len(dataset),
        "questions_per_user": TEST_QS,
        "power": POWER,
        "vocab_range": [vocab_levels[0], vocab_levels[-1]]
    }}, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {output} ({output.stat().st_size/1024:.0f} KB)")
