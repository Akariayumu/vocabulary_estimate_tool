#!/usr/bin/env python3
"""
Cluster words by difficulty using equal-frequency binning.
Ties are broken by word string (alphabetical) to distribute evenly.

Two resolutions: 20 classes (coarse) and 100 classes (fine).
Updates data/stage_vocab.json with cluster_20 and cluster_100 fields.
"""
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path("data/stage_vocab.json")

# ── Load data ──────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    data = json.load(f)

wts = data["word_to_stage"]
words = list(wts.keys())
difficulties = np.array([wts[w]["difficulty"] for w in words], dtype=np.float64)
n = len(difficulties)

print(f"Total words with difficulty: {n}")
print(f"Difficulty range: [{difficulties.min():.4f}, {difficulties.max():.4f}]")
print()

# ── Equal-frequency binning with tie-breaking ─────────────────────
def assign_bins(difficulties, words, n_bins):
    """
    Assign bins based on (difficulty, word) sort order.
    Each bin gets either floor(n/n_bins) or ceil(n/n_bins) words.
    All bins are contiguous and IDs go 0..n_bins-1.
    """
    # Sort by (difficulty, word)
    pairs = list(enumerate(words))
    pairs.sort(key=lambda x: (difficulties[x[0]], x[1]))
    sorted_indices = [p[0] for p in pairs]

    labels = np.empty(n, dtype=int)
    bin_size = n / n_bins
    for b in range(n_bins):
        start = round(b * bin_size)
        end = round((b + 1) * bin_size)
        for idx in sorted_indices[start:end]:
            labels[idx] = b

    return labels

labels_20 = assign_bins(difficulties, words, 20)
labels_100 = assign_bins(difficulties, words, 100)

# ── Write back to JSON ────────────────────────────────────────────
for w, lbl20, lbl100 in zip(words, labels_20, labels_100):
    wts[w]["cluster_20"] = int(lbl20)
    wts[w]["cluster_100"] = int(lbl100)

with open(DATA_PATH, "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Updated data/stage_vocab.json with cluster_20 and cluster_100")
print()

# ── Gather cluster data ───────────────────────────────────────────
clusters_20 = defaultdict(list)
for w, lbl in zip(words, labels_20):
    clusters_20[lbl].append((w, wts[w]["difficulty"]))

clusters_100 = defaultdict(list)
for w, lbl in zip(words, labels_100):
    clusters_100[lbl].append((w, wts[w]["difficulty"]))

# ── Statistics: 20 clusters ───────────────────────────────────────
print("=" * 80)
print("📊 20-CLUSTER REPORT")
print("=" * 80)

for cid in range(20):
    items = clusters_20.get(cid, [])
    if not items:
        print(f"  cluster_20={cid:2d} | EMPTY")
        continue
    diffs = [d for _, d in items]
    words_in = [w for w, _ in items]
    dmin, dmax = min(diffs), max(diffs)
    median_d = np.median(diffs)
    sorted_by_dist = sorted(words_in, key=lambda w: abs(wts[w]["difficulty"] - median_d))
    samples = sorted_by_dist[:3]
    print(f"  cluster_20={cid:2d} | range [{dmin:.4f}, {dmax:.4f}] | count={len(items):5d} | e.g. {samples}")

print()

# ── Statistics: 100 clusters, grouped by 10 ───────────────────────
print("=" * 80)
print("📊 100-CLUSTER REPORT (grouped by 10)")
print("=" * 80)

for group_start in range(0, 100, 10):
    group_end = min(group_start + 10, 100) - 1
    total = 0
    dmin_all, dmax_all = 1.0, 0.0
    details = []
    for cid in range(group_start, group_start + 10):
        items = clusters_100.get(cid, [])
        if not items:
            details.append(f"  c{cid:2d} [EMPTY]")
            continue
        diffs = [d for _, d in items]
        dmin, dmax = min(diffs), max(diffs)
        total += len(items)
        dmin_all = min(dmin_all, dmin)
        dmax_all = max(dmax_all, dmax)
        details.append(f"  c{cid:2d}[{dmin:.3f}-{dmax:.3f}]({len(items)})")
    print(f"  Group {group_start:2d}-{group_end:2d}: range [{dmin_all:.4f}, {dmax_all:.4f}] | total={total:5d}")
    for d in details:
        print(d)
    print()

# ── Verification ──────────────────────────────────────────────────
print("=" * 80)
print("🔍 VERIFICATION")
print("=" * 80)

# 1. Easy words in low cluster
easy_words = ["hello", "apple", "book", "cat", "dog", "she", "he", "yes", "no", "big"]
print("  Easy words (expected low cluster):")
for w in easy_words:
    if w in wts:
        c20 = wts[w]["cluster_20"]
        diff = wts[w]["difficulty"]
        print(f"    {w:12s} → cluster_20={c20}, difficulty={diff:.4f}")
    else:
        print(f"    {w:12s} → NOT FOUND")

# 2. Hard words in high cluster
hard = ["conundrum", "ephemeral", "ubiquitous", "synecdoche", "zeitgeist",
        "idiosyncratic", "soliloquy", "sesquipedalian"]
print("  Hard words (expected high cluster):")
for w in hard:
    if w in wts:
        c20 = wts[w]["cluster_20"]
        diff = wts[w]["difficulty"]
        print(f"    {w:16s} → cluster_20={c20}, difficulty={diff:.4f}")
    else:
        print(f"    {w:16s} → NOT FOUND")

# 3. Cross-boundary check: identical difficulties across adjacent bins
print("  Tie-distribution check (difficulty=0.9447 words):")
d9447_words = [w for w in wts if abs(wts[w]["difficulty"] - 0.9447) < 1e-9]
if d9447_words:
    print(f"    {len(d9447_words)} words with difficulty=0.9447")
    clusters_for_9447 = defaultdict(list)
    for w in d9447_words:
        clusters_for_9447[wts[w]["cluster_20"]].append(w)
    for cid in sorted(clusters_for_9447):
        print(f"      cluster_20={cid}: {len(clusters_for_9447[cid])} words")

print()

# 4. Distribution balance
sizes_20 = [len(clusters_20.get(i, [])) for i in range(20)]
print(f"  20-cluster sizes: min={min(sizes_20)}, max={max(sizes_20)}, mean={np.mean(sizes_20):.1f}, "
      f"std={np.std(sizes_20):.1f}")

sizes_100 = [len(clusters_100.get(i, [])) for i in range(100)]
empty_100 = [i for i, s in enumerate(sizes_100) if s == 0]
nz100 = [s for s in sizes_100 if s > 0]
print(f"  100-cluster sizes: min={min(nz100)}, max={max(nz100)}, mean={np.mean(nz100):.1f}, "
      f"std={np.std(nz100):.1f}")
if empty_100:
    print(f"  ⚠️  Empty 100-clusters: {empty_100}")
else:
    print(f"  ✅ All 100 clusters are non-empty.")

c20_set = sorted(set(labels_20))
c100_set = sorted(set(labels_100))
print(f"  cluster_20 labels: {len(c20_set)}/{20} in use ({c20_set})")
print(f"  cluster_100 labels: {len(c100_set)}/{100} in use (range {min(c100_set)}-{max(c100_set)})")
