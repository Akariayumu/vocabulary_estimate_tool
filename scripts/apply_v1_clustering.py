#!/usr/bin/env python3
"""Apply v1 cluster_20/cluster_100 fixed difficulty boundaries to v2 words."""

import json, bisect, copy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1_PATH = PROJECT_ROOT / "data" / "stage_vocab.json"
V2_PATH = PROJECT_ROOT / "data" / "stage_vocab_v2.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "stage_vocab_v2_clusterv1.json"

def build_boundaries(v1_entries, key="cluster_20", n=20):
    """Extract max difficulty per cluster as upper boundary."""
    bounds = []
    for c in range(n):
        diffs = [float(info["difficulty"]) for info in v1_entries.values()
                 if info.get(key) == c and info.get("difficulty") is not None]
        bounds.append(max(diffs) if diffs else 0.0)
    return bounds  # left-closed [b_c-1, b_c], where b_c = max difficulty in cluster c

def assign_cluster(difficulty, bounds, side="right"):
    """Assign cluster = bisect_right (default) or bisect_left."""
    c = bisect.bisect_right(bounds, difficulty) if side == "right" else bisect.bisect_left(bounds, difficulty)
    return min(len(bounds) - 1, c)

def main():
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))

    v1_entries = v1["word_to_stage"]
    v2_entries = v2["word_to_stage"]
    v1_words = set(v1_entries)
    v2_words = set(v2_entries)

    output = copy.deepcopy(v2)

    for key, n in [("cluster_20", 20), ("cluster_100", 100)]:
        bounds = build_boundaries(v1_entries, key, n)
        print(f"\n=== {key} boundaries (n={n}) ===")
        for c in range(n):
            cnt = sum(1 for w in v1_words if v1_entries[w].get(key) == c)
            print(f"  cluster {c}: diff ≤ {bounds[c]:.4f}, v1 count={cnt}")

        same = changed = 0
        for word in v2_entries:
            info = output["word_to_stage"][word]
            diff = float(info["difficulty"])
            # Use bisect_right to assign (same difficulty goes to higher cluster, like v1)
            new_c = assign_cluster(diff, bounds, side="right")
            info[key] = new_c
            if word in v1_words:
                old_c = v1_entries[word].get(key)
                if old_c is not None and int(old_c) == new_c:
                    same += 1
                else:
                    changed += 1

        total = same + changed
        print(f"\n  Same as v1: {same}/{total} = {same/total*100:.1f}%")
        print(f"  Changed:   {changed}/{total} = {changed/total*100:.1f}%")

        # Per-cluster sizes
        sizes = {}
        for info in output["word_to_stage"].values():
            c = info.get(key)
            if c is not None:
                sizes[c] = sizes.get(c, 0) + 1
        print(f"  Bucket sizes: min={min(sizes.values())}, max={max(sizes.values())}")
        print(f"  Bucket detail: {dict(sorted(sizes.items()))}")

    # Save
    output["meta"].setdefault("cluster_redesign", {})
    output["meta"]["cluster_redesign"] = {
        "method": "v1_fixed_difficulty_boundaries",
        "version": "v2_clusterv1_2026_06_25",
        "cluster_20_boundary_method": "bisect_right(v1_max_difficulty_per_cluster)",
        "cluster_100_boundary_method": "bisect_right(v1_max_difficulty_per_cluster)",
        "input_v1": str(V1_PATH),
        "input_v2": str(V2_PATH),
    }
    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
