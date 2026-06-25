#!/usr/bin/env python3
"""Apply v1 fixed difficulty cluster boundaries to a v2 stage vocab.

The boundaries are extracted from the original v1 vocabulary as the maximum
difficulty in each existing cluster. Assignment uses the first v1 upper
boundary that is greater than or equal to the word's current difficulty.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V1 = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_V2 = PROJECT_ROOT / "data" / "stage_vocab_v2.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stage_vocab_v2_fixed_cluster.json"


def load_vocab(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("word_to_stage"), dict):
        raise ValueError(f"{path} must contain a word_to_stage object")
    return data


def usable_entries(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = data["word_to_stage"]
    return {
        word: info
        for word, info in entries.items()
        if isinstance(word, str)
        and isinstance(info, dict)
        and info.get("difficulty") is not None
    }


def extract_upper_bounds(
    entries: dict[str, dict[str, Any]], cluster_key: str, n_clusters: int
) -> list[float]:
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for word, info in entries.items():
        if info.get(cluster_key) is None:
            raise ValueError(f"{word!r} is missing {cluster_key}")
        cluster = int(info[cluster_key])
        if not 0 <= cluster < n_clusters:
            raise ValueError(f"{word!r} has invalid {cluster_key}={cluster}")
        by_cluster[cluster].append(float(info["difficulty"]))

    bounds: list[float] = []
    for cluster in range(n_clusters):
        values = by_cluster.get(cluster)
        if not values:
            raise ValueError(f"v1 has no entries for {cluster_key}={cluster}")
        bounds.append(max(values))

    for left, right in zip(bounds, bounds[1:]):
        if right + 1e-12 < left:
            raise ValueError(f"{cluster_key} boundaries are not non-decreasing")
    return bounds


def assign_by_upper_bounds(difficulty: float, upper_bounds: list[float]) -> int:
    cluster = bisect.bisect_left(upper_bounds, difficulty)
    if cluster >= len(upper_bounds):
        return len(upper_bounds) - 1
    return cluster


def apply_clusters(
    vocab: dict[str, Any],
    cluster_20_bounds: list[float],
    cluster_100_bounds: list[float],
) -> None:
    for word, info in vocab["word_to_stage"].items():
        if not isinstance(info, dict) or info.get("difficulty") is None:
            raise ValueError(f"{word!r} is missing difficulty")
        difficulty = float(info["difficulty"])
        info["cluster_20"] = assign_by_upper_bounds(difficulty, cluster_20_bounds)
        info["cluster_100"] = assign_by_upper_bounds(difficulty, cluster_100_bounds)


def bucket_stats(entries: dict[str, dict[str, Any]], cluster_key: str, n_clusters: int) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[float]] = defaultdict(list)
    for info in entries.values():
        if info.get(cluster_key) is None or info.get("difficulty") is None:
            continue
        by_cluster[int(info[cluster_key])].append(float(info["difficulty"]))

    rows: list[dict[str, Any]] = []
    for cluster in range(n_clusters):
        values = by_cluster.get(cluster, [])
        rows.append(
            {
                "cluster": cluster,
                "count": len(values),
                "min": min(values) if values else math.nan,
                "max": max(values) if values else math.nan,
            }
        )
    return rows


def consistency_summary(
    v1_entries: dict[str, dict[str, Any]],
    v2_entries: dict[str, dict[str, Any]],
    cluster_key: str,
) -> dict[str, Any]:
    common = sorted(set(v1_entries) & set(v2_entries))
    same = 0
    movement: Counter[int] = Counter()
    top_moves: Counter[tuple[int, int]] = Counter()

    for word in common:
        old = int(v1_entries[word][cluster_key])
        new = int(v2_entries[word][cluster_key])
        if old == new:
            same += 1
        else:
            movement[abs(new - old)] += 1
            top_moves[(old, new)] += 1

    total = len(common)
    changed = total - same
    return {
        "common": total,
        "same": same,
        "changed": changed,
        "same_pct": same / total if total else 0.0,
        "changed_pct": changed / total if total else 0.0,
        "movement": movement,
        "top_moves": top_moves.most_common(10),
    }


def find_tie_boundaries(
    entries: dict[str, dict[str, Any]], cluster_key: str, n_clusters: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = bucket_stats(entries, cluster_key, n_clusters)
    for cluster in range(n_clusters - 1):
        left_max = stats[cluster]["max"]
        right_min = stats[cluster + 1]["min"]
        if not math.isnan(left_max) and not math.isnan(right_min) and abs(left_max - right_min) < 1e-12:
            counts: Counter[int] = Counter()
            for info in entries.values():
                if abs(float(info["difficulty"]) - left_max) < 1e-12:
                    counts[int(info[cluster_key])] += 1
            rows.append(
                {
                    "after_cluster": cluster,
                    "difficulty": left_max,
                    "word_count_at_difficulty": sum(counts.values()),
                    "cluster_counts": dict(sorted(counts.items())),
                }
            )
    return rows


def fmt_float(value: float) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.4f}"


def print_boundary_summary(bounds: list[float], label: str) -> None:
    print(f"{label} upper boundaries:")
    if len(bounds) <= 20:
        print("  " + ", ".join(f"{i}:{value:.4f}" for i, value in enumerate(bounds)))
    else:
        for start in range(0, len(bounds), 10):
            chunk = bounds[start : start + 10]
            print("  " + ", ".join(f"{start + i}:{value:.4f}" for i, value in enumerate(chunk)))
    print()


def print_consistency(label: str, summary: dict[str, Any]) -> None:
    print(f"{label} consistency for original words:")
    print(
        f"  same={summary['same']:,}/{summary['common']:,} "
        f"({summary['same_pct']:.1%}), changed={summary['changed']:,} "
        f"({summary['changed_pct']:.1%})"
    )
    if summary["top_moves"]:
        moves = ", ".join(f"{old}->{new}:{count}" for (old, new), count in summary["top_moves"])
        print(f"  top moves: {moves}")
    print()


def print_bucket_report(rows: list[dict[str, Any]], label: str) -> None:
    print(f"{label} bucket sizes and difficulty ranges:")
    for row in rows:
        print(
            f"  {label}={row['cluster']:>2} | count={row['count']:>5} "
            f"| range [{fmt_float(row['min'])}, {fmt_float(row['max'])}]"
        )
    print()


def add_metadata(
    output: dict[str, Any],
    *,
    v1_path: Path,
    v2_path: Path,
    cluster_20_bounds: list[float],
    cluster_100_bounds: list[float],
    c20_summary: dict[str, Any],
    c100_summary: dict[str, Any],
) -> None:
    meta = output.setdefault("meta", {})
    meta["fixed_clustering"] = {
        "name": "v1_fixed_difficulty_upper_boundaries",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "base_vocab": str(v1_path.relative_to(PROJECT_ROOT) if v1_path.is_relative_to(PROJECT_ROOT) else v1_path),
        "input_vocab": str(v2_path.relative_to(PROJECT_ROOT) if v2_path.is_relative_to(PROJECT_ROOT) else v2_path),
        "assignment": "first v1 cluster whose extracted upper difficulty boundary is >= word difficulty; values above the final boundary use the final cluster",
        "tie_policy": "difficulty_only_lower_bucket_for_exact_boundary_ties",
        "cluster_20_upper_bounds": cluster_20_bounds,
        "cluster_100_upper_bounds": cluster_100_bounds,
        "original_word_consistency": {
            "cluster_20": {
                "common": c20_summary["common"],
                "same": c20_summary["same"],
                "changed": c20_summary["changed"],
                "same_pct": c20_summary["same_pct"],
            },
            "cluster_100": {
                "common": c100_summary["common"],
                "same": c100_summary["same"],
                "changed": c100_summary["changed"],
                "same_pct": c100_summary["same_pct"],
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply v1 fixed difficulty boundaries to v2 vocabulary clusters.")
    parser.add_argument("--v1", type=Path, default=DEFAULT_V1, help="Original v1 stage vocab JSON.")
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2, help="V2 stage vocab JSON to relabel.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v1_path = args.v1.resolve()
    v2_path = args.v2.resolve()
    output_path = args.output.resolve()

    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} already exists; pass --force to overwrite it explicitly")

    v1 = load_vocab(v1_path)
    v2 = load_vocab(v2_path)
    v1_entries = usable_entries(v1)

    cluster_20_bounds = extract_upper_bounds(v1_entries, "cluster_20", 20)
    cluster_100_bounds = extract_upper_bounds(v1_entries, "cluster_100", 100)
    print_boundary_summary(cluster_20_bounds, "cluster_20")
    print_boundary_summary(cluster_100_bounds, "cluster_100")

    tie_20 = find_tie_boundaries(v1_entries, "cluster_20", 20)
    tie_100 = find_tie_boundaries(v1_entries, "cluster_100", 100)
    print(f"Boundary ties in v1: cluster_20={len(tie_20)}, cluster_100={len(tie_100)}")
    if tie_20:
        examples = ", ".join(
            f"after {row['after_cluster']} at {row['difficulty']:.4f} "
            f"({row['word_count_at_difficulty']} words)"
            for row in tie_20[:5]
        )
        print(f"  cluster_20 examples: {examples}")
    print()

    output = copy.deepcopy(v2)
    apply_clusters(output, cluster_20_bounds, cluster_100_bounds)
    output_entries = usable_entries(output)

    c20_summary = consistency_summary(v1_entries, output_entries, "cluster_20")
    c100_summary = consistency_summary(v1_entries, output_entries, "cluster_100")
    print_consistency("cluster_20", c20_summary)
    print_consistency("cluster_100", c100_summary)

    print_bucket_report(bucket_stats(output_entries, "cluster_20", 20), "cluster_20")
    print_bucket_report(bucket_stats(output_entries, "cluster_100", 100), "cluster_100")

    add_metadata(
        output,
        v1_path=v1_path,
        v2_path=v2_path,
        cluster_20_bounds=cluster_20_bounds,
        cluster_100_bounds=cluster_100_bounds,
        c20_summary=c20_summary,
        c100_summary=c100_summary,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
