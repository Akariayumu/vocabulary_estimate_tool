#!/usr/bin/env python3
"""Evaluate enhanced difficulty quality against the baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_ENHANCED_VOCAB = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate enhanced difficulty quality."
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE_VOCAB)
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED_VOCAB)
    parser.add_argument("--run-sim", action="store_true",
                        help="Also run quick simulation evaluation (may be slow).")
    parser.add_argument("--sim-users", type=int, default=200,
                        help="Simulation users if --run-sim.")
    return parser.parse_args(argv)


def _bucket_stats(diffs: dict[str, float], cluster_key: str, word_info: dict) -> dict[int, Any]:
    """Compute per-cluster statistics."""
    clusters: dict[int, list[float]] = {}
    for word, d in diffs.items():
        info = word_info.get(word, {})
        c = info.get(cluster_key)
        if c is not None:
            clusters.setdefault(int(c), []).append(d)
    return {
        c: {
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for c, vals in sorted(clusters.items())
    }


def check_anomalies(
    diffs: dict[str, float],
    word_info: dict,
    label: str,
) -> list[dict]:
    """Find potentially anomalous difficulty values."""
    anomalies: list[dict] = []
    for word, d in diffs.items():
        info = word_info.get(word, {})
        stages = info.get("first_stage", "")
        is_easy_stage = stages in ("primary_3", "primary_4", "primary_5", "primary_6", "junior_7")
        is_hard_stage = stages in ("ielts", "cet6")

        if is_easy_stage and d > 0.7:
            anomalies.append({"word": word, "difficulty": d, "stage": stages, "issue": "easy word too hard"})
        if is_hard_stage and d < 0.4:
            anomalies.append({"word": word, "difficulty": d, "stage": stages, "issue": "hard word too easy"})

    return anomalies


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Load both
    with open(args.baseline, encoding="utf-8") as f:
        base_data = json.load(f)
    with open(args.enhanced, encoding="utf-8") as f:
        enh_data = json.load(f)

    base_wts: dict = base_data["word_to_stage"]
    enh_wts: dict = enh_data["word_to_stage"]

    words = sorted(set(base_wts.keys()) & set(enh_wts.keys()))
    print(f"Words in common: {len(words)}")

    base_diffs = {w: base_wts[w]["difficulty"] for w in words}
    enh_diffs = {w: enh_wts[w]["difficulty"] for w in words}

    base_arr = np.array(list(base_diffs.values()))
    enh_arr = np.array(list(enh_diffs.values()))
    delta = enh_arr - base_arr

    print(f"\n=== Distribution comparison ===")
    print(f"{'':>15} {'Baseline':>10} {'Enhanced':>10} {'Delta':>10}")
    print(f"{'Mean':>15} {base_arr.mean():>10.4f} {enh_arr.mean():>10.4f} {delta.mean():>+10.4f}")
    print(f"{'Std':>15} {base_arr.std():>10.4f} {enh_arr.std():>10.4f} {delta.std():>+10.4f}")
    print(f"{'Min':>15} {base_arr.min():>10.4f} {enh_arr.min():>10.4f} {delta.min():>+10.4f}")
    print(f"{'Max':>15} {base_arr.max():>10.4f} {enh_arr.max():>10.4f} {delta.max():>+10.4f}")
    print(f"{'|delta|>0.01':>15} {'':>10} {np.sum(np.abs(delta) > 0.01):>10} {'':>10}")

    print(f"\n=== Correlation ===")
    corr = float(np.corrcoef(base_arr, enh_arr)[0, 1])
    print(f"  Pearson r: {corr:.6f}")

    print(f"\n=== Per-cluster std (smaller = more consistent within class) ===")
    base_clusters = _bucket_stats(base_diffs, "cluster_20", base_wts)
    enh_clusters = _bucket_stats(enh_diffs, "cluster_20", enh_wts)
    base_mean_std = np.mean([s["std"] for s in base_clusters.values()])
    enh_mean_std = np.mean([s["std"] for s in enh_clusters.values()])
    print(f"  Baseline cluster_20 mean std: {base_mean_std:.4f}")
    print(f"  Enhanced  cluster_20 mean std: {enh_mean_std:.4f}")
    print(f"  Change: {(enh_mean_std - base_mean_std):+.4f}")
    if enh_mean_std < base_mean_std:
        print(f"  ✅ Enhanced: cluster std reduced (more consistent)")
    else:
        print(f"  ⚠️  Enhanced: cluster std increased (less consistent)")

    base_clusters_100 = _bucket_stats(base_diffs, "cluster_100", base_wts)
    enh_clusters_100 = _bucket_stats(enh_diffs, "cluster_100", enh_wts)
    base_mstd100 = np.mean([s["std"] for s in base_clusters_100.values()])
    enh_mstd100 = np.mean([s["std"] for s in enh_clusters_100.values()])
    print(f"  Baseline cluster_100 mean std: {base_mstd100:.4f}")
    print(f"  Enhanced  cluster_100 mean std: {enh_mstd100:.4f}")
    print(f"  Change: {(enh_mstd100 - base_mstd100):+.4f}")

    print(f"\n=== Anomaly check ===")
    base_anom = check_anomalies(base_diffs, base_wts, "baseline")
    enh_anom = check_anomalies(enh_diffs, enh_wts, "enhanced")
    print(f"  Baseline anomalies: {len(base_anom)}")
    print(f"  Enhanced  anomalies: {len(enh_anom)}")
    if base_anom:
        print(f"  Baseline examples: {[a['word'] for a in base_anom[:5]]}")
    if enh_anom:
        print(f"  Enhanced examples: {[a['word'] for a in enh_anom[:5]]}")

    print(f"\n=== Top-20 corrections (words with largest |delta|) ===")
    deltas = [(w, base_diffs[w], enh_diffs[w], enh_diffs[w] - base_diffs[w]) for w in words]
    deltas.sort(key=lambda x: -abs(x[3]))
    for w, b, e, d in deltas[:20]:
        stage = base_wts.get(w, {}).get("first_stage", "?")
        print(f"  {w:20s}  stage={stage:10s}  base={b:.4f}  enh={e:.4f}  delta={d:+.4f}")

    # Optional: quick simulation
    if args.run_sim:
        print(f"\n=== Running quick simulation ({args.sim_users} users) ===")
        sys.path.insert(0, str(PROJECT_ROOT))
        from tests.simulation_eval import run_evaluation

        print("  Baseline:")
        base_out = PROJECT_ROOT / "outputs" / "sim_eval_baseline_quick.json"
        base_res = run_evaluation(
            n_users=args.sim_users,
            output=base_out,
            stage_vocab_path=args.baseline,
            quiet=True,
        )
        print(f"    MAE={base_res['summary']['mae']}  R2={base_res['summary']['r2']}")

        print("  Enhanced:")
        enh_out = PROJECT_ROOT / "outputs" / "sim_eval_enhanced_quick.json"
        enh_res = run_evaluation(
            n_users=args.sim_users,
            output=enh_out,
            stage_vocab_path=args.enhanced,
            quiet=True,
        )
        print(f"    MAE={enh_res['summary']['mae']}  R2={enh_res['summary']['r2']}")

        print(f"\n  Summary:")
        delta_mae = enh_res['summary']['mae'] - base_res['summary']['mae']
        delta_r2 = enh_res['summary']['r2'] - base_res['summary']['r2']
        print(f"    MAE delta: {delta_mae:+.1f}")
        print(f"    R²  delta: {delta_r2:+.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())