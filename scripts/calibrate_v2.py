#!/usr/bin/env python3
"""Search calibration parameters for the v2 clustered stage vocabulary.

This script deliberately does not modify ``vocab_estimator/stratified_quiz.py``.
It reuses the synthetic quiz simulator, caches each user's fitted ``theta`` and
unscaled ``raw_sum``, then evaluates alternative calibration functions against
the same simulated response set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.simulation_eval import (  # noqa: E402
    _bucket_name,
    _difficulty_logits,
    _expected_vocab_from_logits,
    generate_synthetic_users,
    load_vocab_bank,
    simulate_quiz,
)
from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig  # noqa: E402
from vocab_estimator.stratified_quiz import StratifiedQuiz  # noqa: E402


DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab_v2_clusterv1.json"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "calibration_search_results.md"
DEFAULT_RECORDS_JSON = PROJECT_ROOT / "outputs" / "calibration_search_records.json"


@dataclass(frozen=True)
class SimRecord:
    user_id: int
    true_vocab: int
    true_theta: float
    theta_hat: float
    raw_sum: float
    current_point_estimate: int
    total_questions: int
    phase1_total: int
    phase2_total: int
    bucket: str


def piecewise_calibrate(estimate: float, config: EstimatorConfig = DEFAULT_CONFIG) -> float:
    """Mirror StratifiedQuiz._piecewise_calibrate without touching the class."""
    if estimate <= 0:
        return estimate

    prev_boundary = 0.0
    prev_value = 0.0
    for boundary, slope in config.piecewise_knots:
        boundary_f = float(boundary)
        if estimate <= boundary_f:
            return float(prev_value + (estimate - prev_boundary) * slope)
        prev_value += (boundary_f - prev_boundary) * slope
        prev_boundary = boundary_f

    return float(prev_value + (estimate - prev_boundary) * config.piecewise_knots[-1][1])


def tanh_piecewise_calibrate(estimate: float, config: EstimatorConfig = DEFAULT_CONFIG) -> float:
    """Mirror StratifiedQuiz._calibrate."""
    if estimate <= 0:
        return estimate

    cal = float(config.calibration_native_max) * math.tanh(config.calibration_k * estimate)
    if config.enable_piecewise_calibration:
        cal = piecewise_calibrate(cal, config)
    return float(cal)


def estimate_scale(raw_sum: float, scale_factor: float) -> float:
    return tanh_piecewise_calibrate(raw_sum * scale_factor)


def theta_piecewise_scale(theta: float) -> float:
    if theta < -1.0:
        return 1.0
    if theta > 1.0:
        return 0.65
    return 0.8


def metric_summary(records: Sequence[SimRecord], predictor: Callable[[SimRecord], float]) -> dict[str, Any]:
    true_vals = np.array([r.true_vocab for r in records], dtype=float)
    pred_vals = np.array([predictor(r) for r in records], dtype=float)
    errors = pred_vals - true_vals

    mae = float(np.mean(np.abs(errors))) if len(records) else 0.0
    rmse = float(np.sqrt(np.mean(errors**2))) if len(records) else 0.0
    mean_bias = float(np.mean(errors)) if len(records) else 0.0
    if len(records) < 2 or np.std(true_vals) == 0 or np.std(pred_vals) == 0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(true_vals, pred_vals)[0, 1])
    sst = float(np.sum((true_vals - np.mean(true_vals)) ** 2))
    sse = float(np.sum(errors**2))
    r2 = 0.0 if sst == 0 else 1.0 - sse / sst

    return {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(float(r2), 6),
        "correlation": round(corr, 6),
        "mean_bias": round(mean_bias, 3),
    }


def bucket_summaries(records: Sequence[SimRecord], predictor: Callable[[SimRecord], float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in ["low_1k_3k", "mid_3k_8k", "high_8k_15k"]:
        subset = [r for r in records if r.bucket == bucket]
        if not subset:
            continue
        rows.append({"bucket": bucket, "n": len(subset), **metric_summary(subset, predictor)})
    return rows


def simulate_records(
    *,
    n_users: int,
    stage_vocab_path: Path,
    seed: int,
    true_min: int,
    true_max: int,
    quiet: bool,
) -> list[SimRecord]:
    vocab_bank = load_vocab_bank(stage_vocab_path)
    difficulty_logits = _difficulty_logits(vocab_bank)
    quiz = StratifiedQuiz(stage_vocab_path=stage_vocab_path)
    records: list[SimRecord] = []

    users = generate_synthetic_users(
        n_users=n_users,
        vocab_bank=vocab_bank,
        true_min=true_min,
        true_max=true_max,
        seed=seed,
    )
    for idx, user in enumerate(users, start=1):
        simulation = simulate_quiz(user, quiz)
        responses = simulation["responses"]
        theta_hat, _ = quiz.fit_ability(responses)
        raw_sum = _expected_vocab_from_logits(theta_hat, difficulty_logits)
        current_estimate = quiz.estimate_with_ci(responses)
        records.append(
            SimRecord(
                user_id=user.user_id,
                true_vocab=int(user.true_vocab),
                true_theta=float(user.true_theta),
                theta_hat=float(theta_hat),
                raw_sum=float(raw_sum),
                current_point_estimate=int(current_estimate["point_estimate"]),
                total_questions=int(len(responses)),
                phase1_total=int(simulation["phase1_total"]),
                phase2_total=int(simulation["phase2_total"]),
                bucket=_bucket_name(user.true_vocab),
            )
        )
        if not quiet and (idx == n_users or idx % 100 == 0):
            print(f"simulated {idx}/{n_users} users", file=sys.stderr)

    return records


def fit_linear(records: Sequence[SimRecord]) -> tuple[float, float]:
    x = np.array([r.raw_sum for r in records], dtype=float)
    y = np.array([r.true_vocab for r in records], dtype=float)
    beta_1, beta_0 = np.polyfit(x, y, deg=1)
    return float(beta_0), float(beta_1)


def fit_power(records: Sequence[SimRecord]) -> tuple[float, float]:
    x = np.array([max(r.raw_sum, 1e-9) for r in records], dtype=float)
    y = np.array([max(r.true_vocab, 1e-9) for r in records], dtype=float)
    alpha, log_beta = np.polyfit(np.log(x), np.log(y), deg=1)
    return float(math.exp(log_beta)), float(alpha)


def clipped(value: float, ceiling: float = float(DEFAULT_CONFIG.calibration_ceiling)) -> float:
    return max(0.0, min(float(value), ceiling))


def make_results(records: Sequence[SimRecord]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for scale in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
        predictor = lambda r, s=scale: estimate_scale(r.raw_sum, s)
        results.append(
            {
                "method": "A",
                "name": f"linear_scale_{scale:.2f}",
                "params": {"scale_factor": scale},
                "summary": metric_summary(records, predictor),
                "bucket_errors": bucket_summaries(records, predictor),
                "predictor": predictor,
            }
        )

    method_b = lambda r: estimate_scale(r.raw_sum, theta_piecewise_scale(r.theta_hat))
    results.append(
        {
            "method": "B",
            "name": "theta_piecewise_scale",
            "params": {"theta<-1": 1.0, "-1<=theta<=1": 0.8, "theta>1": 0.65},
            "summary": metric_summary(records, method_b),
            "bucket_errors": bucket_summaries(records, method_b),
            "predictor": method_b,
        }
    )

    beta_0, beta_1 = fit_linear(records)
    linear = lambda r: clipped(beta_0 + beta_1 * r.raw_sum)
    results.append(
        {
            "method": "C",
            "name": "ols_linear_raw_sum",
            "params": {"beta_0": beta_0, "beta_1": beta_1},
            "summary": metric_summary(records, linear),
            "bucket_errors": bucket_summaries(records, linear),
            "predictor": linear,
        }
    )

    beta, alpha = fit_power(records)
    power = lambda r: clipped(beta * (max(r.raw_sum, 1e-9) ** alpha))
    results.append(
        {
            "method": "C",
            "name": "power_raw_sum",
            "params": {"beta": beta, "alpha": alpha},
            "summary": metric_summary(records, power),
            "bucket_errors": bucket_summaries(records, power),
            "predictor": power,
        }
    )

    return results


def best_result(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return min(results, key=lambda r: (r["summary"]["mae"], abs(r["summary"]["mean_bias"])))


def fmt_float(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_report(
    *,
    stage_vocab_path: Path,
    search_records: Sequence[SimRecord],
    validation_records: Sequence[SimRecord],
    search_results: Sequence[dict[str, Any]],
    validation_result: dict[str, Any],
    validation_all_results: Sequence[dict[str, Any]],
    output_records_json: Path,
) -> str:
    best = best_result(search_results)
    mean_phase2_search = float(np.mean([r.phase2_total for r in search_records])) if search_records else 0.0
    mean_phase2_validation = float(np.mean([r.phase2_total for r in validation_records])) if validation_records else 0.0

    result_rows = []
    for result in search_results:
        summary = result["summary"]
        result_rows.append(
            [
                result["method"],
                result["name"],
                fmt_float(summary["mae"]),
                fmt_float(summary["rmse"]),
                fmt_float(summary["r2"], 6),
                fmt_float(summary["mean_bias"]),
                json.dumps(result["params"], ensure_ascii=False),
            ]
        )

    bucket_rows = [
        [
            row["bucket"],
            row["n"],
            fmt_float(row["mae"]),
            fmt_float(row["r2"], 6),
            fmt_float(row["mean_bias"]),
        ]
        for row in validation_result["bucket_errors"]
    ]
    validation_rows = []
    for result in validation_all_results:
        summary = result["summary"]
        validation_rows.append(
            [
                result["method"],
                result["name"],
                fmt_float(summary["mae"]),
                fmt_float(summary["rmse"]),
                fmt_float(summary["r2"], 6),
                fmt_float(summary["mean_bias"]),
            ]
        )

    validation_summary = validation_result["summary"]
    lines = [
        "# v2_clusterv1 Calibration Search Results",
        "",
        f"- Stage vocab: `{stage_vocab_path}`",
        f"- Search users: {len(search_records)}",
        f"- Validation users: {len(validation_records)}",
        f"- Mean Phase 2 questions: search={mean_phase2_search:.3f}, validation={mean_phase2_validation:.3f}",
        f"- Cached simulation records: `{output_records_json}`",
        "",
        "## Current Calibration Logic",
        "",
        "- `StratifiedQuiz._vocab_at_theta(theta)` sums `P(known | theta)` over every staged word, then applies the hard-coded empirical multiplier `total * 0.8`.",
        "- `estimate_with_ci(responses)` fits `theta`, calls `_vocab_at_theta()` for the point and CI endpoints, then passes those values into `_calibrate()`.",
        f"- `_calibrate()` applies `calibration_native_max * tanh(calibration_k * estimate)`, currently `{DEFAULT_CONFIG.calibration_native_max} * tanh({DEFAULT_CONFIG.calibration_k} * estimate)`.",
        f"- If `enable_piecewise_calibration` is true, `_piecewise_calibrate()` applies the configured knots `{DEFAULT_CONFIG.piecewise_knots}` after the tanh stage.",
        "",
        "Hard-coded or empirical parameters in this path:",
        "",
        markdown_table(
            ["Location", "Parameter", "Value"],
            [
                ["`StratifiedQuiz._vocab_at_theta`", "raw scale", "`0.8`"],
                ["`EstimatorConfig.calibration_native_max`", "tanh ceiling", DEFAULT_CONFIG.calibration_native_max],
                ["`EstimatorConfig.calibration_k`", "tanh rate", DEFAULT_CONFIG.calibration_k],
                ["`EstimatorConfig.enable_piecewise_calibration`", "piecewise on/off", DEFAULT_CONFIG.enable_piecewise_calibration],
                ["`EstimatorConfig.piecewise_knots`", "post-tanh slopes", f"`{DEFAULT_CONFIG.piecewise_knots}`"],
                ["`StratifiedQuiz._mle_theta`", "MAP prior variance", "`2.0`"],
                ["`StratifiedQuiz._mle_theta`", "theta clamp", "`[-10.0, 10.0]`"],
                ["`StratifiedQuiz.fit_ability`", "CI multiplier", "`1.96`"],
                ["`StratifiedQuiz.phase2_sample`", "default n_per_class", "`4`"],
                ["`tests.simulation_eval.simulate_quiz`", "phase2_n_per_class", "`8`"],
            ],
        ),
        "",
        "Note: `tests/simulation_eval.py` stores calibrated output as `point_estimate`, but its top-level summary currently computes MAE/R²/bias from uncalibrated `estimated_vocab`. This report recomputes all metrics from calibrated predictions.",
        "",
        "## Search Results",
        "",
        markdown_table(
            ["Method", "Name", "MAE", "RMSE", "R²", "Bias", "Params"],
            result_rows,
        ),
        "",
        "## Recommendation",
        "",
        f"Best search result by MAE: `{best['name']}` with params `{json.dumps(best['params'], ensure_ascii=False)}`.",
        "",
        "## 2000 User Validation",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["MAE", fmt_float(validation_summary["mae"])],
                ["RMSE", fmt_float(validation_summary["rmse"])],
                ["R²", fmt_float(validation_summary["r2"], 6)],
                ["Correlation", fmt_float(validation_summary["correlation"], 6)],
                ["Mean bias", fmt_float(validation_summary["mean_bias"])],
            ],
        ),
        "",
        "Bucket errors for the recommended calibration:",
        "",
        markdown_table(["Bucket", "n", "MAE", "R²", "Bias"], bucket_rows),
        "",
        "Validation cross-check for every searched calibration:",
        "",
        markdown_table(
            ["Method", "Name", "MAE", "RMSE", "R²", "Bias"],
            validation_rows,
        ),
        "",
    ]
    return "\n".join(lines)


def serializable_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for result in results:
        output.append(
            {
                "method": result["method"],
                "name": result["name"],
                "params": result["params"],
                "summary": result["summary"],
                "bucket_errors": result["bucket_errors"],
            }
        )
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search v2 calibration parameters without editing StratifiedQuiz.")
    parser.add_argument("--stage-vocab", type=Path, default=DEFAULT_STAGE_VOCAB)
    parser.add_argument("--search-users", type=int, default=500)
    parser.add_argument("--validation-users", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=4242)
    parser.add_argument("--true-min", type=int, default=1000)
    parser.add_argument("--true-max", type=int, default=15000)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--records-json", type=Path, default=DEFAULT_RECORDS_JSON)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    search_records = simulate_records(
        n_users=args.search_users,
        stage_vocab_path=args.stage_vocab,
        seed=args.seed,
        true_min=args.true_min,
        true_max=args.true_max,
        quiet=args.quiet,
    )
    search_results = make_results(search_records)
    best = best_result(search_results)

    validation_records = simulate_records(
        n_users=args.validation_users,
        stage_vocab_path=args.stage_vocab,
        seed=args.validation_seed,
        true_min=args.true_min,
        true_max=args.true_max,
        quiet=args.quiet,
    )
    validation_predictor = best["predictor"]
    validation_result = {
        "method": best["method"],
        "name": best["name"],
        "params": best["params"],
        "summary": metric_summary(validation_records, validation_predictor),
        "bucket_errors": bucket_summaries(validation_records, validation_predictor),
    }
    validation_all_results = [
        {
            "method": result["method"],
            "name": result["name"],
            "params": result["params"],
            "summary": metric_summary(validation_records, result["predictor"]),
            "bucket_errors": bucket_summaries(validation_records, result["predictor"]),
        }
        for result in search_results
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.records_json.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(
            stage_vocab_path=args.stage_vocab,
            search_records=search_records,
            validation_records=validation_records,
            search_results=search_results,
            validation_result=validation_result,
            validation_all_results=validation_all_results,
            output_records_json=args.records_json,
        ),
        encoding="utf-8",
    )
    args.records_json.write_text(
        json.dumps(
            {
                "stage_vocab": str(args.stage_vocab),
                "search_records": [asdict(r) for r in search_records],
                "validation_records": [asdict(r) for r in validation_records],
                "search_results": serializable_results(search_results),
                "validation_result": validation_result,
                "validation_all_results": validation_all_results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.quiet:
        print(f"best: {best['name']} {json.dumps(best['params'], ensure_ascii=False)}")
        print(f"validation: {json.dumps(validation_result['summary'], ensure_ascii=False)}")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
