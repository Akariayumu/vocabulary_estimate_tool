#!/usr/bin/env python3
"""Explore Phase-1 question counts for the stratified Rasch quiz.

The experiment reuses the existing synthetic-user framework:

* generate Rasch users with true vocabulary in a configurable range
* draw one 40-question StratifiedQuiz Phase 1 sample per user
* score the prefixes [10, 15, 20, 25, 30, 35, 40]
* fit ability from each prefix only, with no Phase 2 refinement

Outputs:

* JSON with summaries and per-user records
* Markdown table for reports
* optional PNG line chart when matplotlib is installed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.simulation_eval import (  # noqa: E402
    DEFAULT_STAGE_VOCAB,
    _bucket_errors,
    _bucket_name,
    _difficulty_logits,
    _expected_vocab_from_logits,
    _metrics,
    generate_synthetic_users,
    load_vocab_bank,
    sigmoid,
    _logit,
)
from vocab_estimator.stratified_quiz import StratifiedQuiz  # noqa: E402


DEFAULT_COUNTS = [10, 15, 20, 25, 30, 35, 40]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "question_count_exploration.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "outputs" / "question_count_exploration.md"
DEFAULT_PLOT = PROJECT_ROOT / "outputs" / "question_count_exploration.png"

# Front-load broad difficulty coverage for streaming prefixes. The first 15
# questions cover easy/hard extremes plus the middle bands; the first 20 cover
# all cluster_20 classes once; questions 21-40 add the second item per class.
STREAMING_CLUSTER_ORDER = [0, 19, 5, 15, 10, 2, 7, 12, 17, 4, 9, 14, 18, 1, 6, 11, 16, 3, 8, 13]


def _answer_item(user: Any, item: dict[str, Any], rng: random.Random) -> tuple[str, bool]:
    """Generate one Rasch-model response for a synthetic user."""
    d_logit = _logit(max(0.001, min(0.999, float(item["difficulty"]))))
    p_known = sigmoid(user.true_theta - d_logit)
    return item["word"], rng.random() < p_known


def _order_phase1_items(items: Sequence[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    """Return Phase-1 items in a prefix-friendly or original order."""
    if policy == "shuffled":
        return list(items)
    if policy != "streaming":
        raise ValueError(f"unknown order policy: {policy}")

    by_cluster: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        by_cluster.setdefault(int(item["cluster_20"]), []).append(item)

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append_wave() -> None:
        for c20 in STREAMING_CLUSTER_ORDER:
            bucket = by_cluster.get(c20, [])
            while bucket and bucket[0]["word"] in seen:
                bucket.pop(0)
            if not bucket:
                continue
            item = bucket.pop(0)
            ordered.append(item)
            seen.add(item["word"])

    append_wave()
    append_wave()

    # Defensive fallback in case a future sampler returns non-2-per-class data.
    for item in items:
        if item["word"] not in seen:
            ordered.append(item)
            seen.add(item["word"])
    return ordered


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.array(values, dtype=float))) if values else 0.0


def _summarize_count(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute quality and uncertainty diagnostics for one question count."""
    metrics = _metrics(records)
    vocab_widths = [r["vocab_ci_high"] - r["vocab_ci_low"] for r in records]
    theta_widths = [r["theta_ci_high"] - r["theta_ci_low"] for r in records]
    abs_errors = [abs(r["bias"]) for r in records]
    return {
        **metrics,
        "n_users": len(records),
        "mean_vocab_ci_width": round(_mean(vocab_widths), 3),
        "mean_theta_ci_width": round(_mean(theta_widths), 4),
        "median_abs_error": round(float(np.median(abs_errors)), 3) if abs_errors else 0.0,
        "p90_abs_error": round(float(np.percentile(abs_errors, 90)), 3) if abs_errors else 0.0,
        "bucket_errors": _bucket_errors(records),
    }


def run_experiment(
    *,
    n_users: int = 300,
    counts: Sequence[int] = DEFAULT_COUNTS,
    seed: int = 42,
    true_min: int = 1000,
    true_max: int = 15000,
    adaptive: bool = True,
    order_policy: str = "streaming",
    stage_vocab_path: str | Path = DEFAULT_STAGE_VOCAB,
    output: str | Path = DEFAULT_OUTPUT,
    markdown: str | Path = DEFAULT_MARKDOWN,
    plot: str | Path | None = DEFAULT_PLOT,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the question-count exploration and write result artifacts."""
    t0 = time.time()
    counts = sorted({int(c) for c in counts})
    if not counts:
        raise ValueError("counts must not be empty")
    if min(counts) < 3:
        raise ValueError("all question counts must be >= 3")
    if max(counts) > 40:
        raise ValueError("this experiment uses Phase 1 prefixes, so counts must be <= 40")

    vocab_bank = load_vocab_bank(stage_vocab_path)
    difficulty_logits = _difficulty_logits(vocab_bank)
    quiz = StratifiedQuiz(stage_vocab_path=stage_vocab_path)
    users = list(
        generate_synthetic_users(
            n_users=n_users,
            vocab_bank=vocab_bank,
            true_min=true_min,
            true_max=true_max,
            seed=seed,
        )
    )

    records_by_count: dict[int, list[dict[str, Any]]] = {count: [] for count in counts}

    for idx, user in enumerate(users, start=1):
        sample_rng = random.Random(user.seed)
        response_rng = random.Random(user.seed ^ 0x9E3779B9)

        phase1_items = _order_phase1_items(
            quiz.phase1_sample(adaptive=adaptive, rng=sample_rng),
            order_policy,
        )
        all_responses = [_answer_item(user, item, response_rng) for item in phase1_items]

        for count in counts:
            responses = all_responses[:count]
            theta_hat, theta_ci = quiz.fit_ability(responses)
            estimated_vocab = int(round(_expected_vocab_from_logits(theta_hat, difficulty_logits)))
            vocab_ci_low = int(round(_expected_vocab_from_logits(theta_ci[0], difficulty_logits)))
            vocab_ci_high = int(round(_expected_vocab_from_logits(theta_ci[1], difficulty_logits)))
            bias = int(estimated_vocab - user.true_vocab)

            records_by_count[count].append(
                {
                    "user_id": int(user.user_id),
                    "question_count": int(count),
                    "true_theta": round(float(user.true_theta), 6),
                    "true_vocab": int(user.true_vocab),
                    "estimated_vocab": estimated_vocab,
                    "bias": bias,
                    "abs_error": abs(bias),
                    "theta": round(float(theta_hat), 6),
                    "theta_ci_low": round(float(theta_ci[0]), 6),
                    "theta_ci_high": round(float(theta_ci[1]), 6),
                    "vocab_ci_low": vocab_ci_low,
                    "vocab_ci_high": vocab_ci_high,
                    "correct_count": int(sum(known for _, known in responses)),
                    "bucket": _bucket_name(user.true_vocab),
                }
            )

        if not quiet and (idx == n_users or idx % 50 == 0):
            print(f"simulated {idx}/{n_users} users", file=sys.stderr)

    summaries = {
        str(count): _summarize_count(records)
        for count, records in records_by_count.items()
    }

    result = {
        "metadata": {
            "simulation_model": "rasch_theta",
            "response_probability": "sigmoid(true_theta - logit(difficulty))",
            "question_policy": "prefixes of one StratifiedQuiz.phase1_sample per user",
            "order_policy": order_policy,
            "phase2": "disabled",
            "adaptive_phase1": bool(adaptive),
            "n_users": int(n_users),
            "seed": int(seed),
            "true_min": int(true_min),
            "true_max": int(true_max),
            "counts": list(counts),
            "vocab_bank_size": len(vocab_bank),
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "summaries": summaries,
        "records_by_count": {str(k): v for k, v in records_by_count.items()},
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    markdown_path = Path(markdown)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_text = build_markdown_table(result)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    if plot:
        maybe_write_plot(result, Path(plot))

    if not quiet:
        print(markdown_text)
        print(f"wrote {output_path}")
        print(f"wrote {markdown_path}")
        if plot:
            print(f"plot: {plot}")

    return result


def build_markdown_table(result: dict[str, Any]) -> str:
    """Build a compact Markdown table from experiment summaries."""
    lines = [
        "| 题量 | MAE | RMSE | R² | Corr | 平均偏差 | θ CI宽度 | 词汇CI宽度 | P90误差 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in result["metadata"]["counts"]:
        row = result["summaries"][str(count)]
        lines.append(
            "| {count} | {mae:.0f} | {rmse:.0f} | {r2:.4f} | {corr:.4f} | "
            "{bias:.0f} | {theta_width:.3f} | {vocab_width:.0f} | {p90:.0f} |".format(
                count=count,
                mae=row["mae"],
                rmse=row["rmse"],
                r2=row["r2"],
                corr=row["correlation"],
                bias=row["mean_bias"],
                theta_width=row["mean_theta_ci_width"],
                vocab_width=row["mean_vocab_ci_width"],
                p90=row["p90_abs_error"],
            )
        )
    return "\n".join(lines) + "\n"


def maybe_write_plot(result: dict[str, Any], path: Path) -> None:
    """Write MAE/R² line chart if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"skip plot: matplotlib unavailable ({exc})", file=sys.stderr)
        return

    counts = result["metadata"]["counts"]
    maes = [result["summaries"][str(c)]["mae"] for c in counts]
    r2s = [result["summaries"][str(c)]["r2"] for c in counts]

    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax1.plot(counts, maes, marker="o", color="#1f77b4", label="MAE")
    ax1.set_xlabel("Phase 1 question count")
    ax1.set_ylabel("MAE (words)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(counts, r2s, marker="s", color="#d62728", label="R²")
    ax2.set_ylabel("R²", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(min(r2s) - 0.01, min(1.0, max(r2s) + 0.01))

    fig.suptitle("StratifiedQuiz Phase 1 Question Count Exploration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-users", type=int, default=300, help="Number of synthetic users.")
    parser.add_argument("--counts", type=int, nargs="+", default=DEFAULT_COUNTS, help="Question counts to test.")
    parser.add_argument("--seed", type=int, default=42, help="Synthetic-user random seed.")
    parser.add_argument("--true-min", type=int, default=1000, help="Minimum true vocabulary size.")
    parser.add_argument("--true-max", type=int, default=15000, help="Maximum true vocabulary size.")
    parser.add_argument("--stage-vocab", type=Path, default=DEFAULT_STAGE_VOCAB, help="Path to stage_vocab.json.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN, help="Output Markdown table path.")
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT, help="Output PNG path.")
    parser.add_argument("--no-plot", action="store_true", help="Do not attempt to generate a plot.")
    parser.add_argument("--balanced", action="store_true", help="Use non-adaptive balanced Phase 1 sampling.")
    parser.add_argument(
        "--order-policy",
        choices=["streaming", "shuffled"],
        default="streaming",
        help="Question order before taking prefixes.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress and table output.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_experiment(
        n_users=args.n_users,
        counts=args.counts,
        seed=args.seed,
        true_min=args.true_min,
        true_max=args.true_max,
        adaptive=not args.balanced,
        order_policy=args.order_policy,
        stage_vocab_path=args.stage_vocab,
        output=args.output,
        markdown=args.markdown,
        plot=None if args.no_plot else args.plot,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
