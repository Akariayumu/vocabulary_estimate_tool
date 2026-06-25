"""为词汇量估算器运行重复随机采样检查。

默认实验：3 种样本量 x 3 种 response-noise 设置 x 100 次运行 =
900 次分组估算试验。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, variance
from typing import Any

from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig
from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "batch_verification_report.json"


@dataclass(frozen=True)
class VerificationCombo:
    """一个采样/noise 配置。"""

    name: str
    per_bucket: int
    noise: float


CLASS_TRUE_VOCAB = {
    "C": 7600,
    "F": 6100,
    "P": 4300,
    "K": 2700,
}


def known_probability(rank: int, true_vocab: int, sharpness: float = 4.0) -> float:
    """学习者知道某个频率 rank 单词的平滑概率。"""

    rank = max(1, rank)
    true_vocab = max(1, true_vocab)
    logit = sharpness * (math.log(true_vocab) - math.log(rank))
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, logit))))


def simulate_responses(
    items: list[tuple[str, int, str]],
    true_vocab: int,
    rng: random.Random,
    noise: float,
) -> list[tuple[str, bool]]:
    """为一个学习者组生成合成 known/unknown responses。"""

    responses: list[tuple[str, bool]] = []
    for word, rank, _bucket in items:
        known = rng.random() < known_probability(rank, true_vocab)
        if rng.random() < noise:
            known = not known
        responses.append((word, known))
    return responses


def build_config(bootstrap_iterations: int) -> EstimatorConfig:
    """返回针对重复验证速度调优的配置。"""

    return EstimatorConfig(
        random_seed=DEFAULT_CONFIG.random_seed,
        vocab_size=DEFAULT_CONFIG.vocab_size,
        min_vocab_size=DEFAULT_CONFIG.min_vocab_size,
        bucket_boundaries=DEFAULT_CONFIG.bucket_boundaries,
        levels=DEFAULT_CONFIG.levels,
        transition_margin=DEFAULT_CONFIG.transition_margin,
        bootstrap_iterations=bootstrap_iterations,
        confidence_interval=DEFAULT_CONFIG.confidence_interval,
        confidence_high_ratio=DEFAULT_CONFIG.confidence_high_ratio,
        confidence_mid_ratio=DEFAULT_CONFIG.confidence_mid_ratio,
        logistic_l2=DEFAULT_CONFIG.logistic_l2,
        logistic_max_iter=DEFAULT_CONFIG.logistic_max_iter,
        logistic_lr=DEFAULT_CONFIG.logistic_lr,
        default_sample_per_bucket=DEFAULT_CONFIG.default_sample_per_bucket,
        adaptive_boundary_rate=DEFAULT_CONFIG.adaptive_boundary_rate,
        adaptive_focus_width=DEFAULT_CONFIG.adaptive_focus_width,
        ordered_classes=DEFAULT_CONFIG.ordered_classes,
        coverage_targets=DEFAULT_CONFIG.coverage_targets,
        abbreviation_max_len=DEFAULT_CONFIG.abbreviation_max_len,
        min_word_len=DEFAULT_CONFIG.min_word_len,
        fallback_rank_step=DEFAULT_CONFIG.fallback_rank_step,
    )


def run_combo(
    combo: VerificationCombo,
    runs: int,
    base_seed: int,
    config: EstimatorConfig,
    vocab_bank: VocabBank,
) -> dict[str, Any]:
    """针对一个组合运行重复模拟并聚合统计量。"""

    group_estimates = {group: [] for group in config.ordered_classes}
    adjusted_estimates = {group: [] for group in config.ordered_classes}
    consistency_flags: list[bool] = []
    sample_sizes: list[int] = []

    for run_idx in range(runs):
        seed = base_seed + run_idx + combo.per_bucket * 1000 + int(combo.noise * 10000)
        rng = random.Random(seed)
        sampler = VocabularySampler(vocab_bank, config, seed=seed)
        estimator = VocabEstimator(vocab_bank, config, seed=seed)
        test_items = sampler.balanced_sample(per_bucket=combo.per_bucket)

        grouped = {
            group: simulate_responses(test_items, true_vocab, rng, combo.noise)
            for group, true_vocab in CLASS_TRUE_VOCAB.items()
        }
        result = estimator.estimate_groups(grouped)
        consistency = result["ordering_consistency"]
        consistency_flags.append(bool(consistency["was_consistent"]))
        sample_sizes.append(len(test_items))

        for group in config.ordered_classes:
            row = result["classes"][group]
            group_estimates[group].append(int(row["point_estimate"]))
            adjusted_estimates[group].append(int(row.get("order_adjusted_estimate", row["point_estimate"])))

    return {
        "combo": asdict(combo),
        "runs": runs,
        "sample_size_mean": mean(sample_sizes) if sample_sizes else 0,
        "ordering_consistency_rate": mean(consistency_flags) if consistency_flags else 0.0,
        "groups": {
            group: summarize_values(values, adjusted_estimates[group], CLASS_TRUE_VOCAB[group])
            for group, values in group_estimates.items()
        },
    }


def summarize_values(values: list[int], adjusted: list[int], true_vocab: int) -> dict[str, Any]:
    """返回一个组的均值/方差/误差指标。"""

    if not values:
        return {
            "true_vocab": true_vocab,
            "mean": 0,
            "variance": 0,
            "mean_absolute_error": 0,
            "adjusted_mean": 0,
            "min": 0,
            "max": 0,
        }
    return {
        "true_vocab": true_vocab,
        "mean": round(mean(values), 2),
        "variance": round(variance(values), 2) if len(values) > 1 else 0.0,
        "mean_absolute_error": round(mean(abs(value - true_vocab) for value in values), 2),
        "adjusted_mean": round(mean(adjusted), 2),
        "min": min(values),
        "max": max(values),
    }


def run_verification(
    runs: int = 100,
    base_seed: int = 2026,
    bootstrap_iterations: int = 0,
) -> dict[str, Any]:
    """运行完整 9 组合验证套件。"""

    config = build_config(bootstrap_iterations)
    vocab_bank = VocabBank(config)
    combos = [
        VerificationCombo(name=f"sample{per_bucket}_noise{noise:.2f}", per_bucket=per_bucket, noise=noise)
        for per_bucket in (4, 8, 12)
        for noise in (0.00, 0.05, 0.10)
    ]

    combo_reports = [
        run_combo(combo, runs=runs, base_seed=base_seed, config=config, vocab_bank=vocab_bank)
        for combo in combos
    ]
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total_trials": len(combos) * runs,
        "runs_per_combo": runs,
        "combo_count": len(combos),
        "class_true_vocab": CLASS_TRUE_VOCAB,
        "vocab_bank": {
            "size": len(vocab_bank),
            "used_fallback": vocab_bank.used_fallback,
            "bucket_sizes": vocab_bank.bucket_sizes(),
        },
        "config": {
            "bootstrap_iterations": bootstrap_iterations,
            "sample_sizes": [4, 8, 12],
            "noise_rates": [0.0, 0.05, 0.10],
        },
        "overall_ordering_consistency_rate": round(
            mean(report["ordering_consistency_rate"] for report in combo_reports),
            4,
        ),
        "combos": combo_reports,
    }


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 900-sample vocabulary estimator verification.")
    parser.add_argument("--runs", type=int, default=100, help="Runs per combo. Default: 100")
    parser.add_argument("--seed", type=int, default=2026, help="Base random seed.")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=0,
        help="Bootstrap iterations per estimate. Default 0 keeps the 900-run verifier fast.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help="Report JSON path. Default: reports/batch_verification_report.json",
    )
    args = parser.parse_args()

    report = run_verification(
        runs=args.runs,
        base_seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    write_report(report, output)

    print(f"wrote: {display_path(output)}")
    print(f"total_trials: {report['total_trials']}")
    print(f"overall_ordering_consistency_rate: {report['overall_ordering_consistency_rate']}")


if __name__ == "__main__":
    main()
