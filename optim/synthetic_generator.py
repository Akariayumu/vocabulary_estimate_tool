"""用于校准流程验证的合成测试 response 生成器。

生成一组合成测试者，其 (α, β) 参数已知，然后在分层词表上渲染二元 responses。
生成的数据可输入 ``calibration_trainer``，用于验证能否从数据中恢复原始全局参数
（β、k、piecewise_knots）。"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vocab_estimator.config import DEFAULT_CONFIG, EstimatorConfig
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.vocab_model import VocabEstimator


# ---------------------------------------------------------------------------
# 已知的 “ground truth” 参数
# ---------------------------------------------------------------------------

TRUE_BETA = -0.285
TRUE_K = 0.0000792
TRUE_KNOTS = [(3000, 1.0), (8000, 0.42), (22000, 1.35)]
TRUE_MAX_V = 20000.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def piecewise_calibrate(x: float, knots: list[tuple[int, float]]) -> float:
    if x <= 0:
        return x
    prev_boundary = 0.0
    prev_value = 0.0
    for boundary, slope in knots:
        if x <= boundary:
            return prev_value + (x - prev_boundary) * slope
        prev_value += (float(boundary) - prev_boundary) * slope
        prev_boundary = float(boundary)
    return prev_value + (x - prev_boundary) * knots[-1][1]


def calibrate_vocab(raw: float) -> float:
    """应用 TRUE 校准参数。"""
    cal = TRUE_MAX_V * math.tanh(TRUE_K * raw)
    return piecewise_calibrate(cal, TRUE_KNOTS)


# ---------------------------------------------------------------------------
# 合成用户生成
# ---------------------------------------------------------------------------

def generate_user(
    user_id: int,
    alpha: float,
    bank: VocabBank,
    sampler: VocabularySampler,
    questions_per_user: int = 80,
    noise: float = 0.02,
    rng: random.Random | None = None,
) -> list[tuple[str, bool]]:
    """生成一个合成用户的 responses。

    Args:
        alpha: 每个用户的 intercept（能力参数）。
        bank: 词库。
        sampler: 用于分层选词。
        questions_per_user: 该用户的测试题数。
        noise: 标签翻转概率（模拟粗心错误）。
        rng: 用于复现的随机状态。

    Returns:
        [(word, known), ...]"""
    if rng is None:
        rng = random.Random(42 + user_id)

    # 为该用户采样分层词表
    items = sampler.balanced_sample(per_bucket=4)
    # 需要时获取更多词
    while len(items) < questions_per_user:
        extra = sampler.balanced_sample(per_bucket=2)
        items.extend(extra)

    # shuffle 并裁剪
    rng.shuffle(items)
    items = items[:questions_per_user]

    responses: list[tuple[str, bool]] = []
    for word, rank, bucket in items:
        log_rank = math.log(max(rank, 1))
        logit = alpha + TRUE_BETA * log_rank
        prob = 1.0 / (1.0 + math.exp(-logit))
        # 应用 noise
        prob = (1.0 - noise) * prob + noise * 0.5
        known = rng.random() < prob
        responses.append((word, known))

    return responses


def generate_population(
    bank: VocabBank,
    n_users: int = 100,
    questions_per_user: int = 80,
    seed: int = 42,
) -> dict[int, list[tuple[str, bool]]]:
    """生成一组合成测试者。

    α 值从 -5（低能力，约 1500 词）到 +3（高能力，约 15000 词）均匀分布。
    在 TRUE_BETA = -0.285 时，这对应约 2500 到 21000 的 raw estimates。

    分布：
        20% 低能力    (α ∈ [-5.0, -3.0])
        40% 中能力    (α ∈ [-3.0,  0.0])
        30% 高能力    (α ∈ [ 0.0,  2.0])
        10% 很高能力  (α ∈ [ 2.0,  3.5])"""
    rng = random.Random(seed)
    sampler = VocabularySampler(bank, DEFAULT_CONFIG, seed=seed)

    population: dict[int, list[tuple[str, bool]]] = {}

    # 分层 α 生成
    alpha_ranges = [
        (-5.0, -3.0, int(n_users * 0.20)),
        (-3.0, 0.0, int(n_users * 0.40)),
        (0.0, 2.0, int(n_users * 0.30)),
        (2.0, 3.5, n_users - int(n_users * 0.90)),
    ]

    uid = 0
    for low, high, count in alpha_ranges:
        for _ in range(count):
            if uid >= n_users:
                break
            alpha = low + (high - low) * rng.random()
            responses = generate_user(
                uid, alpha, bank, sampler,
                questions_per_user=questions_per_user,
                noise=0.02,
                rng=random.Random(seed + uid),
            )
            population[uid] = responses
            uid += 1

    return population


# ---------------------------------------------------------------------------
# 验证运行器
# ---------------------------------------------------------------------------

def synthetic_validation(
    bank: VocabBank,
    bucket_labels: list[str],
    n_users: int = 50,
) -> dict[str, Any]:
    """运行合成验证：生成数据、训练模型并检查参数恢复精度。

    返回包含恢复参数的 summary metrics。"""
    from .calibration_trainer import train_torch, train_numpy, load_responses

    print(f"Generating {n_users} synthetic users...")
    population = generate_population(bank, n_users=n_users, questions_per_user=80)

    # 保存到临时文件，再加载回来（保持 API 边界干净）
    tmp_path = Path("/tmp/synthetic_calibration_data.json")
    json_data = {
        "users": [
            {"user_id": uid, "responses": [{"word": w, "known": k} for w, k in resp]}
            for uid, resp in population.items()
        ]
    }
    tmp_path.write_text(json.dumps(json_data, ensure_ascii=False))

    print("Training...")
    params = train_numpy(load_responses(str(tmp_path)), bank, bucket_labels,
                         n_epochs=500, lr=0.001)

    print("\n=== Validation Results ===")
    print(f"  True β:         {TRUE_BETA:.6f}")
    print(f"  Recovered β:    {params.beta:.6f}")
    print(f"  β error:        {abs(params.beta - TRUE_BETA):.6f}")
    print(f"  True k:         {TRUE_K:.8f}")
    print(f"  Recovered k:    {params.calibration_k:.8f}")
    print(f"  k error:        {abs(params.calibration_k - TRUE_K):.8f}")
    print(f"  True knots:     {TRUE_KNOTS}")
    print(f"  Recovered knots:{params.piecewise_knots}")
    print(f"  Training loss:  {params.training_loss:.6f}")

    beta_ok = abs(params.beta - TRUE_BETA) < 0.05
    k_ok = abs(params.calibration_k - TRUE_K) < 1e-5
    slope_errors = [
        abs(params.piecewise_knots[i][1] - TRUE_KNOTS[i][1]) < 0.1
        for i in range(len(TRUE_KNOTS))
    ]
    knots_ok = all(slope_errors)

    print(f"\n  β recovery:  {'✓' if beta_ok else '✗'}")
    print(f"  k recovery:  {'✓' if k_ok else '✗'}")
    print(f"  knots slope: {'✓' if knots_ok else '✗'}")
    print(f"  OVERALL:     {'✓ PASS' if all([beta_ok, k_ok, knots_ok]) else '✗ FAIL'}")

    tmp_path.unlink(missing_ok=True)

    return {
        "true_beta": TRUE_BETA,
        "recovered_beta": params.beta,
        "true_k": TRUE_K,
        "recovered_k": params.calibration_k,
        "true_knots": TRUE_KNOTS,
        "recovered_knots": params.piecewise_knots,
        "training_loss": params.training_loss,
        "beta_ok": beta_ok,
        "k_ok": k_ok,
        "knots_ok": knots_ok,
    }


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic calibration data.")
    parser.add_argument("--n-users", type=int, default=100, help="Number of synthetic users")
    parser.add_argument("--output", default="synthetic_dataset.json", help="Output JSON path")
    parser.add_argument("--validate", action="store_true", help="Run train-and-recover validation")
    args = parser.parse_args()

    bank = VocabBank(DEFAULT_CONFIG)

    if args.validate:
        bucket_labels = list(bank.words_by_bucket.keys())
        synthetic_validation(bank, bucket_labels, n_users=args.n_users)
        return

    population = generate_population(bank, n_users=args.n_users, questions_per_user=80)
    json_data = {
        "dataset_meta": {
            "synthetic": True,
            "n_users": args.n_users,
            "true_beta": TRUE_BETA,
            "true_k": TRUE_K,
            "true_knots": [(b, s) for b, s in TRUE_KNOTS],
        },
        "users": [
            {"user_id": uid, "responses": [{"word": w, "known": k} for w, k in resp]}
            for uid, resp in population.items()
        ],
    }
    Path(args.output).write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
    print(f"Generated {args.n_users} synthetic users → {args.output}")


# ---------------------------------------------------------------------------
# 新增：合成训练数据生成器（frequency-weighted sampling）
# ---------------------------------------------------------------------------


def build_known_vocab(
    bank: VocabBank,
    vocab_size: int,
) -> set[str]:
    """为给定 ``vocab_size`` 的虚拟测试者构建“已知词集合”。

    词从高频 bucket 开始累计。若某个 bucket 只需部分词，则取其随机子集。

    Args:
        bank: 词库。
        vocab_size: 虚拟用户的目标词汇量。

    Returns:
        已知词集合。"""
    bucket_labels_in_order = [
        "1k", "2k", "3k", "5k", "8k", "10k", "15k", "20k", "30k",
    ]
    known: set[str] = set()
    remaining = vocab_size

    for label in bucket_labels_in_order:
        items = list(bank.words_by_bucket.get(label, []))
        if not items:
            continue
        if remaining >= len(items):
            # 取整个 bucket
            for item in items:
                known.add(item.word)
            remaining -= len(items)
        else:
            # 取随机子集。使用由
            # target vocab_size 派生的确定性 seed，使子集可复现且去相关
            # 与测试题采样 rng 去相关。
            inner_rng = random.Random(vocab_size * 12345)
            chosen = inner_rng.sample(items, remaining)
            for item in chosen:
                known.add(item.word)
            remaining = 0
            break

    return known


SYNTHETIC_VOCAB_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000,
                         10000, 12000, 15000]


def generate_synthetic_data(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions: int = 100,
    power: float = 0.5,
    seed: int = 42,
) -> dict[int, dict]:
    """为校准生成合成训练数据。

    对每个 ``vocab_size``：
      1. 构建“已知词集合”（从高频 bucket 填充到 V）
      2. 使用 frequency-weighted distribution 抽样 N 道测试题
         （高频词更常出现，类似真实考试）
      3. 根据 known set 将每题标注为 known=True/False

    Args:
        bank: 词库。
        vocab_sizes: 要模拟的词汇量列表，默认 ``SYNTHETIC_VOCAB_SIZES``。
        n_questions: 每个合成用户的测试题数。
        power: 测试题采样使用的 power-law 指数。
        seed: 随机种子。

    Returns:
        {vocab_size: {"responses": [(word, known), ...],
                      "known_set_size": int,
                      "known_rate": float}, ...}"""
    if vocab_sizes is None:
        vocab_sizes = list(SYNTHETIC_VOCAB_SIZES)

    result: dict[int, dict] = {}

    for vsize in sorted(vocab_sizes):
        # 步骤 1：构建已知词集合
        known_words = build_known_vocab(bank, vsize)

        # 步骤 2：抽样测试题（frequency-weighted）
        from .interval_sampler import sample_test_questions
        sampled = sample_test_questions(bank, n=n_questions, power=power, seed=seed)

        # 步骤 3：标注每道题
        responses: list[tuple[str, bool]] = []
        for sw in sampled:
            known = sw.word in known_words
            responses.append((sw.word, known))

        known_count = sum(1 for _, k in responses if k)
        known_rate = known_count / len(responses) if responses else 0.0

        result[vsize] = {
            "responses": responses,
            "known_set_size": len(known_words),
            "known_rate": known_rate,
            "n_questions": len(responses),
        }

    return result


def describe_synthetic_dataset(
    data: dict[int, dict],
) -> str:
    """返回合成数据集的易读报告。"""
    lines = []
    lines.append(f"Synthetic training data ({len(data)} vocabulary sizes)\n")
    lines.append(f"  {'Vocab':>7s}  {'KnownSet':>9s}  {'Q':>4s}  {'Known':>7s}  {'Known%':>7s}")
    lines.append(f"  {'------':>7s}  {'--------':>9s}  {'---':>4s}  {'------':>7s}  {'-------':>7s}")
    for vsize in sorted(data):
        d = data[vsize]
        lines.append(
            f"  {vsize:>7d}  {d['known_set_size']:>9d}"
            f"  {d['n_questions']:>4d}  {d['known_rate']*d['n_questions']:>5.0f}"
            f"  {d['known_rate']:>6.1%}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 合成校准训练循环
# ---------------------------------------------------------------------------


def run_synthetic_training(
    bank: VocabBank,
    vocab_sizes: list[int] | None = None,
    n_questions: int = 100,
    power: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """运行合成训练验证。

    对每个 ``vocab_size``，生成合成测试者 responses，并输入 ``VocabEstimator`` 得到预测，
    然后报告预测词汇量与实际词汇量。

    Returns:
        每个 vocab size 一个 dict，包含预测详情。"""
    from vocab_estimator.config import DEFAULT_CONFIG
    from vocab_estimator.vocab_model import VocabEstimator

    if vocab_sizes is None:
        vocab_sizes = [2000, 5000, 8000, 10000]

    data = generate_synthetic_data(
        bank, vocab_sizes=vocab_sizes,
        n_questions=n_questions, power=power, seed=seed,
    )

    estimator = VocabEstimator(bank, DEFAULT_CONFIG, seed=seed)
    results: list[dict] = []

    for vsize in sorted(data):
        responses = data[vsize]["responses"]
        known_rate = data[vsize]["known_rate"]

        pred = estimator.estimate_single(responses)
        pred_v = pred["point_estimate"]
        loss = (pred_v - vsize) ** 2

        results.append({
            "vocab_size": vsize,
            "predicted": pred_v,
            "loss": loss,
            "known_rate": known_rate,
            "n_questions": len(responses),
            "raw_estimate": pred.get("raw_estimate"),
            "logistic_estimate": pred.get("logistic_estimate"),
        })

    return results


if __name__ == "__main__":
    main()
