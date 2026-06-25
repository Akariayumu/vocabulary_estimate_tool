"""词汇量估算器配置。

所有阈值集中在这里，便于实验调参而不需要改动实现模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EstimatorConfig:
    """词库构建与估算流程的运行时配置。"""

    random_seed: int = 0  # 0 = 使用系统熵，非零值 = 固定随机种子以便复现
    vocab_size: int = 30_000
    min_vocab_size: int = 20_000

    # bucket 标签表示“频率排名不超过该边界的词”。实际区间为
    # 实际区间：(previous_boundary, boundary]。
    bucket_boundaries: tuple[int, ...] = (
        1_000,
        2_000,
        3_000,
        5_000,
        8_000,
        10_000,
        15_000,
        20_000,
        30_000,
    )

    levels: tuple[tuple[str, int, int | None], ...] = (
        ("初中", 1_500, 2_500),
        ("高中", 2_500, 4_000),
        ("四级", 4_000, 5_500),
        ("六级", 5_500, 7_500),
        ("六级+ / 考研", 7_500, 10_000),
        ("专业/母语级", 10_000, None),
    )
    transition_margin: int = 250

    bootstrap_iterations: int = 300
    confidence_interval: float = 0.90
    confidence_high_ratio: float = 0.20
    confidence_mid_ratio: float = 0.40

    logistic_l2: float = 1.0
    logistic_max_iter: int = 800
    logistic_lr: float = 0.05

    default_sample_per_bucket: int = 12
    adaptive_boundary_rate: float = 0.50
    adaptive_focus_width: float = 0.25

    ordered_classes: tuple[str, ...] = ("C", "F", "P", "K")

    coverage_targets: tuple[float, float] = (0.95, 0.98)

    # 基于公开研究的词汇量校准
    # Nation (2006)：母语者约 20,000 个 word families
    # Goulden, Nation & Read (1990)：大学毕业生约 20,000
    # Milton (2009)：中国 EFL 学习者上限约 9,000-10,000
    # CET-6 阈值：约 6,000-7,000；TEM-8 阈值：约 10,000-13,000
    calibration_native_max: int = 20_000
    calibration_ceiling: int = 22_000
    calibration_k: float = 0.0000691  # tanh 饱和速率
    #   k=0.0000691：calibrated = max_v * tanh(k * raw)，再做分段校准
    #   raw 21303 (all-correct) → cal 18049   (母语者水平)
    #   raw 10000 → cal 10335   (专业/母语级)
    #   raw 8000  → cal  7877   (六级+ / 考研)
    #   raw 6000  → cal  5181   (四级)
    #   raw 4000  → cal  4076   (高中/四级过渡)
    #   raw 2000  → cal  2747   (初中/高中过渡)

    abbreviation_max_len: int = 5
    min_word_len: int = 2
    fallback_rank_step: int = 100

    # 用于 bucket 已知率 Bayesian smoothing 的 Beta prior 参数
    # 每个 bucket 的 prior 会按归一化到 [0, 1] 的 bucket 索引，在
    # beta_prior_max_knowledge（高频 / 低 rank）和
    # beta_prior_min_knowledge（低频 / 高 rank）之间插值。
    # bucket_alpha 公式 = prior_alpha_base + interpolated_knowledge_counts
    # bucket_beta  公式 = prior_beta_base  + (1 - interpolated_knowledge_counts)
    beta_prior_alpha_base: float = 0.5
    beta_prior_beta_base: float = 0.5
    beta_prior_max_knowledge: float = 4.0
    beta_prior_min_knowledge: float = 0.5

    # --- Weighted logistic fitting（方案 A）---
    # 启用后，每个 (word, known) 样本会按频率 rank 加权。
    # 高 rank（罕见）词权重更低，让模型优先拟合常见词。
    # 这会得到更保守的估算。
    enable_weighted_fitting: bool = True

    # 权重公式：w = 1 / (1 + log2(max(rank, 10) / 10))
    # 仅在 enable_weighted_fitting 为 True 时生效。

    # --- 分段线性校准（方案 B）---
    # 在 tanh 阶段之后应用。将中段（3000-8000）压缩 55%，
    # 让约四级水平的学生落在 4000-5000 左右；再把高段放大 28%，
    # 使全对结果可达到约 18000。
    # 分段：
    #   [0, 3000]    → slope 1.00（恒等）
    #   [3000, 8000]  → slope 0.45（中段压缩）
    #   [8000, 22000] → slope 1.28（高段增强）
    enable_piecewise_calibration: bool = True

    # 每项为 (upper_boundary, slope_for_segment)。
    # 第一个分段始终从 0 开始，并使用第一项的 slope。
    piecewise_knots: tuple[tuple[int, float], ...] = (
        (3_000, 1.00),
        (8_000, 0.45),
        (22_000, 1.28),
    )

    # 两阶段 adaptive testing 参数
    phase1_question_count: int = 30
    enable_phase2: bool = False
    stage2_boundary_low: float = 0.20
    stage2_boundary_high: float = 0.80
    stage2_extra_per_bucket: int = 8


DEFAULT_CONFIG = EstimatorConfig()


def bucket_label(boundary: int) -> str:
    """返回 rank bucket 边界的紧凑显示标签。"""

    if boundary % 1000 == 0:
        return f"{boundary // 1000}k"
    return str(boundary)


def bucket_beta_prior(bucket_index: int, n_buckets: int, config: EstimatorConfig = DEFAULT_CONFIG) -> tuple[float, float]:
    """返回频率 bucket 的 (alpha, beta) Beta prior 伪计数。

    低索引 bucket（高频 → 低 rank）使用偏“已知”的 prior。
    高索引 bucket（低频 → 高 rank）使用偏“未知”的 prior。
    插值按 bucket 索引线性进行。
    """
    t = bucket_index / max(n_buckets - 1, 1)
    knowledge_pseudo = config.beta_prior_max_knowledge + t * (
        config.beta_prior_min_knowledge - config.beta_prior_max_knowledge
    )
    ignorance_pseudo = config.beta_prior_min_knowledge + t * (
        config.beta_prior_max_knowledge - config.beta_prior_min_knowledge
    )
    alpha = config.beta_prior_alpha_base + knowledge_pseudo
    beta = config.beta_prior_beta_base + ignorance_pseudo
    return alpha, beta


BUCKET_LABELS = tuple(bucket_label(b) for b in DEFAULT_CONFIG.bucket_boundaries)
