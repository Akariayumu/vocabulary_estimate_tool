"""Configuration for the vocabulary estimator.

All thresholds are centralized here so experiments can tune the model without
touching the implementation modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EstimatorConfig:
    """Runtime configuration for vocabulary bank construction and estimation."""

    random_seed: int = 0  # 0 = use system entropy, non-zero = fixed for reproducibility
    vocab_size: int = 30_000
    min_vocab_size: int = 20_000

    # Bucket labels mean "words up to this frequency rank". The actual bucket
    # intervals are (previous_boundary, boundary].
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

    # Vocabulary size calibration based on published research
    # Nation (2006): ~20,000 word families for native speakers
    # Goulden, Nation & Read (1990): ~20,000 for university graduates
    # Milton (2009): Chinese EFL learners max ~9,000-10,000
    # CET-6 threshold: ~6,000-7,000; TEM-8 threshold: ~10,000-13,000
    calibration_native_max: int = 20_000
    calibration_ceiling: int = 22_000
    calibration_k: float = 0.0000691  # tanh saturation rate
    #   k=0.0000691: calibrated = max_v * tanh(k * raw), then piecewise
    #   raw 21303 (all-correct) → cal 18049   (母语者水平)
    #   raw 10000 → cal 10335   (专业/母语级)
    #   raw 8000  → cal  7877   (六级+ / 考研)
    #   raw 6000  → cal  5181   (四级)
    #   raw 4000  → cal  4076   (高中/四级过渡)
    #   raw 2000  → cal  2747   (初中/高中过渡)

    abbreviation_max_len: int = 5
    min_word_len: int = 2
    fallback_rank_step: int = 100

    # Beta prior parameters for Bayesian smoothing of bucket known-rates
    # Each bucket's prior is interpolated between beta_prior_max_knowledge
    # (high-frequency / low-rank) and beta_prior_min_knowledge
    # (low-frequency / high-rank) using bucket index normalised to [0, 1].
    # bucket_alpha = prior_alpha_base + interpolated_knowledge_counts
    # bucket_beta  = prior_beta_base  + (1 - interpolated_knowledge_counts)
    beta_prior_alpha_base: float = 0.5
    beta_prior_beta_base: float = 0.5
    beta_prior_max_knowledge: float = 4.0
    beta_prior_min_knowledge: float = 0.5

    # --- Weighted logistic fitting (方案 A) ---
    # When enabled, each (word, known) sample is weighted by frequency rank.
    # High-rank (rare) words get lower weight, so the model prioritises
    # getting common words right.  This produces a more conservative estimate.
    enable_weighted_fitting: bool = True

    # Weight formula: w = 1 / (1 + log2(max(rank, 10) / 10))
    # Effective only when enable_weighted_fitting is True.

    # --- Piecewise linear calibration (方案 B) ---
    # Applied AFTER the tanh stage. Compresses the mid-range (3000-8000)
    # by 55% to keep ~四级 students around 4000-5000, then expands
    # the upper range by 28% so all-correct reaches ~18000.
    # Segments:
    #   [0, 3000]    → slope 1.00  (identity)
    #   [3000, 8000]  → slope 0.45  (mid-range compression)
    #   [8000, 22000] → slope 1.28  (high-end boost)
    enable_piecewise_calibration: bool = True

    # Each entry is (upper_boundary, slope_for_segment).
    # The first segment always starts at 0 with the first entry's slope.
    piecewise_knots: tuple[tuple[int, float], ...] = (
        (3_000, 1.00),
        (8_000, 0.45),
        (22_000, 1.28),
    )

    # Two-stage adaptive testing parameters
    phase1_question_count: int = 30
    enable_phase2: bool = False
    stage2_boundary_low: float = 0.20
    stage2_boundary_high: float = 0.80
    stage2_extra_per_bucket: int = 8


DEFAULT_CONFIG = EstimatorConfig()


def bucket_label(boundary: int) -> str:
    """Return a compact display label for a rank bucket boundary."""

    if boundary % 1000 == 0:
        return f"{boundary // 1000}k"
    return str(boundary)


def bucket_beta_prior(bucket_index: int, n_buckets: int, config: EstimatorConfig = DEFAULT_CONFIG) -> tuple[float, float]:
    """Return (alpha, beta) Beta prior pseudocounts for a frequency bucket.

    Low-index buckets (high frequency → low rank) get a knowledge-leaning prior.
    High-index buckets (low frequency → high rank) get an ignorance-leaning prior.
    Interpolation is linear by bucket index.
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
