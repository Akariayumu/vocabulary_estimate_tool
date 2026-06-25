"""词汇的综合难度评分。

结合两个维度：
- 教育阶段优先级（课程顺序）
- Wordfreq rank（语料频率）

分数范围：[0, 1]，越高表示越难。
"""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

from .vocab_bank import VocabBank

# ── 归一化辅助函数 ───────────────────────────────────────────────────────────

_VOCAB_SIZE = 30_000


def _norm_stage(priority: int) -> float:
    """将阶段优先级 [1..11] 做 min-max 归一化到 [0, 1]。"""
    return (priority - 1) / 10.0


def _norm_rank(rank: int) -> float:
    """将 wordfreq rank [1..30000] 做 log 归一化到 [0, 1]。

    选择 log 尺度是因为 wordfreq rank 遵循 Zipf 分布。
    线性归一化会把前 1000 个 rank 压缩到约 0.03。
    """
    return math.log(rank + 1) / math.log(_VOCAB_SIZE + 1)


# ── 缺失 rank：每个阶段的 rank 中位数 ─────────────────────────────────────────
# 根据实际 stage_vocab × wordfreq 数据计算。

_STAGE_MEDIAN_RANK: Dict[str, int] = {
    "primary_3": 1482,
    "primary_4": 1821,
    "primary_5": 1019,
    "primary_6": 1294,
    "junior_7": 922,
    "junior_8": 1660,
    "junior_9": 2714,
    "senior": 3608,
    "cet4": 4299,
    "cet6": 6775,
    "ielts": 7218,
}

# ── 缺失阶段：根据 rank 估计 priority ────────────────────────────────────────
# 对经验 first_stage priority 与 rank 做平滑分段插值。

_RANK_TO_PRIORITY_KNOTS: List[Tuple[float, float]] = [
    (0.0, 4.5),       # 超高频 → 约 junior_7
    (500.0, 7.0),     # 很常见 → 约 junior_9
    (1000.0, 7.5),    # 常见
    (2000.0, 8.0),    # → senior 阶段
    (3000.0, 8.5),
    (5000.0, 9.0),    # → cet4 阶段
    (8000.0, 9.5),
    (10000.0, 10.0),  # → cet6 阶段
    (18000.0, 10.5),
    (30000.0, 11.0),  # → ielts 阶段
]


def _estimate_priority_from_rank(rank: int) -> float:
    """在 rank→priority knots 之间做线性插值。"""
    if rank <= _RANK_TO_PRIORITY_KNOTS[0][0]:
        return _RANK_TO_PRIORITY_KNOTS[0][1]
    if rank >= _RANK_TO_PRIORITY_KNOTS[-1][0]:
        return _RANK_TO_PRIORITY_KNOTS[-1][1]
    for i in range(len(_RANK_TO_PRIORITY_KNOTS) - 1):
        r1, p1 = _RANK_TO_PRIORITY_KNOTS[i]
        r2, p2 = _RANK_TO_PRIORITY_KNOTS[i + 1]
        if r1 <= rank <= r2:
            t = (rank - r1) / (r2 - r1) if r2 != r1 else 0.0
            return p1 + t * (p2 - p1)
    return _RANK_TO_PRIORITY_KNOTS[-1][1]


# ── Stage key → priority 查表 ────────────────────────────────────────────────

_STAGE_PRIORITY = {
    "primary_3": 1,
    "primary_4": 2,
    "primary_5": 3,
    "primary_6": 4,
    "junior_7": 5,
    "junior_8": 6,
    "junior_9": 7,
    "senior": 8,
    "cet4": 9,
    "cet6": 10,
    "ielts": 11,
}


def compute_difficulty_scores(
    stage_vocab_path: str,
    bank: VocabBank,
    alpha: float = 0.60,
    beta: float = 0.40,
    *,
    estimate_missing_stage: bool = False,
) -> Dict[str, float]:
    """为 stage_vocab.json 中所有词计算综合难度分数。

    Args:
        stage_vocab_path: ``stage_vocab.json`` 的路径。
        bank: ``VocabBank`` 实例（提供 ``get_rank``）。
        alpha: 阶段优先级项的权重（默认 0.60）。
        beta: wordfreq-rank 项的权重（默认 0.40）。
        estimate_missing_stage: 为 True 时，也会给没有阶段条目的 bank-only 词评分，
            其阶段由 rank 估计。

    Returns:
        ``{word: difficulty_score}``，其中 ``difficulty_score ∈ [0, 1]``。
    """
    with open(stage_vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    stages: Dict = data["stages"]
    word_to_stage: Dict = data["word_to_stage"]

    # ── 步骤 1：计算每个阶段的 rank 中位数，用于填补缺失数据 ─────────────
    # （使用实时中位数而非硬编码常量，以适应数据变化。）
    stage_median_rank: Dict[str, int] = {}
    for sk, sv in stages.items():
        ranks = []
        for w in sv["words"]:
            r = bank.get_rank(w)
            if r is not None:
                ranks.append(r)
        if ranks:
            ranks.sort()
            stage_median_rank[sk] = ranks[len(ranks) // 2]
        else:
            stage_median_rank[sk] = _STAGE_MEDIAN_RANK.get(sk, 15000)

    # ── 步骤 2：计算分数 ───────────────────────────────────────────────
    scores: Dict[str, float] = {}

    for word, info in word_to_stage.items():
        first_stage = info["first_stage"]
        priority = _STAGE_PRIORITY.get(first_stage)
        if priority is None:
            # fallback：格式良好的数据中不应发生
            priority = _STAGE_PRIORITY.get("ielts", 11)

        rank = bank.get_rank(word)
        if rank is None:
            rank = stage_median_rank.get(first_stage, 15000)

        ns = _norm_stage(priority)
        nr = _norm_rank(rank)
        scores[word] = alpha * ns + beta * nr

    # ── 步骤 3（可选）：为 bank-only 词评分 ─────────────────────────────
    if estimate_missing_stage:
        bank_words_total = 0
        for item in bank.items:
            w = item.word
            if w not in scores and w not in word_to_stage:
                rank = item.rank
                est_pri = _estimate_priority_from_rank(rank)
                ns = _norm_stage(est_pri)
                nr = _norm_rank(rank)
                scores[w] = alpha * ns + beta * nr
                bank_words_total += 1

    return scores


def update_stage_vocab_with_difficulty(
    stage_vocab_path: str,
    bank: VocabBank,
    alpha: float = 0.60,
    beta: float = 0.40,
    *,
    output_path: Optional[str] = None,
) -> None:
    """为 stage_vocab.json 中每个词条添加 ``difficulty`` 字段。

    原地修改文件；若提供 ``output_path``，则写入该路径。
    """
    with open(stage_vocab_path, encoding="utf-8") as f:
        data = json.load(f)

    scores = compute_difficulty_scores(
        stage_vocab_path, bank, alpha=alpha, beta=beta
    )

    for word, info in data["word_to_stage"].items():
        info["difficulty"] = round(scores.get(word, 0.0), 4)

    dest = output_path or stage_vocab_path
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
