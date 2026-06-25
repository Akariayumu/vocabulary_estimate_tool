#!/usr/bin/env python3
"""使用统一校准重新计算增强词库 difficulty。

此脚本有意不改动 ``data/stage_vocab.json`` 和
``data/stage_vocab_enhanced.json``，而是写入新的
``data/stage_vocab_v2.json``。
"""

from __future__ import annotations

import argparse
import bisect
import copy
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vocab_estimator.difficulty import (  # noqa: E402
    _RANK_TO_PRIORITY_KNOTS,
    _STAGE_MEDIAN_RANK,
    _estimate_priority_from_rank,
)
from vocab_estimator.vocab_bank import VocabBank  # noqa: E402


DEFAULT_ORIGINAL = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_ENHANCED = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "stage_vocab_v2.json"
DEFAULT_EXAM_DIR = PROJECT_ROOT / "data" / "exam_vocab"
VOCAB_SIZE = 30_000
ALPHA = 0.60
BETA = 0.40
TZ_SHANGHAI = timezone(timedelta(hours=8))

STAGE_PRIORITY: dict[str, float] = {
    "primary_3": 1.0,
    "primary_4": 2.0,
    "primary_5": 3.0,
    "primary_6": 4.0,
    "junior_7": 5.0,
    "junior_8": 6.0,
    "junior_9": 7.0,
    "senior": 8.0,
    "cet4": 9.0,
    "cet6": 10.0,
    "ielts": 11.0,
}

PRIORITY_STAGE: list[tuple[float, str]] = [
    (1.49, "primary_3"),
    (2.49, "primary_4"),
    (3.49, "primary_5"),
    (4.49, "primary_6"),
    (5.49, "junior_7"),
    (6.49, "junior_8"),
    (7.49, "junior_9"),
    (8.49, "senior"),
    (9.49, "cet4"),
    (10.49, "cet6"),
    (11.00, "ielts"),
]

EXAM_FILES = {
    "coca20000": "coca20000.txt",
    "toefl": "toefl.txt",
    "gre": "gre.txt",
    "cet6": "cet6.txt",
    "gaokao": "gaokao.txt",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("word_to_stage"), dict):
        raise ValueError(f"{path} must contain a word_to_stage object")
    return data


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_wordlist_ranks(path: Path) -> dict[str, int]:
    ranks: dict[str, int] = {}
    if not path.exists():
        return ranks
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0].lower() if line.strip() else ""
        if token and token not in ranks:
            ranks[token] = len(ranks) + 1
    return ranks


def load_exam_vocab(exam_dir: Path) -> dict[str, dict[str, int]]:
    return {
        name: load_wordlist_ranks(exam_dir / filename)
        for name, filename in EXAM_FILES.items()
    }


def norm_stage(priority: float) -> float:
    return (priority - 1.0) / 10.0


def norm_rank(rank: int) -> float:
    bounded = max(1, min(VOCAB_SIZE, int(rank)))
    return math.log(bounded + 1) / math.log(VOCAB_SIZE + 1)


def compute_difficulty(priority: float, rank: int) -> float:
    return ALPHA * norm_stage(priority) + BETA * norm_rank(rank)


def priority_to_stage(priority: float) -> str:
    bounded = max(1.0, min(11.0, priority))
    for upper, stage in PRIORITY_STAGE:
        if bounded <= upper:
            return stage
    return "ielts"


def rank_for_priority(priority: float) -> int:
    """选择插值 priority 接近 ``priority`` 的 rank。"""
    bounded = max(1.0, min(11.0, priority))
    knots = _RANK_TO_PRIORITY_KNOTS
    if bounded <= knots[0][1]:
        return max(1, int(knots[0][0] or 1))
    if bounded >= knots[-1][1]:
        return int(knots[-1][0])
    for i in range(len(knots) - 1):
        r1, p1 = knots[i]
        r2, p2 = knots[i + 1]
        if p1 <= bounded <= p2:
            t = (bounded - p1) / (p2 - p1) if p2 != p1 else 0.0
            return max(1, min(VOCAB_SIZE, int(round(r1 + t * (r2 - r1)))))
    return int(knots[-1][0])


def source_blob(info: dict[str, Any], exam_vocab: dict[str, dict[str, int]], word: str) -> str:
    parts = [str(s).lower() for s in info.get("sources") or []]
    category = (info.get("expansion") or {}).get("category")
    if category:
        parts.append(str(category).lower())
    for name, ranks in exam_vocab.items():
        if word in ranks:
            parts.append(name)
    return " ".join(parts)


def priority_for_new_word(
    word: str,
    info: dict[str, Any],
    rank: int,
    exam_vocab: dict[str, dict[str, int]],
) -> tuple[float, str]:
    sources = source_blob(info, exam_vocab, word)
    source_priorities: list[tuple[float, str]] = []

    if "toefl" in sources:
        source_priorities.append((9.5, "source:toefl"))
    if "gre" in sources:
        source_priorities.append((10.0, "source:gre"))
    if "modern_domain" in sources or "manual_domain" in sources:
        source_priorities.append((9.0, "source:modern_domain"))

    if source_priorities:
        return max(source_priorities, key=lambda item: item[0])
    if "coca20000" in sources:
        return _estimate_priority_from_rank(rank), "rank:coca20000"
    return _estimate_priority_from_rank(rank), "rank:default"


def rank_for_word(
    word: str,
    info: dict[str, Any],
    *,
    bank: VocabBank,
    coca_ranks: dict[str, int],
    is_original_word: bool,
    priority: float | None,
) -> tuple[int, str]:
    bank_rank = bank.get_rank(word)
    if bank_rank is not None:
        return max(1, min(VOCAB_SIZE, int(bank_rank))), "wordfreq"

    coca_rank = coca_ranks.get(word.lower())
    if coca_rank is not None:
        return max(1, min(VOCAB_SIZE, int(coca_rank))), "coca20000"

    if is_original_word:
        first_stage = info.get("first_stage")
        if isinstance(first_stage, str) and first_stage in _STAGE_MEDIAN_RANK:
            return _STAGE_MEDIAN_RANK[first_stage], "stage_median"

    if priority is not None:
        return rank_for_priority(priority), "priority_inverse"

    first_stage = info.get("first_stage")
    if isinstance(first_stage, str) and first_stage in _STAGE_MEDIAN_RANK:
        return _STAGE_MEDIAN_RANK[first_stage], "stage_median"
    return 15_000, "global_default"


def rebuild_stages(vocab: dict[str, Any]) -> None:
    word_to_stage = vocab["word_to_stage"]
    stages = vocab.get("stages") or {}
    for stage_info in stages.values():
        stage_info["words"] = []

    for word, info in word_to_stage.items():
        first_stage = info.get("first_stage")
        all_stages = info.get("all_stages") or ([first_stage] if first_stage else [])
        clean_stages = [stage for stage in all_stages if stage in stages]
        if not clean_stages and first_stage in stages:
            clean_stages = [first_stage]
        info["all_stages"] = clean_stages
        if first_stage not in stages and clean_stages:
            info["first_stage"] = clean_stages[0]
        for stage in clean_stages:
            stages[stage]["words"].append(word)

    for stage_info in stages.values():
        words = sorted(set(stage_info["words"]))
        stage_info["words"] = words
        stage_info["count"] = len(words)


def rebuild_quantile_clusters(vocab: dict[str, Any]) -> None:
    word_to_stage = vocab["word_to_stage"]
    sorted_words = sorted(
        (float(info["difficulty"]), word)
        for word, info in word_to_stage.items()
        if info.get("difficulty") is not None
    )
    total = len(sorted_words)
    for index, (_, word) in enumerate(sorted_words):
        word_to_stage[word]["cluster_20"] = min(19, int(index * 20 / total))
        word_to_stage[word]["cluster_100"] = min(99, int(index * 100 / total))


def median(values: list[float]) -> float:
    values = sorted(values)
    if not values:
        raise ValueError("median() requires at least one value")
    return values[len(values) // 2]


def monotonic(values: list[float]) -> list[float]:
    """返回适合构造阈值的非递减版本。"""
    result: list[float] = []
    current = 0.0
    for value in values:
        current = max(current, value)
        result.append(current)
    return result


def assign_anchored_cluster(difficulty: float, centers: list[float]) -> int:
    boundaries = [(centers[i] + centers[i + 1]) / 2.0 for i in range(len(centers) - 1)]
    return min(len(centers) - 1, bisect.bisect_right(boundaries, difficulty))


def rebuild_anchored_clusters(vocab: dict[str, Any], original: dict[str, Any]) -> dict[str, Any]:
    """将原始词保留在 legacy buckets；新增词按 v2 分数放置。

    这样既保留现有 quiz/history 数据的迁移语义，又能为每个词和新词放置使用统一的 v2 difficulty。
    """
    word_to_stage = vocab["word_to_stage"]
    original_entries = original["word_to_stage"]
    original_words = set(original_entries)
    stats: dict[str, Any] = {"anchor_centers": {}}

    for cluster_key, n_classes in (("cluster_20", 20), ("cluster_100", 100)):
        centers: list[float] = []
        for cluster in range(n_classes):
            values = [
                float(word_to_stage[word]["difficulty"])
                for word, info in original_entries.items()
                if word in word_to_stage and info.get(cluster_key) == cluster
            ]
            if values:
                centers.append(median(values))
            elif centers:
                centers.append(centers[-1])
            else:
                centers.append(0.0)
        centers = monotonic(centers)
        stats["anchor_centers"][cluster_key] = [round(value, 4) for value in centers]

        for word, info in word_to_stage.items():
            if word in original_words and original_entries[word].get(cluster_key) is not None:
                info[cluster_key] = int(original_entries[word][cluster_key])
            else:
                info[cluster_key] = assign_anchored_cluster(float(info["difficulty"]), centers)

    return stats


def rebuild_overlap_and_sort(vocab: dict[str, Any]) -> None:
    word_to_stage = vocab["word_to_stage"]
    stages = vocab.get("stages") or {}
    stage_sets = {stage: set(info["words"]) for stage, info in stages.items()}
    vocab["overlap_matrix"] = {
        stage_a: {
            stage_b: len(words_a & words_b)
            for stage_b, words_b in stage_sets.items()
            if stage_a != stage_b
        }
        for stage_a, words_a in stage_sets.items()
    }
    vocab["word_to_stage"] = dict(sorted(word_to_stage.items()))


def recalibrate(
    original: dict[str, Any],
    enhanced: dict[str, Any],
    exam_vocab: dict[str, dict[str, int]],
    *,
    cluster_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_entries = original["word_to_stage"]
    enhanced_entries = enhanced["word_to_stage"]
    original_words = set(original_entries)
    coca_ranks = exam_vocab["coca20000"]
    bank = VocabBank()

    output = copy.deepcopy(enhanced)
    stats: dict[str, Any] = {
        "word_count": len(output["word_to_stage"]),
        "original_word_count": len(original_words & set(output["word_to_stage"])),
        "added_word_count": len(set(output["word_to_stage"]) - original_words),
        "rank_source_counts": Counter(),
        "priority_source_counts": Counter(),
        "difficulty_delta_original": {},
    }

    original_deltas: list[float] = []
    for word, info in output["word_to_stage"].items():
        is_original = word in original_words
        if is_original:
            original_info = original_entries[word]
            first_stage = original_info.get("first_stage")
            priority = STAGE_PRIORITY.get(first_stage, 11.0)
            priority_source = f"original_stage:{first_stage}"
        else:
            preliminary_rank, preliminary_rank_source = rank_for_word(
                word,
                info,
                bank=bank,
                coca_ranks=coca_ranks,
                is_original_word=False,
                priority=None,
            )
            priority, priority_source = priority_for_new_word(
                word, info, preliminary_rank, exam_vocab
            )
            # 对 exam-only 词使用已知 priority 重新运行 rank fallback。
            rank, rank_source = rank_for_word(
                word,
                info,
                bank=bank,
                coca_ranks=coca_ranks,
                is_original_word=False,
                priority=priority,
            )
            stats["rank_source_counts"][rank_source] += 1
            stats["priority_source_counts"][priority_source] += 1
            info["first_stage"] = priority_to_stage(priority)
            info["all_stages"] = [info["first_stage"]]
            info.setdefault("expansion", {})["v2_priority"] = round(priority, 3)
            info.setdefault("expansion", {})["v2_rank_for_score"] = rank
            info.setdefault("expansion", {})["v2_priority_source"] = priority_source
            info.setdefault("expansion", {})["v2_rank_source"] = rank_source
            info["difficulty"] = round(compute_difficulty(priority, rank), 4)
            continue

        rank, rank_source = rank_for_word(
            word,
            original_entries[word],
            bank=bank,
            coca_ranks=coca_ranks,
            is_original_word=True,
            priority=priority,
        )
        stats["rank_source_counts"][rank_source] += 1
        stats["priority_source_counts"][priority_source] += 1
        old_difficulty = original_entries[word].get("difficulty")
        new_difficulty = round(compute_difficulty(priority, rank), 4)
        info["difficulty"] = new_difficulty
        if isinstance(old_difficulty, (int, float)):
            original_deltas.append(abs(new_difficulty - float(old_difficulty)))

    if original_deltas:
        original_deltas.sort()
        stats["difficulty_delta_original"] = {
            "mean_abs": round(sum(original_deltas) / len(original_deltas), 6),
            "p50_abs": round(original_deltas[len(original_deltas) // 2], 6),
            "p90_abs": round(original_deltas[int(0.9 * (len(original_deltas) - 1))], 6),
            "max_abs": round(max(original_deltas), 6),
        }

    rebuild_stages(output)
    if cluster_mode == "anchored":
        cluster_stats = rebuild_anchored_clusters(output, original)
    elif cluster_mode == "quantile":
        rebuild_quantile_clusters(output)
        cluster_stats = {}
    else:
        raise ValueError(f"unsupported cluster mode: {cluster_mode}")
    rebuild_overlap_and_sort(output)
    stats["rank_source_counts"] = dict(stats["rank_source_counts"])
    stats["priority_source_counts"] = dict(stats["priority_source_counts"])
    stats["cluster_mode"] = cluster_mode
    stats["cluster_stats"] = cluster_stats
    return output, stats


def write_output(path: Path, data: dict[str, Any], *, original: Path, enhanced: Path, exam_dir: Path, stats: dict[str, Any]) -> None:
    meta = data.setdefault("meta", {})
    meta["generated_at"] = datetime.now(TZ_SHANGHAI).isoformat(timespec="seconds")
    meta["generator"] = "scripts/redesign_difficulty.py"
    meta["difficulty_redesign"] = {
        "version": "v2_unified_stage_rank_2026_06_25",
        "formula": "difficulty = 0.60 * ((priority - 1) / 10) + 0.40 * log(rank + 1) / log(30001)",
        "alpha": ALPHA,
        "beta": BETA,
        "vocab_size": VOCAB_SIZE,
        "original_priority": "word in original stage_vocab.json uses original first_stage -> priority",
        "new_priority": {
            "gre": 10.0,
            "toefl": 9.5,
            "modern_domain": 9.0,
            "coca20000_only": "estimate_priority_from_rank(rank)",
            "default": "estimate_priority_from_rank(rank)",
        },
        "rank": {
            "primary": "VocabBank wordfreq rank",
            "fallback_1": "COCA wordlist line number",
            "fallback_2_original": "_STAGE_MEDIAN_RANK[first_stage]",
            "fallback_2_added": "inverse _RANK_TO_PRIORITY_KNOTS from assigned priority",
        },
        "clusters": (
            "anchored: original words keep legacy clusters and added words are assigned from v2 difficulty "
            "against original-word anchor centers; quantile mode is available via --cluster-mode quantile"
        ),
        "input_hashes": {
            str(original.relative_to(PROJECT_ROOT)): file_sha256(original),
            str(enhanced.relative_to(PROJECT_ROOT)): file_sha256(enhanced),
            **{
                str((exam_dir / filename).relative_to(PROJECT_ROOT)): file_sha256(exam_dir / filename)
                for filename in EXAM_FILES.values()
                if (exam_dir / filename).exists()
            },
        },
        "stats": stats,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute stage_vocab_enhanced difficulty into stage_vocab_v2.")
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED)
    parser.add_argument("--exam-dir", type=Path, default=DEFAULT_EXAM_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cluster-mode",
        choices=("anchored", "quantile"),
        default="anchored",
        help="anchored preserves original-word cluster labels; quantile recomputes equal-frequency buckets over all words.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original = load_json(args.original)
    enhanced = load_json(args.enhanced)
    exam_vocab = load_exam_vocab(args.exam_dir)

    output, stats = recalibrate(original, enhanced, exam_vocab, cluster_mode=args.cluster_mode)
    write_output(
        args.output,
        output,
        original=args.original.resolve(),
        enhanced=args.enhanced.resolve(),
        exam_dir=args.exam_dir.resolve(),
        stats=stats,
    )

    print(f"wrote {args.output}")
    print(f"words: {stats['word_count']:,}")
    print(f"original words: {stats['original_word_count']:,}")
    print(f"added words: {stats['added_word_count']:,}")
    print(f"rank sources: {stats['rank_source_counts']}")
    print(f"priority sources: {stats['priority_source_counts']}")
    print(f"original difficulty delta: {stats['difficulty_delta_original']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
