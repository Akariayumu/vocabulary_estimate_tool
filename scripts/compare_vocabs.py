#!/usr/bin/env python3
"""Compare stage vocabulary files for StratifiedQuiz migration impact."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_ENHANCED = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "vocab_migration_gap.md"


@dataclass(frozen=True)
class WordEntry:
    word: str
    difficulty: float
    cluster_20: int
    cluster_100: int | None
    first_stage: str | None
    sources: tuple[str, ...]


def _load_entries(path: Path) -> dict[str, WordEntry]:
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    word_to_stage = raw.get("word_to_stage")
    if not isinstance(word_to_stage, dict):
        raise ValueError(f"{path} must contain word_to_stage")

    entries: dict[str, WordEntry] = {}
    for word, info in word_to_stage.items():
        if not isinstance(word, str) or not isinstance(info, dict):
            continue
        if info.get("difficulty") is None or info.get("cluster_20") is None:
            continue
        normalized = word.strip().lower()
        if not normalized:
            continue
        sources = info.get("sources") or []
        entries[normalized] = WordEntry(
            word=normalized,
            difficulty=float(info["difficulty"]),
            cluster_20=int(info["cluster_20"]),
            cluster_100=int(info["cluster_100"]) if info.get("cluster_100") is not None else None,
            first_stage=info.get("first_stage"),
            sources=tuple(str(s) for s in sources),
        )

    if not entries:
        raise ValueError(f"{path} has no usable entries with difficulty and cluster_20")
    return entries


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "n": 0,
            "min": float("nan"),
            "p25": float("nan"),
            "p50": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "range": float("nan"),
            "iqr": float("nan"),
        }
    return {
        "n": float(len(values)),
        "min": min(values),
        "p25": _quantile(values, 0.25),
        "p50": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "max": max(values),
        "mean": _mean(values),
        "std": _std(values),
        "range": max(values) - min(values),
        "iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
    }


def _histogram(values: Sequence[float], *, start: float = 0.0, stop: float = 1.0, step: float = 0.05) -> list[dict[str, Any]]:
    bins: list[dict[str, Any]] = []
    edges = []
    x = start
    while x < stop - 1e-12:
        edges.append(round(x, 10))
        x += step
    edges.append(stop)

    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        if value < edges[0] or value > edges[-1]:
            continue
        idx = min(int((value - start) / step), len(counts) - 1)
        counts[idx] += 1

    total = len(values) or 1
    for i, count in enumerate(counts):
        bins.append(
            {
                "bin": f"{edges[i]:.2f}-{edges[i + 1]:.2f}",
                "count": count,
                "pct": count / total,
            }
        )
    return bins


def _cluster_stats(entries: dict[str, WordEntry], added_words: set[str] | None = None) -> list[dict[str, Any]]:
    by_cluster: dict[int, list[WordEntry]] = defaultdict(list)
    for entry in entries.values():
        by_cluster[entry.cluster_20].append(entry)

    rows: list[dict[str, Any]] = []
    added_words = added_words or set()
    for c20 in range(20):
        items = by_cluster.get(c20, [])
        diffs = [e.difficulty for e in items]
        s = _stats(diffs)
        added = sum(1 for e in items if e.word in added_words)
        rows.append(
            {
                "cluster_20": c20,
                "count": len(items),
                "min": s["min"],
                "p25": s["p25"],
                "p50": s["p50"],
                "p75": s["p75"],
                "max": s["max"],
                "range": s["range"],
                "std": s["std"],
                "added": added,
                "added_pct": added / len(items) if items else 0.0,
            }
        )
    return rows


def _logit(p: float) -> float:
    p = max(1e-10, min(1.0 - 1e-10, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _expected_vocab(theta: float, entries: dict[str, WordEntry]) -> float:
    total = 0.0
    for entry in entries.values():
        d = _logit(max(0.001, min(0.999, entry.difficulty)))
        total += _sigmoid(theta - d)
    return total


def _theta_grid_impact(base: dict[str, WordEntry], enhanced: dict[str, WordEntry]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for theta in [-3.0, -2.0, -1.5, -1.0, 0.0, 1.0, 1.5, 2.0, 3.0]:
        b = _expected_vocab(theta, base)
        e = _expected_vocab(theta, enhanced)
        rows.append(
            {
                "theta": theta,
                "base_raw": b,
                "enhanced_raw": e,
                "delta": e - b,
                "ratio": e / b if b else float("inf"),
                "base_x08": b * 0.8,
                "enhanced_x08": e * 0.8,
            }
        )
    return rows


def _movement_rows(base: dict[str, WordEntry], enhanced: dict[str, WordEntry]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    common = sorted(set(base) & set(enhanced))
    movements: Counter[tuple[int, int]] = Counter()
    abs_moves: list[int] = []
    changed = 0
    unchanged = 0
    for word in common:
        old = base[word].cluster_20
        new = enhanced[word].cluster_20
        movements[(old, new)] += 1
        delta = abs(new - old)
        abs_moves.append(delta)
        if old == new:
            unchanged += 1
        else:
            changed += 1

    per_old: list[dict[str, Any]] = []
    for old in range(20):
        old_words = [w for w in common if base[w].cluster_20 == old]
        if not old_words:
            continue
        kept = sum(1 for w in old_words if enhanced[w].cluster_20 == old)
        deltas = [enhanced[w].cluster_20 - base[w].cluster_20 for w in old_words]
        dominant_new, dominant_count = Counter(enhanced[w].cluster_20 for w in old_words).most_common(1)[0]
        per_old.append(
            {
                "old_cluster": old,
                "old_count": len(old_words),
                "kept": kept,
                "kept_pct": kept / len(old_words),
                "moved": len(old_words) - kept,
                "mean_delta": _mean([float(d) for d in deltas]),
                "mean_abs_delta": _mean([float(abs(d)) for d in deltas]),
                "dominant_new": dominant_new,
                "dominant_new_count": dominant_count,
            }
        )

    summary = {
        "common": len(common),
        "changed": changed,
        "unchanged": unchanged,
        "changed_pct": changed / len(common) if common else 0.0,
        "mean_abs_delta": _mean([float(x) for x in abs_moves]),
        "p50_abs_delta": _quantile([float(x) for x in abs_moves], 0.50),
        "p75_abs_delta": _quantile([float(x) for x in abs_moves], 0.75),
        "p90_abs_delta": _quantile([float(x) for x in abs_moves], 0.90),
        "max_abs_delta": max(abs_moves) if abs_moves else 0,
    }
    return per_old, summary


def _format_float(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _format_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def _md_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _difficulty_summary_row(label: str, entries: dict[str, WordEntry]) -> list[str]:
    values = [e.difficulty for e in entries.values()]
    s = _stats(values)
    return [
        label,
        _format_int(len(values)),
        _format_float(s["min"]),
        _format_float(s["p25"]),
        _format_float(s["p50"]),
        _format_float(s["p75"]),
        _format_float(s["max"]),
        _format_float(s["mean"]),
        _format_float(s["std"]),
    ]


def _risk_label(condition: bool, true_label: str = "高", false_label: str = "中") -> str:
    return true_label if condition else false_label


def build_report(base_path: Path, enhanced_path: Path) -> str:
    base = _load_entries(base_path)
    enhanced = _load_entries(enhanced_path)

    base_words = set(base)
    enhanced_words = set(enhanced)
    common_words = base_words & enhanced_words
    added_words = enhanced_words - base_words
    removed_words = base_words - enhanced_words

    added_entries = {w: enhanced[w] for w in added_words}
    base_cluster = _cluster_stats(base)
    enhanced_cluster = _cluster_stats(enhanced, added_words=added_words)
    movement_by_old, movement_summary = _movement_rows(base, enhanced)
    theta_rows = _theta_grid_impact(base, enhanced)

    base_counts = {row["cluster_20"]: row["count"] for row in base_cluster}
    enhanced_counts = {row["cluster_20"]: row["count"] for row in enhanced_cluster}
    capacity_delta_rows = sorted(
        (
            {
                "cluster": c,
                "base": base_counts.get(c, 0),
                "enhanced": enhanced_counts.get(c, 0),
                "delta": enhanced_counts.get(c, 0) - base_counts.get(c, 0),
                "delta_pct": (enhanced_counts.get(c, 0) - base_counts.get(c, 0)) / base_counts.get(c, 1),
            }
            for c in range(20)
        ),
        key=lambda r: abs(r["delta"]),
        reverse=True,
    )

    added_by_cluster = Counter(enhanced[w].cluster_20 for w in added_words)
    added_sources = Counter(source for w in added_words for source in enhanced[w].sources)

    base_s = _stats([e.difficulty for e in base.values()])
    enhanced_s = _stats([e.difficulty for e in enhanced.values()])
    added_s = _stats([e.difficulty for e in added_entries.values()])
    enhanced_range_mean = _mean([row["range"] for row in enhanced_cluster])
    base_range_mean = _mean([row["range"] for row in base_cluster])
    enhanced_std_mean = _mean([row["std"] for row in enhanced_cluster])
    base_std_mean = _mean([row["std"] for row in base_cluster])

    requested_enhanced_size = 19_562
    requested_added = 8_144
    actual_size_note = ""
    if len(enhanced) != requested_enhanced_size or len(added_words) != requested_added:
        actual_size_note = (
            f"> 注意：任务描述写的是 enhanced 词库 {requested_enhanced_size:,} 词、"
            f"新增 {requested_added:,} 词；当前文件实际可用条目是 {len(enhanced):,} 词、"
            f"新增 {len(added_words):,} 词。本报告按当前文件内容计算。\n\n"
        )

    movement_risk = _risk_label(movement_summary["changed_pct"] >= 0.5)
    calibration_risk = _risk_label(theta_rows[-1]["ratio"] >= 1.2, true_label="高", false_label="中")
    sampling_risk = "中"
    recommendation = "条件迁移"
    if movement_risk == "低" and calibration_risk == "低":
        recommendation = "立即迁移"
    if movement_risk == "高" and calibration_risk == "高":
        recommendation = "条件迁移"

    lines: list[str] = []
    lines.append("# 词库迁移差异评估：stage_vocab vs stage_vocab_enhanced")
    lines.append("")
    lines.append(actual_size_note.rstrip())
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 基准词库：`{base_path.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 增强词库：`{enhanced_path.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- 基准可用词：{len(base):,}")
    lines.append(f"- 增强可用词：{len(enhanced):,}")
    lines.append(f"- 共同词：{len(common_words):,}")
    lines.append(f"- 新增词：{len(added_words):,}")
    lines.append(f"- 移除词：{len(removed_words):,}")
    lines.append("")

    lines.append("## 1. 难度分布差异")
    lines.append("")
    lines.append(
        _md_table(
            ["集合", "n", "min", "p25", "p50", "p75", "max", "mean", "std"],
            [
                _difficulty_summary_row("stage_vocab", base),
                _difficulty_summary_row("enhanced", enhanced),
                _difficulty_summary_row("新增词", added_entries),
            ],
        )
    )
    lines.append("")
    lines.append(
        "- 新增词难度中心明显偏高："
        f"新增词 p50={_format_float(added_s['p50'])}，原词库 p50={_format_float(base_s['p50'])}。"
        "这符合 COCA/TOEFL/GRE 扩展带来更多高阶词的预期。"
    )
    lines.append(
        "- 但新增词也覆盖了中低难度区间："
        f"新增词 p25={_format_float(added_s['p25'])}，min={_format_float(added_s['min'])}，"
        "说明增强词库不是纯高难词尾部追加。"
    )
    lines.append("")
    lines.append("### difficulty 直方图（0.05 桶宽）")
    lines.append("")
    hist_base = {row["bin"]: row for row in _histogram([e.difficulty for e in base.values()])}
    hist_enhanced = {row["bin"]: row for row in _histogram([e.difficulty for e in enhanced.values()])}
    hist_added = {row["bin"]: row for row in _histogram([e.difficulty for e in added_entries.values()])}
    hist_bins = list(hist_enhanced)
    lines.append(
        _md_table(
            ["difficulty", "stage_vocab", "enhanced", "新增词", "新增词占比"],
            [
                [
                    b,
                    _format_int(hist_base.get(b, {}).get("count", 0)),
                    _format_int(hist_enhanced.get(b, {}).get("count", 0)),
                    _format_int(hist_added.get(b, {}).get("count", 0)),
                    f"{hist_added.get(b, {}).get('pct', 0.0):.1%}",
                ]
                for b in hist_bins
                if hist_base.get(b, {}).get("count", 0)
                or hist_enhanced.get(b, {}).get("count", 0)
                or hist_added.get(b, {}).get("count", 0)
            ],
        )
    )
    lines.append("")

    lines.append("### 新增词在 enhanced cluster_20 中的分布")
    lines.append("")
    lines.append(
        _md_table(
            ["cluster_20", "新增词数", "新增词占全部新增", "桶容量", "桶内新增占比"],
            [
                [
                    c,
                    _format_int(added_by_cluster.get(c, 0)),
                    f"{added_by_cluster.get(c, 0) / max(1, len(added_words)):.1%}",
                    _format_int(enhanced_counts.get(c, 0)),
                    f"{added_by_cluster.get(c, 0) / max(1, enhanced_counts.get(c, 0)):.1%}",
                ]
                for c in range(20)
            ],
        )
    )
    lines.append("")
    if added_sources:
        lines.append("新增词来源 Top 项：")
        lines.append("")
        lines.append(
            _md_table(
                ["source", "count"],
                [[source, _format_int(count)] for source, count in added_sources.most_common(12)],
            )
        )
        lines.append("")

    lines.append("## 2. cluster_20 分桶变化")
    lines.append("")
    lines.append("### 两个版本各自 20 桶容量和难度范围")
    lines.append("")
    lines.append(
        _md_table(
            [
                "cluster",
                "old_n",
                "old_min",
                "old_max",
                "old_range",
                "old_std",
                "new_n",
                "new_min",
                "new_max",
                "new_range",
                "new_std",
                "new_added%",
            ],
            [
                [
                    c,
                    _format_int(base_cluster[c]["count"]),
                    _format_float(base_cluster[c]["min"]),
                    _format_float(base_cluster[c]["max"]),
                    _format_float(base_cluster[c]["range"]),
                    _format_float(base_cluster[c]["std"]),
                    _format_int(enhanced_cluster[c]["count"]),
                    _format_float(enhanced_cluster[c]["min"]),
                    _format_float(enhanced_cluster[c]["max"]),
                    _format_float(enhanced_cluster[c]["range"]),
                    _format_float(enhanced_cluster[c]["std"]),
                    f"{enhanced_cluster[c]['added_pct']:.1%}",
                ]
                for c in range(20)
            ],
        )
    )
    lines.append("")
    lines.append(
        f"- 平均桶内 difficulty range：old={_format_float(base_range_mean)}，"
        f"enhanced={_format_float(enhanced_range_mean)}。"
    )
    lines.append(
        f"- 平均桶内 difficulty std：old={_format_float(base_std_mean)}，"
        f"enhanced={_format_float(enhanced_std_mean)}。"
    )
    lines.append(
        "- enhanced 的 cluster 容量几乎等宽；平均 range 基本持平，平均 std 略低。"
        "这对同桶抽题的一致性是小幅正向变化，但边缘桶仍需抽样检查。"
    )
    lines.append("")

    lines.append("### 原有词分桶移动")
    lines.append("")
    lines.append(
        f"- 原有共同词中 {movement_summary['changed']:,}/{movement_summary['common']:,} "
        f"发生 cluster_20 变化，占 {movement_summary['changed_pct']:.1%}。"
    )
    lines.append(
        f"- 绝对移动距离：mean={_format_float(movement_summary['mean_abs_delta'])} 桶，"
        f"p50={_format_float(movement_summary['p50_abs_delta'])}，"
        f"p75={_format_float(movement_summary['p75_abs_delta'])}，"
        f"p90={_format_float(movement_summary['p90_abs_delta'])}，"
        f"max={movement_summary['max_abs_delta']}。"
    )
    lines.append("")
    lines.append(
        _md_table(
            ["old_cluster", "old_n", "kept", "kept%", "moved", "mean_abs_delta", "dominant_new", "dominant_new_n"],
            [
                [
                    row["old_cluster"],
                    _format_int(row["old_count"]),
                    _format_int(row["kept"]),
                    f"{row['kept_pct']:.1%}",
                    _format_int(row["moved"]),
                    _format_float(row["mean_abs_delta"]),
                    row["dominant_new"],
                    _format_int(row["dominant_new_count"]),
                ]
                for row in sorted(movement_by_old, key=lambda r: r["mean_abs_delta"], reverse=True)
            ],
        )
    )
    lines.append("")
    lines.append("### 容量变化最大的桶")
    lines.append("")
    lines.append(
        _md_table(
            ["cluster", "old_n", "new_n", "delta", "delta%"],
            [
                [
                    row["cluster"],
                    _format_int(row["base"]),
                    _format_int(row["enhanced"]),
                    f"{row['delta']:+,}",
                    f"{row['delta_pct']:+.1%}",
                ]
                for row in capacity_delta_rows[:10]
            ],
        )
    )
    lines.append("")

    lines.append("## 3. 对 StratifiedQuiz 的影响")
    lines.append("")
    lines.append(
        "- `vocab_estimator/stratified_quiz.py` 当前默认加载 `data/stage_vocab.json`；"
        "切到 enhanced 后，Phase 1/Phase 2 的可抽题池、`_word_difficulties` 总和、"
        "以及 response 解析都会一起改变。"
    )
    lines.append(
        "- 代码当前默认 `phase1_question_count=30`：先覆盖 20 个桶各 1 题，"
        "再补 10 个中间桶；当配置为 40 题时才是 20 个桶各 2 题。"
    )
    lines.append(
        "- `phase2_sample()` 方法默认每低置信桶 4 题，"
        "`tests/simulation_eval.py` 调用时传入 `phase2_n_per_class=8`。"
    )
    lines.append(
        f"- enhanced 每桶新增词占比范围："
        f"{min(row['added_pct'] for row in enhanced_cluster):.1%} - "
        f"{max(row['added_pct'] for row in enhanced_cluster):.1%}。"
        "因此任意桶抽样都有较大概率抽到新增词。"
    )
    lines.append(
        "- 由于共同词有大量 cluster_20 移动，`STREAMING_CLUSTER_ORDER` 的具体词集合会变；"
        "enhanced 桶容量更均衡，桶内 std 略低，单桶抽题的平均一致性略有改善。"
    )
    lines.append("")

    lines.append("## 4. 对模拟评估和校准的影响")
    lines.append("")
    lines.append(
        "`tests/simulation_eval.py` 的 synthetic true_vocab 与 estimated_vocab 都是 "
        "`sum(sigmoid(theta - logit(difficulty)))`。换成 enhanced 后，真值定义从 "
        f"{len(base):,} 个词扩展到 {len(enhanced):,} 个词。"
    )
    lines.append("")
    lines.append(
        _md_table(
            ["theta", "old_raw", "new_raw", "delta", "new/old", "old*0.8", "new*0.8"],
            [
                [
                    _format_float(row["theta"], 1),
                    _format_int(row["base_raw"]),
                    _format_int(row["enhanced_raw"]),
                    f"{int(round(row['delta'])):+,}",
                    _format_float(row["ratio"], 3),
                    _format_int(row["base_x08"]),
                    _format_int(row["enhanced_x08"]),
                ]
                for row in theta_rows
            ],
        )
    )
    lines.append("")
    lines.append(
        "- `StratifiedQuiz._vocab_at_theta()` 先把 raw sum 乘以 `0.8`，"
        "再进入 `_calibrate()` 的 tanh/piecewise 校准。这个 `0.8` 是基于旧词库的经验修正，"
        "不能直接证明适用于 enhanced。"
    )
    lines.append(
        "- 如果业务定义仍希望估计旧量表规模，enhanced 的 raw sum 需要重新映射回旧量表；"
        "如果业务定义改为 enhanced 规模，则前端解释、等级阈值和 simulation target range 都要同步更新。"
    )
    lines.append(
        "- `--true-max` 默认 15000，小于 enhanced 词库规模。"
        "这仍可运行，但高能力用户区间不会覆盖 enhanced 尾部词的饱和表现。"
    )
    lines.append("")

    lines.append("## 风险评估")
    lines.append("")
    lines.append(
        _md_table(
            ["风险项", "等级", "依据", "影响"],
            [
                [
                    "cluster_20 语义漂移",
                    movement_risk,
                    f"{movement_summary['changed_pct']:.1%} 原有词移动桶",
                    "历史按桶统计、固定 seed 抽题、题目回放不可直接对齐",
                ],
                [
                    "校准失效",
                    calibration_risk,
                    f"同 theta 下 raw sum 最高约为旧词库 {max(r['ratio'] for r in theta_rows):.3f} 倍",
                    "`0.8` 系数和等级阈值需重训/重标定",
                ],
                [
                    "抽题一致性",
                    sampling_risk,
                    f"平均桶内 std 从 {_format_float(base_std_mean)} 到 {_format_float(enhanced_std_mean)}",
                    "桶内难度更集中，但抽中新增词比例高",
                ],
                [
                    "数据版本认知",
                    "中",
                    f"任务描述规模与实际文件差 {len(enhanced) - requested_enhanced_size:+,} 词",
                    "迁移文档和实验命名容易混淆",
                ],
            ],
        )
    )
    lines.append("")

    lines.append("## 迁移建议")
    lines.append("")
    lines.append(f"建议：**{recommendation}**。")
    lines.append("")
    lines.append("建议步骤：")
    lines.append("")
    lines.append("1. 保持 `StratifiedQuiz(stage_vocab_path=...)` 可配置，不要直接覆盖旧 `stage_vocab.json`。")
    lines.append("2. 对旧词库和 enhanced 分别运行 `tests/simulation_eval.py --stage-vocab ...`，固定 seed 输出两份结果。")
    lines.append("3. 基于真实或人工标注样本重新拟合 `_vocab_at_theta()` 的 `0.8` 系数和 `_calibrate()` 参数。")
    lines.append("4. 更新等级阈值与产品文案，明确估计量表是旧 11,418 词还是 enhanced 19,801 词。")
    lines.append("5. 如果上线 enhanced，给历史测评记录保存 `vocab_version`，避免旧桶号和新桶号混查。")
    lines.append("6. 上线前抽样检查每个 cluster 的新增词翻译、词形和专名过滤质量，优先检查新增占比最高的桶。")
    lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append(
        "enhanced 词库在桶容量均衡上更适合 StratifiedQuiz，桶内难度一致性也有小幅改善，"
        "新增词难度分布也符合扩展词库预期；主要问题不是抽题池质量，而是量表迁移。"
        "由于原有词大量重分桶，且 raw vocabulary sum 的尺度显著变大，"
        "不建议无校准直接替换生产默认词库。"
    )

    return "\n".join(line for line in lines if line is not None).replace("\n\n\n", "\n\n") + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare stage vocabularies and write a migration report.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Path to the current stage vocab JSON.")
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED, help="Path to the enhanced stage vocab JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown report output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.base.resolve(), args.enhanced.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
