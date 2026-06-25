#!/usr/bin/env python3
"""清洗 stage vocabulary 中的非英文词条并报告翻译缺失情况。"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "vocab_cleaning_report.md"
VOCAB_SPECS = (
    ("v1", PROJECT_ROOT / "data" / "stage_vocab.json", PROJECT_ROOT / "data" / "stage_vocab_clean_v1.json"),
    (
        "v2",
        PROJECT_ROOT / "data" / "stage_vocab_v2_clusterv1.json",
        PROJECT_ROOT / "data" / "stage_vocab_clean_v2.json",
    ),
)

VALID_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z'-]*$")


@dataclass(frozen=True)
class VocabSpec:
    name: str
    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class CleaningStats:
    name: str
    input_path: Path
    output_path: Path
    total_words: int
    dirty_words: list[str]
    missing_translation_words: list[str]
    missing_translation_field_words: list[str]
    empty_translation_words: list[str]
    cleaned_words: int
    invalid_character_counts: dict[str, int]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("word_to_stage"), dict):
        raise ValueError(f"{path} must contain a word_to_stage object")
    return data


def _is_valid_word(word: str) -> bool:
    return bool(VALID_WORD_RE.fullmatch(word))


def _invalid_character_counts(words: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in words:
        for char in word:
            if not re.fullmatch(r"[A-Za-z'-]", char):
                counts[char] = counts.get(char, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _find_missing_translation(word_to_stage: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    missing_field: list[str] = []
    empty_string: list[str] = []

    for word, info in word_to_stage.items():
        if not isinstance(info, dict) or "translation" not in info:
            missing_field.append(word)
            continue
        translation = info.get("translation")
        if isinstance(translation, str) and not translation.strip():
            empty_string.append(word)

    missing_all = missing_field + empty_string
    return missing_all, missing_field, empty_string


def _rebuild_stages(vocab: dict[str, Any]) -> None:
    stages = vocab.get("stages")
    word_to_stage = vocab.get("word_to_stage")
    if not isinstance(stages, dict) or not isinstance(word_to_stage, dict):
        return

    for stage_info in stages.values():
        if isinstance(stage_info, dict):
            stage_info["words"] = []
            stage_info["count"] = 0

    for word, info in word_to_stage.items():
        if not isinstance(info, dict):
            continue
        first_stage = info.get("first_stage")
        all_stages = info.get("all_stages")
        if not isinstance(all_stages, list):
            all_stages = [first_stage] if first_stage else []
        clean_stages = [stage for stage in all_stages if stage in stages]
        if not clean_stages and first_stage in stages:
            clean_stages = [first_stage]
        info["all_stages"] = clean_stages
        if first_stage not in stages and clean_stages:
            info["first_stage"] = clean_stages[0]
        for stage in clean_stages:
            stage_info = stages.get(stage)
            if isinstance(stage_info, dict):
                stage_info["words"].append(word)

    for stage_info in stages.values():
        if isinstance(stage_info, dict):
            words = sorted(set(stage_info.get("words", [])))
            stage_info["words"] = words
            stage_info["count"] = len(words)


def _rebuild_overlap_matrix(vocab: dict[str, Any]) -> None:
    stages = vocab.get("stages")
    if not isinstance(stages, dict):
        return

    stage_words = {
        stage: set(info.get("words", []))
        for stage, info in stages.items()
        if isinstance(info, dict) and isinstance(info.get("words"), list)
    }
    vocab["overlap_matrix"] = {
        left: {
            right: len(left_words & right_words)
            for right, right_words in stage_words.items()
            if right != left
        }
        for left, left_words in stage_words.items()
    }


def _stamp_meta(vocab: dict[str, Any], stats: CleaningStats) -> None:
    meta = vocab.setdefault("meta", {})
    if not isinstance(meta, dict):
        vocab["meta"] = {}
        meta = vocab["meta"]

    meta["vocab_cleaning"] = {
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "script": "scripts/clean_vocab.py",
        "source_file": str(stats.input_path.relative_to(PROJECT_ROOT)),
        "allowed_word_pattern": VALID_WORD_RE.pattern,
        "original_word_count": stats.total_words,
        "removed_dirty_word_count": len(stats.dirty_words),
        "cleaned_word_count": stats.cleaned_words,
        "missing_translation_count_original": len(stats.missing_translation_words),
        "missing_translation_field_count_original": len(stats.missing_translation_field_words),
        "empty_translation_count_original": len(stats.empty_translation_words),
    }


def clean_vocab(spec: VocabSpec) -> CleaningStats:
    original = _load_json(spec.input_path)
    vocab = copy.deepcopy(original)
    word_to_stage = vocab["word_to_stage"]

    dirty_words = [word for word in word_to_stage if not _is_valid_word(word)]
    missing_all, missing_field, empty_string = _find_missing_translation(word_to_stage)

    for word in dirty_words:
        del word_to_stage[word]

    vocab["word_to_stage"] = dict(sorted(word_to_stage.items()))
    _rebuild_stages(vocab)
    _rebuild_overlap_matrix(vocab)

    stats = CleaningStats(
        name=spec.name,
        input_path=spec.input_path,
        output_path=spec.output_path,
        total_words=len(original["word_to_stage"]),
        dirty_words=dirty_words,
        missing_translation_words=missing_all,
        missing_translation_field_words=missing_field,
        empty_translation_words=empty_string,
        cleaned_words=len(vocab["word_to_stage"]),
        invalid_character_counts=_invalid_character_counts(dirty_words),
    )
    _stamp_meta(vocab, stats)

    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    with spec.output_path.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
        f.write("\n")

    return stats


def _format_examples(words: list[str], limit: int = 20) -> str:
    if not words:
        return "- 无\n"
    return "".join(f"- `{word}`\n" for word in words[:limit])


def _format_invalid_chars(counts: dict[str, int], limit: int = 20) -> str:
    if not counts:
        return "无"
    parts = []
    for char, count in list(counts.items())[:limit]:
        display = char if char.strip() else repr(char)
        parts.append(f"`{display}`: {count}")
    return ", ".join(parts)


def write_report(stats_list: list[CleaningStats], report_path: Path) -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    lines = [
        "# 词库清洗报告",
        "",
        f"- 生成时间：{now}",
        f"- 允许保留的词形：`{VALID_WORD_RE.pattern}`（英文字母，允许连字符 `-` 和撇号 `'`）",
        f"- 清洗脚本：`scripts/clean_vocab.py`",
        "",
        "## 汇总",
        "",
        "| 词库 | 输入文件 | 总词数 | 脏词数 | 缺失翻译数 | 缺 translation 字段 | 空翻译字符串 | 清洗后词数 | 输出文件 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for stats in stats_list:
        lines.append(
            "| "
            f"{stats.name} | "
            f"`{stats.input_path.relative_to(PROJECT_ROOT)}` | "
            f"{stats.total_words} | "
            f"{len(stats.dirty_words)} | "
            f"{len(stats.missing_translation_words)} | "
            f"{len(stats.missing_translation_field_words)} | "
            f"{len(stats.empty_translation_words)} | "
            f"{stats.cleaned_words} | "
            f"`{stats.output_path.relative_to(PROJECT_ROOT)}` |"
        )

    combined_dirty: dict[str, list[str]] = {}
    for stats in stats_list:
        for word in stats.dirty_words:
            combined_dirty.setdefault(word, []).append(stats.name)

    lines.extend(
        [
            "",
            "## 跨词库脏词来源",
            "",
            f"- 去重脏词数：{len(combined_dirty)}",
            "- 示例（前20个）：",
        ]
    )
    if combined_dirty:
        for word, sources in list(combined_dirty.items())[:20]:
            lines.append(f"  - `{word}`：{', '.join(sources)}")
    else:
        lines.append("  - 无")

    for stats in stats_list:
        lines.extend(
            [
                "",
                f"## {stats.name}",
                "",
                f"- 输入：`{stats.input_path.relative_to(PROJECT_ROOT)}`",
                f"- 输出：`{stats.output_path.relative_to(PROJECT_ROOT)}`",
                f"- 总词数：{stats.total_words}",
                f"- 脏词数：{len(stats.dirty_words)}",
                f"- 缺失翻译数：{len(stats.missing_translation_words)}",
                f"- 缺 translation 字段：{len(stats.missing_translation_field_words)}",
                f"- 空翻译字符串：{len(stats.empty_translation_words)}",
                f"- 清洗后词库大小：{stats.cleaned_words}",
                f"- 非法字符统计：{_format_invalid_chars(stats.invalid_character_counts)}",
                "",
                "### 脏词示例（前20个）",
                "",
                _format_examples(stats.dirty_words),
                "### 缺失翻译示例（前20个）",
                "",
                _format_examples(stats.missing_translation_words),
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗 v1/v2 stage vocabulary 中的非英文词条并生成报告。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = [VocabSpec(name, input_path, output_path) for name, input_path, output_path in VOCAB_SPECS]
    stats_list = [clean_vocab(spec) for spec in specs]
    write_report(stats_list, args.report)

    print("词库清洗完成")
    for stats in stats_list:
        print(
            f"- {stats.name}: total={stats.total_words}, dirty={len(stats.dirty_words)}, "
            f"missing_translation={len(stats.missing_translation_words)}, cleaned={stats.cleaned_words}, "
            f"output={stats.output_path.relative_to(PROJECT_ROOT)}"
        )
    print(f"- report={args.report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
