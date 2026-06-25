#!/usr/bin/env python3
"""带 checkpoint resume 地补全 data/stage_vocab.json 中待翻译项。"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

STAGE_VOCAB_PATH = PROJECT_ROOT / "data" / "stage_vocab.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "translation_checkpoint.json"
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

BATCH_SIZE = 50
SLEEP_SECONDS = 3
MAX_BATCHES_PER_RUN = int(os.environ.get("MAX_BATCHES_PER_RUN", "10"))


class TranslationError(RuntimeError):
    """当前运行应停止并保留 checkpoint 时抛出。"""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_checkpoint(pending_count: int) -> dict[str, int]:
    if not CHECKPOINT_PATH.exists():
        return {"last_index": 0, "filled": 0, "total": pending_count}

    checkpoint = load_json(CHECKPOINT_PATH)
    total = int(checkpoint.get("total") or pending_count)
    return {
        "last_index": max(0, int(checkpoint.get("last_index", 0))),
        "filled": max(0, int(checkpoint.get("filled", 0))),
        "total": max(total, pending_count),
    }


def save_checkpoint(last_index: int, filled: int, total: int) -> None:
    write_json(
        CHECKPOINT_PATH,
        {"last_index": last_index, "filled": filled, "total": total},
    )


def pending_entries(data: dict[str, Any], start_index: int) -> list[tuple[int, str]]:
    entries = []
    for index, (word, info) in enumerate(data["word_to_stage"].items()):
        if index < start_index:
            continue
        if info.get("translation_pending"):
            entries.append((index, word))
    return entries


def count_pending(data: dict[str, Any]) -> int:
    return sum(
        1
        for info in data["word_to_stage"].values()
        if info.get("translation_pending")
    )


def update_translation_meta(data: dict[str, Any]) -> None:
    word_to_stage = data["word_to_stage"]
    translated = sum(1 for info in word_to_stage.values() if info.get("translation"))
    pending = count_pending(data)
    cleanup = data.setdefault("meta", {}).setdefault("cleanup", {})
    cleanup["translation_filled"] = translated
    cleanup["translation_pending"] = pending


def _map_translation(
    results: dict[str, str],
    original: str,
    translated: str,
) -> None:
    original = original.strip()
    translated = translated.strip()
    if not original or not translated:
        return

    original_lines = [line.strip() for line in original.splitlines() if line.strip()]
    translated_lines = [
        line.strip() for line in translated.splitlines() if line.strip()
    ]
    if len(original_lines) == len(translated_lines) and len(original_lines) > 1:
        for src, dst in zip(original_lines, translated_lines):
            results[src.lower()] = dst
        return

    results[original.lower()] = translated


def translate_batch(words: list[str]) -> dict[str, str]:
    params = {
        "client": "gtx",
        "sl": "en",
        "tl": "zh-CN",
        "dt": "t",
        "q": "\n".join(words),
    }

    try:
        response = requests.get(TRANSLATE_URL, params=params, timeout=30)
        if response.status_code == 429:
            raise TranslationError("HTTP 429 rate limit")
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise TranslationError(f"network/API error: {exc}") from exc
    except ValueError as exc:
        raise TranslationError(f"invalid API JSON response: {exc}") from exc

    results: dict[str, str] = {}
    for entry in data[0]:
        if isinstance(entry, list) and len(entry) >= 2:
            _map_translation(results, str(entry[1] or ""), str(entry[0] or ""))

    if not results:
        raise TranslationError("API returned no translations")

    return results


def apply_translations(data: dict[str, Any], words: list[str], translations: dict[str, str]) -> int:
    filled = 0
    word_to_stage = data["word_to_stage"]
    for word in words:
        translation = translations.get(word.lower())
        if not translation:
            continue
        info = word_to_stage[word]
        info["translation"] = translation
        info.pop("translation_pending", None)
        filled += 1
    return filled


def main() -> int:
    data = load_json(STAGE_VOCAB_PATH)
    initial_pending = count_pending(data)
    checkpoint = load_checkpoint(initial_pending)
    total = checkpoint["total"] or initial_pending
    last_index = checkpoint["last_index"]

    if initial_pending == 0:
        save_checkpoint(last_index, total, total)
        print(f"已翻译 {total}/{total}，下一批从索引 {last_index} 开始")
        return 0

    batches_run = 0
    while batches_run < MAX_BATCHES_PER_RUN:
        entries = pending_entries(data, last_index)
        if not entries:
            last_index = 0
            entries = pending_entries(data, last_index)
        if not entries:
            break

        batch = entries[:BATCH_SIZE]
        batch_words = [word for _, word in batch]
        batch_no = batches_run + 1
        print(
            f"Batch {batch_no}: indexes {batch[0][0]}-{batch[-1][0]} "
            f"({len(batch_words)} words)...",
            flush=True,
        )

        try:
            translations = translate_batch(batch_words)
        except TranslationError as exc:
            remaining = count_pending(data)
            filled = max(checkpoint["filled"], total - remaining)
            save_checkpoint(last_index, filled, total)
            print(f"遇到错误：{exc}")
            print(f"已翻译 {filled}/{total}，下一批从索引 {last_index} 开始")
            return 0

        filled_now = apply_translations(data, batch_words, translations)
        last_index = batch[-1][0] + 1
        update_translation_meta(data)
        write_json(STAGE_VOCAB_PATH, data)

        remaining = count_pending(data)
        filled = max(checkpoint["filled"] + filled_now, total - remaining)
        checkpoint = {"last_index": last_index, "filled": filled, "total": total}
        save_checkpoint(last_index, filled, total)

        missing = len(batch_words) - filled_now
        print(f"  写入 {filled_now} 个翻译，未返回 {missing} 个")
        print(f"已翻译 {filled}/{total}，下一批从索引 {last_index} 开始")

        batches_run += 1
        if batches_run < MAX_BATCHES_PER_RUN and count_pending(data) > 0:
            time.sleep(SLEEP_SECONDS)

    remaining = count_pending(data)
    filled = total - remaining
    save_checkpoint(last_index, filled, total)
    if remaining:
        print(f"本次已跑 {batches_run} 批，已翻译 {filled}/{total}，下一批从索引 {last_index} 开始")
    else:
        print(f"全部完成：已翻译 {total}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
