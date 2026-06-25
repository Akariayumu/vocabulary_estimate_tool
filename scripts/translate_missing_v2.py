#!/usr/bin/env python3
"""补全 data/stage_vocab_clean_v2.json 中缺失的 translation 字段。

优先复用 server.translations.TRANSLATIONS；仍缺失的词再通过免费的
Google Translate API 批量翻译。使用 --dry-run 时只打印统计，不写文件、
不调用 API。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT_ROOT))

from server.translations import TRANSLATIONS  # noqa: E402

VOCAB_PATH = PROJECT_ROOT / "data" / "stage_vocab_clean_v2.json"
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
BATCH_SIZE = 150
SAVE_EVERY = 500
SLEEP_SECONDS = 0.5


class TranslationError(RuntimeError):
    """当前批次翻译失败。"""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def is_missing_translation(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    translation = info.get("translation")
    return not isinstance(translation, str) or not translation.strip()


def missing_words(word_to_stage: dict[str, Any]) -> list[str]:
    return [
        word
        for word, info in word_to_stage.items()
        if is_missing_translation(info)
    ]


def normalized_existing_translations() -> dict[str, str]:
    return {
        word.lower(): translation.strip()
        for word, translation in TRANSLATIONS.items()
        if isinstance(word, str)
        and isinstance(translation, str)
        and translation.strip()
    }


def map_translation(
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
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise TranslationError(f"network/API error: {exc}") from exc
    except ValueError as exc:
        raise TranslationError(f"invalid API JSON response: {exc}") from exc

    results: dict[str, str] = {}
    for entry in data[0]:
        if isinstance(entry, list) and len(entry) >= 2:
            map_translation(results, str(entry[1] or ""), str(entry[0] or ""))

    if not results:
        raise TranslationError("API returned no translations")

    return results


def apply_existing_translations(
    word_to_stage: dict[str, Any],
    words: list[str],
    existing: dict[str, str],
    data: dict[str, Any],
) -> tuple[list[str], int]:
    api_words: list[str] = []
    filled = 0
    since_save = 0

    for word in words:
        translation = existing.get(word.lower())
        if not translation:
            api_words.append(word)
            continue

        word_to_stage[word]["translation"] = translation
        filled += 1
        since_save += 1
        if since_save >= SAVE_EVERY:
            write_json(VOCAB_PATH, data)
            print(f"已从现有翻译写入 {filled} 个，增量保存。")
            since_save = 0

    if filled and since_save:
        write_json(VOCAB_PATH, data)

    return api_words, filled


def apply_api_translations(
    word_to_stage: dict[str, Any],
    words: list[str],
    translations: dict[str, str],
) -> int:
    filled = 0
    for word in words:
        translation = translations.get(word.lower())
        if not translation:
            continue
        word_to_stage[word]["translation"] = translation
        filled += 1
    return filled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计缺失数量和可复用翻译数量，不写文件、不调用 API。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_json(VOCAB_PATH)
    word_to_stage = data["word_to_stage"]
    existing = normalized_existing_translations()

    initial_missing_words = missing_words(word_to_stage)
    reusable = [
        word for word in initial_missing_words if existing.get(word.lower())
    ]
    api_needed = [
        word for word in initial_missing_words if not existing.get(word.lower())
    ]

    print(f"词条总数: {len(word_to_stage)}")
    print(f"缺失 translation: {len(initial_missing_words)}")
    print(f"可复用现有翻译: {len(reusable)}")
    print(f"需要调用 API 翻译: {len(api_needed)}")

    if args.dry_run:
        print("dry-run: 未写入文件，未调用 API。")
        return 0

    api_words, filled_from_existing = apply_existing_translations(
        word_to_stage,
        initial_missing_words,
        existing,
        data,
    )

    api_filled = 0
    api_processed = 0
    since_save = 0
    total_batches = (len(api_words) + BATCH_SIZE - 1) // BATCH_SIZE

    for start in range(0, len(api_words), BATCH_SIZE):
        batch = api_words[start:start + BATCH_SIZE]
        batch_no = start // BATCH_SIZE + 1
        print(
            f"Batch {batch_no}/{total_batches} ({len(batch)} words)...",
            end=" ",
            flush=True,
        )

        try:
            translations = translate_batch(batch)
        except TranslationError as exc:
            write_json(VOCAB_PATH, data)
            print(f"\nAPI 错误：{exc}")
            print("已保存当前已写入的翻译，可稍后重新运行继续补全。")
            return 1

        filled_now = apply_api_translations(word_to_stage, batch, translations)
        api_filled += filled_now
        api_processed += len(batch)
        since_save += len(batch)
        print(f"got {len(translations)} translations, wrote {filled_now}")

        if since_save >= SAVE_EVERY:
            write_json(VOCAB_PATH, data)
            print(f"已处理 API 词 {api_processed}/{len(api_words)}，增量保存。")
            since_save = 0

        if start + BATCH_SIZE < len(api_words):
            time.sleep(SLEEP_SECONDS)

    write_json(VOCAB_PATH, data)
    final_missing = len(missing_words(word_to_stage))

    print("\n完成统计:")
    print(f"初始缺失: {len(initial_missing_words)}")
    print(f"现有翻译填充: {filled_from_existing}")
    print(f"API 请求词数: {len(api_words)}")
    print(f"API 成功写入: {api_filled}")
    print(f"API 未返回/未写入: {len(api_words) - api_filled}")
    print(f"剩余缺失: {final_missing}")
    print(f"输出文件: {VOCAB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
