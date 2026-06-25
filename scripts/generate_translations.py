#!/usr/bin/env python3
"""使用免费 Google Translate API 批量生成未翻译词（rank 5001-15000）的中文翻译。

用法：python3 scripts/generate_translations.py
"""

import os, sys, re, time, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
from server.translations import TRANSLATIONS

bank = VocabBank(DEFAULT_CONFIG)
existing = {k.lower(): v for k, v in TRANSLATIONS.items()}

# 查找 rank 5001-15000 中所有缺失翻译
missing_words = []
for item in bank.items:
    if not (5001 <= item.rank <= 15000):
        continue
    w = item.word.lower()
    if w in existing:
        continue
    lemma = bank.lemmatizer.normalize(w).lower()
    if lemma in existing:
        continue
    if len(w) < 3 or (w.isupper() and len(w) <= 4):
        continue
    missing_words.append((item.rank, w))

# 限制为约 4500 个词
MAX_WORDS = 4500
missing_words = missing_words[:MAX_WORDS]
print(f"Missing translations in rank 5001-15000: {len(missing_words)}")

BATCH_SIZE = 150  # 每次 API 调用的词数
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

def translate_batch(words):
    """使用 Google Translate API 翻译词列表。"""
    text = "\n".join(words)
    params = {
        "client": "gtx",
        "sl": "en",
        "tl": "zh-CN",
        "dt": "t",
        "q": text,
    }
    try:
        r = requests.get(TRANSLATE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        # 解析：data[0] 是 [translation, original, ...] 列表
        results = {}
        for entry in data[0]:
            if isinstance(entry, list) and len(entry) >= 2:
                translated = entry[0].strip()
                original = entry[1].strip()
                if translated and original:
                    results[original.lower()] = translated
        return results
    except Exception as e:
        print(f"  API error: {e}")
        return {}

# 分批处理
all_translations = {}
for start in range(0, len(missing_words), BATCH_SIZE):
    batch = missing_words[start:start + BATCH_SIZE]
    batch_num = start // BATCH_SIZE + 1
    total_batches = (len(missing_words) + BATCH_SIZE - 1) // BATCH_SIZE
    
    words = [w for _, w in batch]
    print(f"Batch {batch_num}/{total_batches} ({len(words)} words)...", end=" ", flush=True)
    
    results = translate_batch(words)
    all_translations.update(results)
    print(f"got {len(results)} translations")
    
    # 速率限制：批次间隔 1 秒
    time.sleep(1.0)

print(f"\nTotal translations generated: {len(all_translations)}")

# 写入 translations.py
if all_translations:
    path = os.path.join(PROJECT_ROOT, "server", "translations.py")
    with open(path, "r") as f:
        content = f.read()
    
    lines = content.split("\n")
    insert_at = None
    for i in range(len(lines) - 1, -1, -1):
        s = lines[i].strip()
        if s and s[0] == "}":
            insert_at = i
            break
    
    if insert_at is not None:
        new_entries = []
        for word in sorted(all_translations):
            gloss = all_translations[word]
            if gloss and len(gloss) <= 20:
                new_entries.append(f'    "{word}": "{gloss}",')
        new_lines = lines[:insert_at] + new_entries + lines[insert_at:]
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
        print(f"Appended {len(new_entries)} translations to translations.py")
    else:
        print("ERROR: Could not find insertion point in translations.py")
else:
    print("ERROR: No translations generated")
