#!/usr/bin/env python3
"""Fill ALL missing Chinese translations using Google Translate API.

Covers 8k through 30k buckets (excludes 1k-5k which are 100% covered).
"""

import os, sys, re, time, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import requests
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.vocab_bank import VocabBank
from server.translations import TRANSLATIONS

bank = VocabBank(DEFAULT_CONFIG)
existing = {k.lower(): k for k in TRANSLATIONS}

# Collect ALL missing words from ALL buckets (skip 1k-5k which are complete)
missing_words = []
skip_buckets = {'1k', '2k', '3k', '5k'}
sizes = bank.bucket_sizes()

for item in bank.items:
    bucket = bank.get_bucket(item.word)
    if bucket in skip_buckets:
        continue
    w = item.word.lower()
    if w in existing:
        continue
    lemma = bank.lemmatizer.normalize(w).lower()
    if lemma in existing:
        continue
    # Skip very short words, acronyms, punctuation-like
    if len(w) < 2:
        continue
    if w.isupper() and len(w) <= 4:
        continue
    if not re.match(r'^[a-z][a-z\-\']*[a-z]$', w) and not re.match(r'^[a-z]{2,}$', w):
        continue
    missing_words.append((item.rank, w, bucket))

print(f"Total missing words: {len(missing_words)}")
print()

# Group by bucket
by_bucket = {}
for rank, word, bucket in missing_words:
    by_bucket.setdefault(bucket, []).append((rank, word))
for b in ['8k','10k','15k','20k','30k']:
    if b in by_bucket:
        print(f"  {b}: {len(by_bucket[b])} missing")

BATCH_SIZE = 150  # words per API call
TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"

def translate_batch(words):
    """Translate a list of words using Google Translate API."""
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

# Process in batches
all_translations = {}
total_batches = (len(missing_words) + BATCH_SIZE - 1) // BATCH_SIZE

for start in range(0, len(missing_words), BATCH_SIZE):
    batch = missing_words[start:start + BATCH_SIZE]
    batch_num = start // BATCH_SIZE + 1
    
    words = [w for _, w, _ in batch]
    print(f"\rBatch {batch_num}/{total_batches} ({len(words)} words)...", end="", flush=True)
    
    results = translate_batch(words)
    all_translations.update(results)
    print(f" got {len(results)} good", end="", flush=True)
    
    # Rate limit: 1 second between batches
    time.sleep(1.0)

print(f"\n\nTotal translations generated: {len(all_translations)}")

# Merge into translations.py
if all_translations:
    path = os.path.join(PROJECT_ROOT, "server", "translations.py")
    with open(path, "r") as f:
        content = f.read()
    
    # Find the closing brace
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
            if gloss and len(gloss) <= 30:
                new_entries.append(f'    "{word}": "{gloss}",')
        new_lines = lines[:insert_at] + new_entries + lines[insert_at:]
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
        print(f"Appended {len(new_entries)} translations to translations.py")
        
        # Verify
        from server.translations import TRANSLATIONS
        known = [w for w in [item.word for item in bank.items] if w.lower() in {k.lower(): k for k in TRANSLATIONS}]
        print(f"New total coverage: {len(known)}/{len(bank.items)} ({len(known)/len(bank.items)*100:.1f}%)")
    else:
        print("ERROR: Could not find insertion point in translations.py")
else:
    print("ERROR: No translations generated")
