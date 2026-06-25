#!/usr/bin/env python3
"""
Merge three vocabulary sources:
  A. pep_textbook - xls extracted stage vocab
  B. mahavivo     - GitHub exam syllabus wordlists
  C. official_vocab - built-in exam vocab sets

Produces updated stage_vocab.json with source tracking and confidence scoring.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path.home() / "stu/vocab_estimator/data"
OPTIM_DIR = Path.home() / "stu/vocab_estimator/optim"
OUTPUT = DATA_DIR / "stage_vocab.json"

# ── Load Source A: pep_textbook (existing stage_vocab.json) ──

with open(OUTPUT) as f:
    existing = json.load(f)

pep_stages_raw = existing["stages"]
pep_wts = existing["word_to_stage"]

# Normalize: pep words might have spaces/punctuation differences
pep_words_by_stage = {}
for sname, sdata in pep_stages_raw.items():
    pep_words_by_stage[sname] = [w.strip() for w in sdata["words"]]

# Build pep word set per stage
pep_word_sets = {}
for sname, words in pep_words_by_stage.items():
    pep_word_sets[sname] = {w.lower().strip() for w in words}

print("=== Source A: pep_textbook ===")
for sname, ws in pep_word_sets.items():
    print(f"  {sname}: {len(ws)} words")

# ── Load Source B: mahavivo ──

def load_mahavivo(path):
    with open(path) as f:
        return {l.strip().lower() for l in f if l.strip()}

gaokao_maha = load_mahavivo(DATA_DIR / "exam_vocab" / "gaokao.txt")
cet6_maha = load_mahavivo(DATA_DIR / "exam_vocab" / "cet6.txt")

print("\n=== Source B: mahavivo ===")
print(f"  gaokao.txt → senior: {len(gaokao_maha)} words")
print(f"  cet6.txt → cet6: {len(cet6_maha)} words")

# ── Load Source C: official_vocab ──

def extract_official_var(source, var_name):
    """Extract word set from a variable like ZHONGKAO_WORDS = ( "..." )"""
    idx = source.find(f'{var_name} = ')
    if idx < 0:
        idx = source.find(f'{var_name}= ')
    assert idx >= 0, f"Could not find {var_name}"
    rest = source[idx:]
    # Find opening paren of the tuple
    paren_start = rest.find('(')
    depth = 0
    all_strings = []
    current_string = ""
    in_string = False
    string_char = None
    for i in range(paren_start, len(rest)):
        ch = rest[i]
        if i == paren_start:
            depth = 1
            continue
        if in_string:
            if ch == '\\':
                continue
            if ch == string_char:
                in_string = False
                all_strings.append(current_string)
                current_string = ""
            else:
                current_string += ch
        else:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                current_string = ""
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
    words = set()
    for s in all_strings:
        for token in s.split():
            token = token.strip().lower()
            if token:
                words.add(token)
    return words

def extract_added_words(source, var_name):
    """Extract the second tuple from VAR = VAR2 + ( ... )"""
    idx = source.find(f'{var_name} = ')
    assert idx >= 0, f"Could not find {var_name}"
    rest = source[idx:]
    plus_idx = rest.find('+ (')
    if plus_idx < 0:
        plus_idx = rest.find('+(')
    assert plus_idx >= 0, f"Could not find + ( in {var_name}"
    tuple_start = rest.find('(', plus_idx)
    depth = 0
    all_strings = []
    current_string = ""
    in_string = False
    string_char = None
    for i in range(tuple_start, len(rest)):
        ch = rest[i]
        if i == tuple_start:
            depth = 1
            continue
        if in_string:
            if ch == '\\':
                continue
            if ch == string_char:
                in_string = False
                all_strings.append(current_string)
                current_string = ""
            else:
                current_string += ch
        else:
            if ch in ('"', "'"):
                in_string = True
                string_char = ch
                current_string = ""
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
    words = set()
    for s in all_strings:
        for token in s.split():
            token = token.strip().lower()
            if token:
                words.add(token)
    return words

with open(OPTIM_DIR / "official_vocab.py") as f:
    ov_source = f.read()

zhongkao_ov = extract_official_var(ov_source, "ZHONGKAO_WORDS")
gaokao_ov = zhongkao_ov | extract_added_words(ov_source, "GAOKAO_WORDS")
cet4_ov = gaokao_ov | extract_added_words(ov_source, "CET4_WORDS")
cet6_ov = cet4_ov | extract_added_words(ov_source, "CET6_WORDS")

print("\n=== Source C: official_vocab ===")
print(f"  ZHONGKAO_WORDS → junior: {len(zhongkao_ov)} words")
print(f"  GAOKAO_WORDS → senior: {len(gaokao_ov)} words")
print(f"  CET4_WORDS → cet4: {len(cet4_ov)} words")
print(f"  CET6_WORDS → cet6: {len(cet6_ov)} words")

# ── Stage ordering (earliest = lowest priority number) ──

STAGE_ORDER = [
    "primary_3", "primary_4", "primary_5", "primary_6",
    "junior_7", "junior_8", "junior_9",
    "senior", "cet4", "cet6", "ielts"
]
STAGE_PRIORITY = {s: i+1 for i, s in enumerate(STAGE_ORDER)}
STAGE_LABELS = {s: pep_stages_raw[s]["label"] for s in STAGE_ORDER if s in pep_stages_raw}
STAGE_LABELS.update({
    "primary_3": "小学三年级", "primary_4": "小学四年级",
    "primary_5": "小学五年级", "primary_6": "小学六年级",
    "junior_7": "七年级", "junior_8": "八年级", "junior_9": "九年级",
    "senior": "高中", "cet4": "大学四级", "cet6": "大学六级", "ielts": "雅思"
})

# ── Source mapping: what stage each external source maps to ──

# mahavivo: gaokao → senior, cet6.txt → cet6
MAHA_STAGE = {
    # (source_name, stage)
}
MAHA_WORDS = {
    "senior": gaokao_maha,
    "cet6": cet6_maha,
}

# official_vocab: ZHONGKAO → junior (all junior stages), GAOKAO → senior, CET4 → cet4, CET6 → cet6
# For ZHONGKAO, the words could be in junior_7, junior_8, or junior_9
# We assign them to the first junior stage they appear in if already present,
# or the earliest matching stage
OV_WORDS = {
    ("zhongkao", "junior_7"): zhongkao_ov,
    ("gaokao", "senior"): gaokao_ov,
    ("cet4", "cet4"): cet4_ov,
    ("cet6", "cet6"): cet6_ov,
}
# Also give zhongkao words a shot at junior_8 and junior_9
OV_WORDS[("zhongkao", "junior_8")] = zhongkao_ov
OV_WORDS[("zhongkao", "junior_9")] = zhongkao_ov

# ── Build merged word_to_stage ──

# Start from pep_textbook data
word_to_stage = {}
for word, info in pep_wts.items():
    w = word.strip().lower()
    word_to_stage[w] = {
        "first_stage": info["first_stage"],
        "all_stages": list(info["all_stages"]),
        "first_priority": info["first_priority"],
        "sources": ["pep_textbook"],
    }

# Pre-populate with pep sources tracking
# Also record pep_stage_precedence
pep_stage_precedence = {}  # word -> earliest stage from pep
for sname, words in pep_word_sets.items():
    for w in words:
        wl = w.lower().strip()
        if wl not in pep_stage_precedence or STAGE_PRIORITY[sname] < STAGE_PRIORITY[pep_stage_precedence[wl]]:
            pep_stage_precedence[wl] = sname

# ── Add mahavivo words ──

STAGE_SOURCE_MAP = {
    "senior": ("mahavivo", "gaokao.txt"),
    "cet6": ("mahavivo", "cet6.txt"),
}

for stage, words in MAHA_WORDS.items():
    src_name, src_file = STAGE_SOURCE_MAP[stage]
    for w in words:
        wl = w.strip().lower()
        if wl in word_to_stage:
            info = word_to_stage[wl]
            # Add source if not already
            if "mahavivo" not in info.get("sources", []):
                info.setdefault("sources", []).append("mahavivo")
            # Update all_stages
            if stage not in info.get("all_stages", []):
                info.setdefault("all_stages", []).append(stage)
                info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
            # Keep first_stage (earliest)
        else:
            # New word from mahavivo
            word_to_stage[wl] = {
                "first_stage": stage,
                "all_stages": [stage],
                "first_priority": STAGE_PRIORITY[stage],
                "sources": [src_name],
            }

# ── Special handling: mahavivo cet6 REPLACES pep_textbook cet6 ──
# Words that were ONLY in pep_textbook cet6 (not in any earlier stage, not in mahavivo cet6)
# should be removed from cet6 stage.
# But they shouldn't be deleted from word_to_stage if they exist elsewhere.
# If they were ONLY in cet6, they stay but are flagged.

for w in list(pep_word_sets.get("cet6", set())):
    wl = w.lower().strip()
    if wl not in cet6_maha and wl in word_to_stage:
        info = word_to_stage[wl]
        # Remove cet6 from all_stages if it was from pep_textbook only
        # But only if the word exists in no earlier stage
        if "cet6" in info.get("all_stages", []):
            # Check if word exists in any earlier stage
            other_stages = [s for s in info["all_stages"] if s != "cet6"]
            if not other_stages:
                # Word was ONLY in cet6 stage from pep_textbook
                # Keep it but note it's pep-only, add "pep_cet6_removed" marker
                info.setdefault("notes", []).append("removed_from_cet6_by_mahavivo")
            else:
                # Word exists in another stage too, just remove cet6 from all_stages
                info["all_stages"] = other_stages
                # If first_stage was cet6, update it (shouldn't happen if word exists in earlier stage)
                if info["first_stage"] == "cet6" and other_stages:
                    info["first_stage"] = other_stages[0]
                    info["first_priority"] = STAGE_PRIORITY[other_stages[0]]

# ── Add official_vocab words ──

for (ov_name, stage), words in OV_WORDS.items():
    for w in words:
        wl = w.strip().lower()
        if wl in word_to_stage:
            info = word_to_stage[wl]
            if "official_vocab" not in info.get("sources", []):
                info.setdefault("sources", []).append("official_vocab")
            if ov_name == "zhongkao":
                # For zhongkao words, try to apply to junior stages
                # Only add if word is already in some junior stage or earlier
                pass  # don't force stage change
            else:
                if stage not in info.get("all_stages", []):
                    info.setdefault("all_stages", []).append(stage)
                    info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
        else:
            # New word from official_vocab
            word_to_stage[wl] = {
                "first_stage": stage,
                "all_stages": [stage],
                "first_priority": STAGE_PRIORITY[stage],
                "sources": ["official_vocab"],
            }

# ── Rebuild stages from word_to_stage ──

# Build new stage word lists
# Keep original behavior: each word appears in ALL stages it belongs to
new_stages = {}
for stage in STAGE_ORDER:
    stage_words = []
    for w, info in word_to_stage.items():
        if stage in info["all_stages"]:
            stage_words.append(w)
    
    new_stages[stage] = {
        "priority": STAGE_PRIORITY[stage],
        "label": STAGE_LABELS.get(stage, stage),
        "words": sorted(stage_words),
    }

# ── Add source_confidence to word_to_stage ──

# Also handle the cet6 REPLACEMENT properly
for w, info in word_to_stage.items():
    sources = info.get("sources", [])
    if len(sources) >= 2:
        info["source_confidence"] = "high"
    elif len(sources) == 1:
        if sources[0] in ("mahavivo", "official_vocab"):
            info["source_confidence"] = "medium"
        else:
            info["source_confidence"] = "low"
    else:
        info["source_confidence"] = "low"

# ── Fix: for cet6, we need to ensure mahavivo words are properly represented ──
# The cet6 stage should have the union of mahavivo cet6 words and words assigned to cet6
# from other sources. Actually, re-read the requirement:
# "以 mahavivo 为权威：cet6.txt (8028词) 替换 xls 提取的 CET-6 (5518词)"
# This means the cet6.stage words should BE the mahavivo cet6 words.

# For the cet6 stage specifically, rebuild from mahavivo
cet6_stage_words = set()
for w in cet6_maha:
    wl = w.strip().lower()
    if wl in word_to_stage:
        # Word exists somewhere - if it was in an earlier stage, it stays there
        # But it also appears in cet6
        info = word_to_stage[wl]
        if "cet6" not in info["all_stages"]:
            info["all_stages"].append("cet6")
            info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
        if "mahavivo" not in info.get("sources", []):
            info.setdefault("sources", []).append("mahavivo")
        # Keep first_stage (earliest)
    else:
        word_to_stage[wl] = {
            "first_stage": "cet6",
            "all_stages": ["cet6"],
            "first_priority": STAGE_PRIORITY["cet6"],
            "sources": ["mahavivo"],
            "source_confidence": "medium",
        }
    cet6_stage_words.add(wl)

# Also add official_vocab cet6 words that might not be in mahavivo
for w in cet6_ov:
    wl = w.strip().lower()
    if wl not in word_to_stage:
        word_to_stage[wl] = {
            "first_stage": "cet6",
            "all_stages": ["cet6"],
            "first_priority": STAGE_PRIORITY["cet6"],
            "sources": ["official_vocab"],
            "source_confidence": "medium",
        }
    elif "cet6" not in word_to_stage[wl]["all_stages"]:
        info = word_to_stage[wl]
        info["all_stages"].append("cet6")
        info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
        if "official_vocab" not in info.get("sources", []):
            info.setdefault("sources", []).append("official_vocab")

# Build new stages again after fixing cet6
new_stages = {}
for stage in STAGE_ORDER:
    stage_words = []
    for w, info in word_to_stage.items():
        if stage in info["all_stages"]:
            stage_words.append(w)
    
    new_stages[stage] = {
        "priority": STAGE_PRIORITY[stage],
        "label": STAGE_LABELS.get(stage, stage),
        "words": sorted(stage_words),
    }

# ── Rebuild overlap matrix ──

overlap_matrix = {}
for s1 in STAGE_ORDER:
    s1_words = set(new_stages[s1]["words"])
    overlap_matrix[s1] = {}
    for s2 in STAGE_ORDER:
        if s1 == s2:
            continue
        s2_words = set(new_stages[s2]["words"])
        overlap = len(s1_words & s2_words)
        if overlap > 0:
            overlap_matrix[s1][s2] = overlap

# ── Build word_to_stage final (including source_confidence) ──

# Rebuild final word to stage
final_word_to_stage = {}
for w, info in word_to_stage.items():
    sources = info.get("sources", ["pep_textbook"])
    source_confidence = info.get("source_confidence", "low")
    
    final_info = {
        "first_stage": info["first_stage"],
        "all_stages": info["all_stages"],
        "sources": sources,
        "source_confidence": source_confidence,
    }
    if "notes" in info:
        final_info["notes"] = info["notes"]
    final_word_to_stage[w] = final_info

# ── Build meta ──

meta = {
    "sources": {
        "pep_textbook": {
            "files": ["stage_vocab.json (original xls extraction)"],
            "description": "人教版教材词表（从 xls/docx 提取）",
        },
        "mahavivo": {
            "files": ["gaokao.txt", "cet6.txt"],
            "description": "GitHub mahavivo 考纲词表",
        },
        "official_vocab": {
            "description": "内置考纲词表 (official_vocab.py)",
        },
    }
}

# ── Build output ──

output = {
    "meta": meta,
    "stages": new_stages,
    "word_to_stage": dict(sorted(final_word_to_stage.items())),
    "overlap_matrix": overlap_matrix,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Merged vocab written to {OUTPUT}")

# ── Statistics ──

print("\n" + "=" * 60)
print("MERGE REPORT")
print("=" * 60)

# Before/After comparison
print("\n--- Stage Word Counts: Before vs After ---")
print(f"{'Stage':<15} {'Before':>8} {'After':>8} {'Δ':>8}")
print("-" * 42)
for sname in STAGE_ORDER:
    before = len(pep_word_sets.get(sname, set()))
    after = len(new_stages.get(sname, {}).get("words", []))
    delta = after - before
    sign = "+" if delta > 0 else ""
    print(f"{sname:<15} {before:>8} {after:>8} {sign}{delta:>7}")

total_before = sum(len(ws) for ws in pep_word_sets.values())
total_after = sum(len(new_stages[s]["words"]) for s in STAGE_ORDER if s in new_stages)
total_delta = total_after - total_before
print("-" * 42)
print(f"{'TOTAL':<15} {total_before:>8} {total_after:>8} {total_delta:+>8}")

# Source coverage matrix
print("\n--- Source Coverage Matrix ---")
# Count how many words have each source combination
source_counts = defaultdict(int)
source_stage_counts = defaultdict(lambda: defaultdict(int))

for w, info in final_word_to_stage.items():
    sources = tuple(sorted(info["sources"]))
    source_counts[sources] += 1
    fs = info["first_stage"]
    for s in sources:
        source_stage_counts[s][fs] += 1

print(f"{'Source Combination':<45} {'# Words':>8}")
print("-" * 55)
for combo, count in sorted(source_counts.items(), key=lambda x: -x[1]):
    combo_str = ", ".join(combo)
    print(f"{combo_str:<45} {count:>8}")

# Per-source total
print(f"\n--- Per-Source Totals ---")
source_totals = defaultdict(int)
for w, info in final_word_to_stage.items():
    for s in info["sources"]:
        source_totals[s] += 1
for src, cnt in sorted(source_totals.items(), key=lambda x: -x[1]):
    print(f"  {src:<20}: {cnt:>6} words")

# Confidence distribution
print(f"\n--- Source Confidence Distribution ---")
conf_dist = defaultdict(int)
for w, info in final_word_to_stage.items():
    conf_dist[info["source_confidence"]] += 1
for conf, cnt in sorted(conf_dist.items()):
    print(f"  {conf:<10}: {cnt:>6} words")

# Conflicts: words that appear in different stages across different sources
# A conflict is when the same word is in different stages from different sources
# We already handle this by taking the earliest stage, but let's report
print("\n--- Stage Assignment Conflicts (word in different stage from different sources) ---")

# Build: for each word, what stage does each source assign it to?
# pep_textbook already has stages
# mahavivo: gaokao→senior, cet6→cet6
# official_vocab: zhongkao→junior_7/8/9, gaokao→senior, cet4→cet4, cet6→cet6

word_source_stage = defaultdict(list)

# pep_textbook
for w, info in pep_wts.items():
    wl = w.strip().lower()
    word_source_stage[wl].append(("pep_textbook", info["first_stage"]))

# mahavivo
for w in gaokao_maha:
    wl = w.strip().lower()
    word_source_stage[wl].append(("mahavivo", "senior"))
for w in cet6_maha:
    wl = w.strip().lower()
    word_source_stage[wl].append(("mahavivo", "cet6"))

# official_vocab
for w in zhongkao_ov:
    wl = w.strip().lower()
    # Assign to the matching pep stage if exists, else junior_9
    if wl in pep_stage_precedence:
        pep_stage = pep_stage_precedence[wl]
        if pep_stage in ("junior_7", "junior_8", "junior_9", "primary_3", "primary_4", "primary_5", "primary_6"):
            word_source_stage[wl].append(("official_vocab", pep_stage))
        else:
            word_source_stage[wl].append(("official_vocab", "junior_9"))
    else:
        word_source_stage[wl].append(("official_vocab", "junior_9"))
for w in gaokao_ov:
    wl = w.strip().lower()
    if wl not in zhongkao_ov:  # only add for words unique to gaokao tier
        word_source_stage[wl].append(("official_vocab", "senior"))
for w in cet4_ov:
    wl = w.strip().lower()
    if wl not in gaokao_ov:
        word_source_stage[wl].append(("official_vocab", "cet4"))
for w in cet6_ov:
    wl = w.strip().lower()
    if wl not in cet4_ov:
        word_source_stage[wl].append(("official_vocab", "cet6"))

# Find conflicts
conflict_count = 0
for w, assignments in word_source_stage.items():
    if len(assignments) >= 2:
        stages = set(a[1] for a in assignments)
        if len(stages) >= 2:
            # This word is assigned to different stages by different sources
            if conflict_count < 30:
                print(f"  '{w}': ", end="")
                for src, stg in assignments:
                    print(f"[{src}→{stg}] ", end="")
                print()
            conflict_count += 1

print(f"  Total conflicts: {conflict_count}")

# Also report: words removed from cet6 by mahavivo replacement
removed_from_cet6 = []
for w, info in final_word_to_stage.items():
    if "notes" in info and "removed_from_cet6_by_mahavivo" in info.get("notes", []):
        removed_from_cet6.append(w)
print(f"\nWords removed from cet6 stage (mahavivo replacement): {len(removed_from_cet6)}")
if removed_from_cet6:
    print(f"  Sample: {sorted(removed_from_cet6)[:20]}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total unique words in merged vocab: {len(final_word_to_stage)}")
print(f"Total unique word-to-stage entries: {len(final_word_to_stage)}")
for conf in ["high", "medium", "low"]:
    cnt = sum(1 for w, info in final_word_to_stage.items() if info["source_confidence"] == conf)
    print(f"  {conf} confidence: {cnt} words")
