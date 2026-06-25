#!/usr/bin/env python3
"""
合并三个词汇来源：
  A. pep_textbook - 从 xls 提取的阶段词库
  B. mahavivo     - GitHub 考试大纲词表
  C. official_vocab - 内置考试词汇集

生成带来源追踪和置信度评分的更新版 stage_vocab.json。
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path.home() / "stu/vocab_estimator/data"
OPTIM_DIR = Path.home() / "stu/vocab_estimator/optim"
OUTPUT = DATA_DIR / "stage_vocab.json"

# ── 加载来源 A：pep_textbook（现有 stage_vocab.json）──

with open(OUTPUT) as f:
    existing = json.load(f)

pep_stages_raw = existing["stages"]
pep_wts = existing["word_to_stage"]

# 归一化：pep 词可能存在空格/标点差异
pep_words_by_stage = {}
for sname, sdata in pep_stages_raw.items():
    pep_words_by_stage[sname] = [w.strip() for w in sdata["words"]]

# 按阶段构建 pep 词集合
pep_word_sets = {}
for sname, words in pep_words_by_stage.items():
    pep_word_sets[sname] = {w.lower().strip() for w in words}

print("=== Source A: pep_textbook ===")
for sname, ws in pep_word_sets.items():
    print(f"  {sname}: {len(ws)} words")

# ── 加载来源 B：mahavivo ──

def load_mahavivo(path):
    with open(path) as f:
        return {l.strip().lower() for l in f if l.strip()}

gaokao_maha = load_mahavivo(DATA_DIR / "exam_vocab" / "gaokao.txt")
cet6_maha = load_mahavivo(DATA_DIR / "exam_vocab" / "cet6.txt")

print("\n=== Source B: mahavivo ===")
print(f"  gaokao.txt → senior: {len(gaokao_maha)} words")
print(f"  cet6.txt → cet6: {len(cet6_maha)} words")

# ── 加载来源 C：official_vocab ──

def extract_official_var(source, var_name):
    """从 ZHONGKAO_WORDS = ( "..." ) 这类变量中提取词集合"""
    idx = source.find(f'{var_name} = ')
    if idx < 0:
        idx = source.find(f'{var_name}= ')
    assert idx >= 0, f"Could not find {var_name}"
    rest = source[idx:]
    # 查找 tuple 的左括号
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
    """从 VAR = VAR2 + ( ... ) 中提取第二个 tuple"""
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

# ── 阶段排序（越早 = priority 数字越低）──

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

# ── 来源映射：每个外部来源映射到哪个阶段 ──

# mahavivo 映射：gaokao → senior，cet6.txt → cet6
MAHA_STAGE = {
# (来源名, stage)
}
MAHA_WORDS = {
    "senior": gaokao_maha,
    "cet6": cet6_maha,
}

# official_vocab：ZHONGKAO → junior（全部 junior 阶段），GAOKAO → senior，CET4 → cet4，CET6 → cet6
# 对 ZHONGKAO，词可能属于 junior_7、junior_8 或 junior_9
# 如果词已存在，则分配到它首次出现的 junior 阶段；
# 否则分配到最早匹配阶段
OV_WORDS = {
    ("zhongkao", "junior_7"): zhongkao_ov,
    ("gaokao", "senior"): gaokao_ov,
    ("cet4", "cet4"): cet4_ov,
    ("cet6", "cet6"): cet6_ov,
}
# 也让 zhongkao 词尝试匹配 junior_8 和 junior_9
OV_WORDS[("zhongkao", "junior_8")] = zhongkao_ov
OV_WORDS[("zhongkao", "junior_9")] = zhongkao_ov

# ── 构建合并后的 word_to_stage ──

# 从 pep_textbook 数据开始
word_to_stage = {}
for word, info in pep_wts.items():
    w = word.strip().lower()
    word_to_stage[w] = {
        "first_stage": info["first_stage"],
        "all_stages": list(info["all_stages"]),
        "first_priority": info["first_priority"],
        "sources": ["pep_textbook"],
    }

# 预填充 pep 来源追踪
# 同时记录 pep_stage_precedence
pep_stage_precedence = {}  # word -> pep 中最早阶段
for sname, words in pep_word_sets.items():
    for w in words:
        wl = w.lower().strip()
        if wl not in pep_stage_precedence or STAGE_PRIORITY[sname] < STAGE_PRIORITY[pep_stage_precedence[wl]]:
            pep_stage_precedence[wl] = sname

# ── 添加 mahavivo 词 ──

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
            # 如果尚未记录，则添加来源
            if "mahavivo" not in info.get("sources", []):
                info.setdefault("sources", []).append("mahavivo")
            # 更新 all_stages
            if stage not in info.get("all_stages", []):
                info.setdefault("all_stages", []).append(stage)
                info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
            # 保留 first_stage（最早阶段）
        else:
            # 来自 mahavivo 的新词
            word_to_stage[wl] = {
                "first_stage": stage,
                "all_stages": [stage],
                "first_priority": STAGE_PRIORITY[stage],
                "sources": [src_name],
            }

# ── 特殊处理：mahavivo cet6 替换 pep_textbook cet6 ──
# 仅出现在 pep_textbook cet6 的词（不在任何更早阶段，也不在 mahavivo cet6 中）
# 应从 cet6 阶段移除。
# 但如果它们还存在于别处，不应从 word_to_stage 删除。
# 如果它们只在 cet6 中，则保留但打标。

for w in list(pep_word_sets.get("cet6", set())):
    wl = w.lower().strip()
    if wl not in cet6_maha and wl in word_to_stage:
        info = word_to_stage[wl]
        # 若 cet6 仅来自 pep_textbook，则从 all_stages 移除 cet6
        # 但仅当该词不存在于任何更早阶段时这样做
        if "cet6" in info.get("all_stages", []):
            # 检查词是否存在于任何更早阶段
            other_stages = [s for s in info["all_stages"] if s != "cet6"]
            if not other_stages:
                # 该词仅来自 pep_textbook 的 cet6 阶段
                # 保留它，但注明它是 pep-only，并添加 "pep_cet6_removed" 标记
                info.setdefault("notes", []).append("removed_from_cet6_by_mahavivo")
            else:
                # 词也存在于其他阶段时，只从 all_stages 移除 cet6
                info["all_stages"] = other_stages
                # 如果 first_stage 是 cet6，则更新它（若词存在于更早阶段则不应发生）
                if info["first_stage"] == "cet6" and other_stages:
                    info["first_stage"] = other_stages[0]
                    info["first_priority"] = STAGE_PRIORITY[other_stages[0]]

# ── 添加 official_vocab 词 ──

for (ov_name, stage), words in OV_WORDS.items():
    for w in words:
        wl = w.strip().lower()
        if wl in word_to_stage:
            info = word_to_stage[wl]
            if "official_vocab" not in info.get("sources", []):
                info.setdefault("sources", []).append("official_vocab")
            if ov_name == "zhongkao":
                # 对 zhongkao 词，尝试应用到 junior 阶段
                # 仅当词已存在于某个 junior 阶段或更早阶段时添加
                pass  # 不强制改变阶段
            else:
                if stage not in info.get("all_stages", []):
                    info.setdefault("all_stages", []).append(stage)
                    info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
        else:
            # 来自 official_vocab 的新词
            word_to_stage[wl] = {
                "first_stage": stage,
                "all_stages": [stage],
                "first_priority": STAGE_PRIORITY[stage],
                "sources": ["official_vocab"],
            }

# ── 根据 word_to_stage 重建 stages ──

# 构建新的阶段词表
# 保持原行为：每个词出现在它所属的全部 stages 中
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

# ── 向 word_to_stage 添加 source_confidence ──

# 同时正确处理 cet6 替换
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

# ── 修复：对 cet6，需确保 mahavivo 词被正确表示 ──
# cet6 阶段应包含 mahavivo cet6 词和分配到 cet6 的词的并集
# 来自其他来源。重新阅读需求后可知：
# "以 mahavivo 为权威：cet6.txt (8028词) 替换 xls 提取的 CET-6 (5518词)"
# 这意味着 cet6.stage 词应当就是 mahavivo cet6 词。

# 对 cet6 阶段，专门从 mahavivo 重建
cet6_stage_words = set()
for w in cet6_maha:
    wl = w.strip().lower()
    if wl in word_to_stage:
        # 词存在于某处：若它在更早阶段，则保留在那里
        # 但它也会出现在 cet6 中
        info = word_to_stage[wl]
        if "cet6" not in info["all_stages"]:
            info["all_stages"].append("cet6")
            info["all_stages"].sort(key=lambda s: STAGE_PRIORITY.get(s, 99))
        if "mahavivo" not in info.get("sources", []):
            info.setdefault("sources", []).append("mahavivo")
        # 保留 first_stage（最早阶段）
    else:
        word_to_stage[wl] = {
            "first_stage": "cet6",
            "all_stages": ["cet6"],
            "first_priority": STAGE_PRIORITY["cet6"],
            "sources": ["mahavivo"],
            "source_confidence": "medium",
        }
    cet6_stage_words.add(wl)

# 同时添加可能不在 mahavivo 中的 official_vocab cet6 词
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

# 修复 cet6 后再次构建新 stages
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

# ── 重建 overlap matrix ──

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

# ── 构建最终 word_to_stage（包含 source_confidence）──

# 重建最终 word_to_stage
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

# ── 构建 meta ──

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

# ── 构建输出 ──

output = {
    "meta": meta,
    "stages": new_stages,
    "word_to_stage": dict(sorted(final_word_to_stage.items())),
    "overlap_matrix": overlap_matrix,
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n✅ Merged vocab written to {OUTPUT}")

# ── 统计 ──

print("\n" + "=" * 60)
print("MERGE REPORT")
print("=" * 60)

# 前后对比
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

# 来源覆盖矩阵
print("\n--- Source Coverage Matrix ---")
# 统计每种来源组合包含多少词
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

# 每个来源总数
print(f"\n--- Per-Source Totals ---")
source_totals = defaultdict(int)
for w, info in final_word_to_stage.items():
    for s in info["sources"]:
        source_totals[s] += 1
for src, cnt in sorted(source_totals.items(), key=lambda x: -x[1]):
    print(f"  {src:<20}: {cnt:>6} words")

# 置信度分布
print(f"\n--- Source Confidence Distribution ---")
conf_dist = defaultdict(int)
for w, info in final_word_to_stage.items():
    conf_dist[info["source_confidence"]] += 1
for conf, cnt in sorted(conf_dist.items()):
    print(f"  {conf:<10}: {cnt:>6} words")

# 冲突：在不同来源中出现在不同阶段的词
# 冲突指同一个词在不同来源中对应不同阶段
# 我们已通过选取最早阶段处理此问题，但仍进行报告
print("\n--- Stage Assignment Conflicts (word in different stage from different sources) ---")

# 构建：每个来源给每个词分配了什么阶段？
# pep_textbook 已有 stages
# mahavivo 映射：gaokao→senior，cet6→cet6
# official_vocab 映射：zhongkao→junior_7/8/9，gaokao→senior，cet4→cet4，cet6→cet6

word_source_stage = defaultdict(list)

# pep_textbook 来源
for w, info in pep_wts.items():
    wl = w.strip().lower()
    word_source_stage[wl].append(("pep_textbook", info["first_stage"]))

# mahavivo 来源
for w in gaokao_maha:
    wl = w.strip().lower()
    word_source_stage[wl].append(("mahavivo", "senior"))
for w in cet6_maha:
    wl = w.strip().lower()
    word_source_stage[wl].append(("mahavivo", "cet6"))

# official_vocab 来源
for w in zhongkao_ov:
    wl = w.strip().lower()
    # 若存在匹配的 pep 阶段则分配到该阶段，否则分配到 junior_9
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
    if wl not in zhongkao_ov:  # 只为 gaokao tier 独有词添加
        word_source_stage[wl].append(("official_vocab", "senior"))
for w in cet4_ov:
    wl = w.strip().lower()
    if wl not in gaokao_ov:
        word_source_stage[wl].append(("official_vocab", "cet4"))
for w in cet6_ov:
    wl = w.strip().lower()
    if wl not in cet4_ov:
        word_source_stage[wl].append(("official_vocab", "cet6"))

# 查找冲突
conflict_count = 0
for w, assignments in word_source_stage.items():
    if len(assignments) >= 2:
        stages = set(a[1] for a in assignments)
        if len(stages) >= 2:
            # 该词被不同来源分配到不同阶段
            if conflict_count < 30:
                print(f"  '{w}': ", end="")
                for src, stg in assignments:
                    print(f"[{src}→{stg}] ", end="")
                print()
            conflict_count += 1

print(f"  Total conflicts: {conflict_count}")

# 同时报告：因 mahavivo 替换而从 cet6 移除的词
removed_from_cet6 = []
for w, info in final_word_to_stage.items():
    if "notes" in info and "removed_from_cet6_by_mahavivo" in info.get("notes", []):
        removed_from_cet6.append(w)
print(f"\nWords removed from cet6 stage (mahavivo replacement): {len(removed_from_cet6)}")
if removed_from_cet6:
    print(f"  Sample: {sorted(removed_from_cet6)[:20]}")

# 总结
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total unique words in merged vocab: {len(final_word_to_stage)}")
print(f"Total unique word-to-stage entries: {len(final_word_to_stage)}")
for conf in ["high", "medium", "low"]:
    cnt = sum(1 for w, info in final_word_to_stage.items() if info["source_confidence"] == conf)
    print(f"  {conf} confidence: {cnt} words")
