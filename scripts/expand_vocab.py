#!/usr/bin/env python3
"""用考试词表中缺失的学术/高价值词扩展 stage_vocab.json。"""

import json, re, bisect, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_PATH = PROJECT_ROOT / "data" / "stage_vocab.json"
COCA_PATH = PROJECT_ROOT / "data" / "exam_vocab" / "coca20000.txt"
TOEFL_PATH = PROJECT_ROOT / "data" / "exam_vocab" / "toefl.txt"
GRE_PATH = PROJECT_ROOT / "data" / "exam_vocab" / "gre.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "vocab_expansion_report.md"

STAGE_PRIORITY = {"primary_3":1,"primary_4":2,"primary_5":3,"primary_6":4,
                  "junior_7":5,"junior_8":6,"junior_9":7,"senior":8,
                  "cet4":9,"cet6":10,"ielts":11}
STAGE_NAMES = {1:"primary_3",2:"primary_4",3:"primary_5",4:"primary_6",
               5:"junior_7",6:"junior_8",7:"junior_9",8:"senior",
               9:"cet4",10:"cet6",11:"ielts"}

# 现代高价值领域词（技术、学术、环境等）
MODERN_HIGH_VALUE = {
    "algorithmic","blockchain","cryptocurrency","societal","geopolitical",
    "epistemological","epistemology","hegemony","governance","mitigation",
    "biodiversity","neoliberal","anthropogenic","decarbonization",
    "postcolonial","sustainability","intersectional","neocolonial",
    "globalization","industrialization","interdisciplinary",
    "algorithmically","sociologically","stakeholder","stakeholders",
    "benchmark","benchmarking","scalable","scalability","ecosystem",
    "ecosystems","infrastructure","infrastructural","accountability",
    "transparency","legitimacy","legitimate","jurisdiction",
    "protectionism","protectionist","populism","populist",
    "authoritarianism","authoritarian","ideological","ideologically",
    "disinformation","misinformation","surveillance","cybersecurity",
    "automation","computational","quantum","neoliberalism",
    "sustainability","sustainable","redistribute","redistribution",
    "biotechnology","nanotechnology","semiconductor","microprocessor",
    "algorithm","algorithms","hegemony","hegemonic","intersectionality",
    "institutional","institutionalize","unilateral","multilateral",
    "biopolitics","biopolitical","digitization","actionable",
    "evidence-based","data-driven","AI-driven","ground-breaking",
    "large-scale","high-level","low-level","high-impact","low-impact",
    "cutting-edge","state-of-the-art","real-time","real-world",
    "open-source","closed-source","full-stack","deep-learning",
    "machine-learning","natural-language","cross-cultural",
    "decision-making","problem-solving","goal-oriented",
    "collaboration","collaborative","participatory",
    # C.txt 专用
    "inadvertently","multifaceted","accountability","systemic",
    "unprecedented","profound","detrimental","stringent",
    "promulgate","exacerbate","marginalize","marginalization",
}

def load_vocab_dict(path: Path) -> dict[str, int]:
    """加载词表：word -> rank（从 1 开始的行号）。接受字母和连字符词。"""
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = (line.strip().split()[0] if line.strip() else "").lower().strip()
        if not raw:
            continue
        clean = raw.replace("-", "")
        if clean.isalpha() and raw not in result:
            result[raw] = len(result) + 1
    return result

def compute_difficulty(priority: float, norm_rank: float, alpha=0.60, beta=0.40) -> float:
    return round(alpha * priority + beta * norm_rank, 4)

def estimate_priority_from_rank(rank: int) -> float:
    knots = [(0,4.5),(500,7.0),(1000,7.5),(2000,8.0),(3000,8.5),
             (5000,9.0),(8000,9.5),(10000,10.0),(18000,10.5),(30000,11.0)]
    if rank <= knots[0][0]: return knots[0][1]
    if rank >= knots[-1][0]: return knots[-1][1]
    for i in range(len(knots)-1):
        r1,p1 = knots[i]; r2,p2 = knots[i+1]
        if r1 <= rank <= r2:
            t = (rank - r1) / (r2 - r1) if r2 != r1 else 0.0
            return p1 + t * (p2 - p1)
    return knots[-1][1]

def is_noise(word: str) -> bool:
    if len(word) <= 2: return True
    if re.match(r"^[a-z]'[a-z]+$", word): return True
    if re.match(r"^[a-z]*([^aeiou]{5,})[a-z]*$", word): return True
    if word in {"vs","ie","eg","et","al","etc","ps","nb"}: return True
    return False

def has_uppercase(word: str) -> bool:
    return any(c.isupper() for c in word) and not word.isupper()

def main():
    print("Loading stage_vocab.json...")
    base = json.loads(STAGE_PATH.read_text(encoding="utf-8"))
    existing = set(base["word_to_stage"].keys())
    print(f"  Existing words: {len(existing)}")

    # 加载带 rank 的词表
    print("Loading exam wordlists...")
    coca = load_vocab_dict(COCA_PATH)
    toefl = load_vocab_dict(TOEFL_PATH)
    gre = load_vocab_dict(GRE_PATH)
    coca_set = set(coca.keys())
    toefl_set = set(toefl.keys())
    gre_set = set(gre.keys())
    print(f"  COCA: {len(coca_set)}, TOEFL: {len(toefl_set)}, GRE: {len(gre_set)}")

    # 缺失词
    missing_coca = coca_set - existing
    missing_toefl = toefl_set - existing
    missing_gre = gre_set - existing
    print(f"  Missing COCA: {len(missing_coca)}, TOEFL: {len(missing_toefl)}, GRE: {len(missing_gre)}")

    # 过滤 COCA 噪声
    coca_candidates = {w for w in missing_coca if not is_noise(w)}
    # 从 COCA 中移除疑似专有名词
    coca_pn = {w for w in coca_candidates if has_uppercase(w)}
    coca_candidates -= coca_pn
    print(f"\nCOCA candidates after filter: {len(coca_candidates)} (removed noise: {len(missing_coca)-len(coca_candidates)-len(coca_pn)}, proper: {len(coca_pn)})")

    # 过滤 TOEFL 噪声，大部分应保留
    toefl_candidates = {w for w in missing_toefl if not is_noise(w)}
    print(f"TOEFL candidates: {len(toefl_candidates)}")

    # GRE：只保留出现在 COCA、TOEFL 或现代高价值词中的词
    gre_candidates = set()
    for w in missing_gre:
        if is_noise(w):
            continue
        if w in coca_set or w in toefl_set or w in MODERN_HIGH_VALUE:
            gre_candidates.add(w)
    print(f"GRE candidates: {len(gre_candidates)} (filtered: {len(missing_gre)-len(gre_candidates)})")

    # 尚未覆盖的现代高价值词
    modern_add = MODERN_HIGH_VALUE - existing - coca_candidates - toefl_candidates - gre_candidates
    print(f"Modern high-value not yet covered: {len(modern_add)}")

    # 合并
    all_candidates = coca_candidates | toefl_candidates | gre_candidates | modern_add
    print(f"Total candidates: {len(all_candidates)}")

    # 构建新条目
    new_entries = {}
    src_stats = {}

    for word in sorted(all_candidates):
        rank = coca.get(word, 20000)
        norm_rank = min(1.0, max(0.0, (rank - 1) / 30000.0))

        in_coca = word in coca_set
        in_toefl = word in toefl_set
        in_gre = word in gre_set
        in_modern = word in MODERN_HIGH_VALUE

        # 确定 priority
        if in_toefl:
            priority = 9.5
        elif in_gre:
            priority = 10.0
        elif in_modern:
            priority = 9.0
        else:
            priority = estimate_priority_from_rank(rank)

        # 确定 sources 标签
        src_parts = []
        if in_toefl: src_parts.append("toefl")
        if in_gre: src_parts.append("gre")
        if in_coca: src_parts.append("coca20000")
        if in_modern: src_parts.append("modern_domain")
        src_label = "+".join(src_parts) if src_parts else "coca20000_candidate"

        src_stats[src_label] = src_stats.get(src_label, 0) + 1

        pri_int = max(1, min(11, int(round(priority))))
        first_stage = STAGE_NAMES[pri_int]
        stages = [first_stage]
        for s in ["cet4","cet6","ielts"]:
            p = STAGE_PRIORITY[s]
            if p > pri_int and s not in stages:
                stages.append(s)

        norm_priority = (priority - 1) / 10.0
        difficulty = compute_difficulty(norm_priority, norm_rank)

        # 为新词应用 difficulty 提升（它们应比核心词汇更难）
        boost = 0.0
        if in_toefl or in_gre or in_modern:
            boost = 0.12  # TOEFL/GRE/现代词确实更难
        elif in_coca and rank < 10000:
            boost = 0.06  # 较低频 COCA 词仍属中等难度
        elif in_coca:
            boost = 0.10  # 更罕见的 COCA 词
        difficulty = min(0.99, round(difficulty + boost, 4))

        new_entries[word] = {
            "first_stage": first_stage,
            "all_stages": stages,
            "sources": [src_label],
            "source_confidence": "medium",
            "difficulty": difficulty,
        }

    # 构建 enhanced 数据
    enhanced = {
        "meta": {**base["meta"]},
        "stages": {**base["stages"]},
        "word_to_stage": {**base["word_to_stage"]},
        "overlap_matrix": dict(base["overlap_matrix"]),
    }
    enhanced["meta"]["generated_at"] = "2026-06-25T10:30:00+08:00"
    enhanced["meta"]["vocab_expansion"] = {
        "added_words": len(new_entries),
        "total_words_after": len(existing) + len(new_entries),
        "source_breakdown": dict(sorted(src_stats.items(), key=lambda x: -x[1])),
    }

    for word, info in new_entries.items():
        enhanced["word_to_stage"][word] = info

    # 重新计算 clusters
    all_diffs = sorted([float(v["difficulty"]) for v in enhanced["word_to_stage"].values() if v.get("difficulty") is not None])
    n_words = len(all_diffs)

    def cluster_of(d, n_cls):
        idx = bisect.bisect_left(all_diffs, d)
        pct = idx / n_words if n_words else 0
        return min(n_cls - 1, int(pct * n_cls))

    for v in enhanced["word_to_stage"].values():
        if v.get("difficulty") is not None:
            v["cluster_20"] = cluster_of(float(v["difficulty"]), 20)
            v["cluster_100"] = cluster_of(float(v["difficulty"]), 100)

    # 保存
    OUTPUT_PATH.write_text(json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8")
    total_after = len(enhanced["word_to_stage"])
    print(f"\nSaved enhanced vocab: {total_after} words (+{total_after-len(existing)})")

    # === 运行文章估算对比 ===
    print("\nRunning article estimation comparison...")
    sys.path.insert(0, str(PROJECT_ROOT))
    from vocab_estimator.article_estimator import estimate_article as est_art

    test_files = {
        "C": Path("/tmp/extra_materials/语料/C.txt"),
        "F": Path("/tmp/extra_materials/语料/F.txt"),
        "P": Path("/tmp/extra_materials/语料/P.txt"),
        "K": Path("/tmp/extra_materials/语料/K.txt"),
    }

    results = {}
    for label, path in test_files.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        rb = est_art(text, stage_vocab_path=str(STAGE_PATH))
        re_ = est_art(text, stage_vocab_path=str(OUTPUT_PATH))
        results[label] = {"base": rb, "enh": re_}

    # === 写入报告 ===
    report_lines = []
    report_lines.append("# 词库扩展结果报告\n")
    report_lines.append("## 概览\n")
    report_lines.append(f"| 指标 | 扩展前 | 扩展后 | 增量 |")
    report_lines.append(f"|------|:-----:|:-----:|:----:|")
    report_lines.append(f"| 词库大小 | {len(existing)} | {total_after} | +{total_after-len(existing)} |")
    report_lines.append(f"| 新词条数 | - | {len(new_entries)} | - |\n")

    report_lines.append("### 新增词来源分布\n")
    report_lines.append("| 来源 | 数量 |")
    report_lines.append("|------|:----:|")
    for src, cnt in sorted(src_stats.items(), key=lambda x: -x[1]):
        report_lines.append(f"| {src} | {cnt} |")
    report_lines.append("")

    report_lines.append("### 文章估算对比\n")
    report_lines.append("| 文档 | Base词汇量 | Enh词汇量 | Base覆盖率 | Enh覆盖率 | 变化 |")
    report_lines.append("|:---:|:---------:|:---------:|:---------:|:---------:|:----:|")
    for label in ["C","F","P","K"]:
        if label not in results:
            continue
        r = results[label]
        b = r["base"]
        e = r["enh"]
        bv = b.get("estimated_vocab","err")
        ev = e.get("estimated_vocab","err")
        bc = f"{b.get('coverage',{}).get('stage_vocab',0)*100:.1f}%" if isinstance(b.get("coverage"),dict) else "err"
        ec = f"{e.get('coverage',{}).get('stage_vocab',0)*100:.1f}%" if isinstance(e.get("coverage"),dict) else "err"
        ch = f"+{ev-bv}" if isinstance(bv,int) and isinstance(ev,int) else "-"
        report_lines.append(f"| {label} | {bv} | {ev} | {bc} | {ec} | {ch} |")

    report_lines.append("\n### 详情\n")
    for label in ["C","F","P","K"]:
        if label not in results:
            continue
        e = results[label]["enh"]
        report_lines.append(f"#### {label}.txt\n")
        report_lines.append(f"- 估算词汇量: {e.get('estimated_vocab')}")
        report_lines.append(f"- 教育阶段: {e.get('level')}")
        report_lines.append(f"- difficulty 中位数: {e.get('difficulty_median')}")
        report_lines.append(f"- Token覆盖率: {e.get('coverage',{}).get('stage_vocab',0)*100:.1f}%")
        report_lines.append(f"- Unique覆盖率: {e.get('article_stats',{}).get('coverage_unique',0)*100:.1f}%")
        unmatched = e.get("article_stats",{}).get("unmatched_unique_words",[])
        if unmatched:
            report_lines.append(f"- 未匹配词（前20）: {', '.join(unmatched[:20])}")
        report_lines.append("")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport: {REPORT_PATH}")

    print("\n=== Comparison ===")
    for label in ["C","F","P","K"]:
        if label not in results:
            continue
        r = results[label]
        b, e = r["base"], r["enh"]
        bv = b.get("estimated_vocab","?")
        ev = e.get("estimated_vocab","?")
        bcv = b.get("coverage",{}).get("stage_vocab",0) if isinstance(b.get("coverage"),dict) else 0
        ecv = e.get("coverage",{}).get("stage_vocab",0) if isinstance(e.get("coverage"),dict) else 0
        print(f"  {label}: {bv} -> {ev}  (cov: {bcv*100:.1f}% -> {ecv*100:.1f}%)")
        um = e.get("article_stats",{}).get("unmatched_unique_words",[])
        if um:
            print(f"    top unmatched: {', '.join(um[:10])}")

if __name__ == "__main__":
    main()
