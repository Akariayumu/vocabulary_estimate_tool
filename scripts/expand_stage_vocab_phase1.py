#!/usr/bin/env python3
"""Phase 1 stage-vocabulary expansion for article estimation.

Generates:
- data/stage_vocab_enhanced.json
- outputs/vocab_expansion_report.md

The script is intentionally deterministic and uses only local word lists.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vocab_estimator.article_estimator import STOPWORDS, estimate_article

DEFAULT_STAGE_VOCAB = PROJECT_ROOT / "data" / "stage_vocab.json"
DEFAULT_ENHANCED_VOCAB = PROJECT_ROOT / "data" / "stage_vocab_enhanced.json"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "vocab_expansion_report.md"
EXAM_DIR = PROJECT_ROOT / "data" / "exam_vocab"
EXTRA_DOC_DIR = Path("/tmp/extra_materials/语料")
EXTRA_DOCS = {
    "C": EXTRA_DOC_DIR / "C.txt",
    "F": EXTRA_DOC_DIR / "F.txt",
    "P": EXTRA_DOC_DIR / "P.txt",
    "K": EXTRA_DOC_DIR / "K.txt",
}
FALLBACK_DOCS = {
    "C": PROJECT_ROOT / "examples" / "doc_c.txt",
    "F": PROJECT_ROOT / "examples" / "doc_f.txt",
    "P": PROJECT_ROOT / "examples" / "doc_p.txt",
    "K": PROJECT_ROOT / "examples" / "doc_k.txt",
}
ARTICLE_DOCS = EXTRA_DOCS if all(path.exists() for path in EXTRA_DOCS.values()) else FALLBACK_DOCS

TOKEN_RE = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)*$")
LOWER_TOKEN_RE = re.compile(r"^[a-z]+(?:-[a-z]+)*$")
VOCAB_SIZE = 30_000
ALPHA = 0.60
BETA = 0.40

STAGE_PRIORITY = {
    "primary_3": 1,
    "primary_4": 2,
    "primary_5": 3,
    "primary_6": 4,
    "junior_7": 5,
    "junior_8": 6,
    "junior_9": 7,
    "senior": 8,
    "cet4": 9,
    "cet6": 10,
    "ielts": 11,
}

PRIORITY_STAGE = [
    (8.49, "senior"),
    (9.49, "cet4"),
    (10.49, "cet6"),
    (11.00, "ielts"),
]

ABSTRACT_SUFFIXES = (
    "ability",
    "ibility",
    "acy",
    "ance",
    "ancy",
    "ence",
    "ency",
    "hood",
    "ism",
    "ist",
    "ity",
    "logy",
    "ment",
    "ness",
    "ology",
    "ship",
    "sion",
    "tion",
)

DERIVATIONAL_SUFFIXES = (
    "able",
    "al",
    "ance",
    "ant",
    "ary",
    "ation",
    "ence",
    "ent",
    "er",
    "ial",
    "ible",
    "ic",
    "ical",
    "ism",
    "ist",
    "ity",
    "ive",
    "ization",
    "ize",
    "less",
    "logy",
    "ly",
    "ment",
    "ness",
    "ology",
    "ous",
    "ship",
    "sion",
    "tion",
)

DOMAIN_MARKERS = {
    "algorithm",
    "algorithmic",
    "anthropocene",
    "automation",
    "biodiversity",
    "biofuel",
    "biotech",
    "blockchain",
    "carbon",
    "climate",
    "cloud",
    "cryptocurrency",
    "cyber",
    "decarbonization",
    "decentralized",
    "digital",
    "emissions",
    "geopolitical",
    "governance",
    "internet",
    "machine-learning",
    "mitigation",
    "neoliberal",
    "platform",
    "renewable",
    "sustainability",
}

MANUAL_DOMAIN_WORDS = {
    "ai": 9.6,
    "ai-driven": 10.0,
    "algorithmic": 10.2,
    "algorithmically": 10.4,
    "anthropocene": 10.8,
    "biodiversity": 10.3,
    "blockchain": 9.9,
    "carbon-neutral": 10.1,
    "cloud-native": 10.0,
    "cryptocurrency": 10.2,
    "decarbonization": 10.5,
    "decarbonize": 10.4,
    "deepfake": 10.1,
    "disinformation": 10.3,
    "epistemological": 10.8,
    "epistemology": 10.8,
    "geopolitical": 10.4,
    "governance": 10.1,
    "hegemony": 10.8,
    "machine-learning": 10.0,
    "misinformation": 10.1,
    "mitigation": 10.2,
    "neoliberal": 10.6,
    "societal": 10.0,
}

SOURCE_ORDER = [
    "vocab_expansion_phase1",
    "manual_domain",
    "coca20000",
    "exam_vocab:gaokao",
    "exam_vocab:cet6",
    "exam_vocab:toefl",
    "exam_vocab:gre",
    "derived",
]


def read_word_list(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    words: list[str] = []
    variants: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().split()[0] if line.strip() else ""
        if not token:
            continue
        words.append(token)
        variants[token.lower()].append(token)
    return words, variants


def load_exam_vocab() -> dict[str, Any]:
    raw: dict[str, list[str]] = {}
    variants: dict[str, dict[str, list[str]]] = {}
    sets: dict[str, set[str]] = {}
    for name in ("gaokao", "cet6", "toefl", "gre", "coca20000"):
        words, word_variants = read_word_list(EXAM_DIR / f"{name}.txt")
        raw[name] = words
        variants[name] = word_variants
        sets[name] = {normalize_word(w) for w in words if normalize_word(w)}

    coca_rank: dict[str, int] = {}
    for index, token in enumerate(raw["coca20000"], start=1):
        word = normalize_word(token)
        if word and word not in coca_rank:
            coca_rank[word] = index

    return {"raw": raw, "variants": variants, "sets": sets, "coca_rank": coca_rank}


def normalize_word(token: str) -> str | None:
    token = token.strip()
    if not TOKEN_RE.fullmatch(token):
        return None
    return token.lower()


def is_probable_proper(word: str, source_variants: dict[str, dict[str, list[str]]]) -> bool:
    variants = []
    for vocab_variants in source_variants.values():
        variants.extend(vocab_variants.get(word, []))
    if not variants:
        return False

    lower_seen = any(v == word for v in variants)
    if lower_seen:
        return False

    content_variants = [v for v in variants if v.lower() not in {"i"}]
    if not content_variants:
        return False
    return any(any(char.isupper() for char in variant) for variant in content_variants)


def is_noise(word: str) -> bool:
    if word in MANUAL_DOMAIN_WORDS:
        return False
    if not LOWER_TOKEN_RE.fullmatch(word):
        return True
    compact = word.replace("-", "")
    if len(compact) < 3:
        return True
    if word in STOPWORDS:
        return True
    parts = word.split("-")
    if len(parts) > 3:
        return True
    if any(len(part) < 2 for part in parts):
        return True
    return False


def root_candidates(word: str) -> list[str]:
    roots: list[str] = []

    def add(candidate: str) -> None:
        if len(candidate) >= 3 and candidate != word and candidate not in roots:
            roots.append(candidate)

    if "-" in word:
        for part in word.split("-"):
            add(part)
        return roots

    if word.endswith("ically") and len(word) > 8:
        add(word[:-2])
        add(word[:-6] + "ic")
    if word.endswith("ological") and len(word) > 10:
        add(word[:-4] + "y")
    if word.endswith("ical") and len(word) > 6:
        add(word[:-4] + "y")
        add(word[:-2])
        add(word[:-4])
    if word.endswith("ial") and len(word) > 5:
        add(word[:-3] + "y")
        add(word[:-2])
    if word.endswith("ic") and len(word) > 5:
        add(word[:-2])
    if word.endswith("al") and len(word) > 5:
        add(word[:-2])
    if word.endswith("ization") and len(word) > 9:
        add(word[:-7] + "ize")
        add(word[:-7])
    if word.endswith("ation") and len(word) > 7:
        add(word[:-5] + "e")
        add(word[:-3])
    if word.endswith("tion") and len(word) > 6:
        add(word[:-3] + "e")
        add(word[:-4])
    if word.endswith("sion") and len(word) > 6:
        add(word[:-3] + "e")
        add(word[:-4])
    if word.endswith(("ance", "ence")) and len(word) > 7:
        add(word[:-4])
        add(word[:-4] + "e")
    if word.endswith("ity") and len(word) > 6:
        add(word[:-3])
        add(word[:-3] + "e")
        add(word[:-3] + "al")
    if word.endswith("ness") and len(word) > 7:
        add(word[:-4])
    if word.endswith("ment") and len(word) > 7:
        add(word[:-4])
    if word.endswith("ive") and len(word) > 6:
        add(word[:-3])
        add(word[:-3] + "e")
    if word.endswith(("ous", "able", "ible")) and len(word) > 7:
        add(word[:-3])
        add(word[:-4])
    if word.endswith(("ize", "ise")) and len(word) > 6:
        add(word[:-3])
    if word.endswith("ly") and len(word) > 5:
        add(word[:-2])

    return roots


def has_known_root(word: str, base_words: set[str]) -> bool:
    return any(root in base_words for root in root_candidates(word))


def classify_word(
    word: str,
    *,
    source_variants: dict[str, dict[str, list[str]]],
    base_words: set[str],
) -> str:
    if is_noise(word):
        return "noise"
    if word not in MANUAL_DOMAIN_WORDS and is_probable_proper(word, source_variants):
        return "proper_noun"
    if "-" in word:
        return "compound_hyphen"
    if word in MANUAL_DOMAIN_WORDS or is_domain_word(word):
        return "modern_domain"
    if has_known_root(word, base_words):
        return "transparent_derived"
    if word.endswith(ABSTRACT_SUFFIXES):
        return "academic_abstract"
    return "general_content"


def is_domain_word(word: str) -> bool:
    if word in DOMAIN_MARKERS:
        return True
    return any(marker in word for marker in ("cyber", "crypto", "climate", "algorithm"))


def should_include_word(
    word: str,
    category: str,
    sources: set[str],
    *,
    base_words: set[str],
) -> bool:
    if word in MANUAL_DOMAIN_WORDS:
        return True
    if category in {"noise", "proper_noun"}:
        return False
    if "coca20000" in sources:
        return True
    if "toefl" in sources:
        return True
    if "gre" in sources:
        return (
            "toefl" in sources
            or category in {"academic_abstract", "transparent_derived", "modern_domain"}
            or has_known_root(word, base_words)
        )
    if category == "transparent_derived" and sources.intersection({"gaokao", "cet6", "toefl", "gre"}):
        return True
    return False


def coca_priority(rank: int) -> float:
    if rank < 5_000:
        return 8.0 + rank / 5_000
    if rank < 15_000:
        return 9.0 + (rank - 5_000) / 10_000
    return min(11.0, 10.0 + (rank - 15_000) / 15_000)


def norm_stage(priority: float) -> float:
    return (priority - 1.0) / 10.0


def norm_rank(rank: int) -> float:
    return math.log(rank + 1) / math.log(VOCAB_SIZE + 1)


def fallback_rank(priority: float) -> int:
    if priority <= 8.5:
        return 4_000
    if priority <= 9.5:
        return 8_000
    if priority <= 10.5:
        return 16_000
    return 24_000


def priority_to_stage(priority: float) -> str:
    for upper, stage in PRIORITY_STAGE:
        if priority <= upper:
            return stage
    return "ielts"


def compute_new_difficulty(
    word: str,
    sources: set[str],
    coca_rank_map: dict[str, int],
    base_entries: dict[str, dict[str, Any]],
) -> tuple[float, float, int]:
    priorities: list[float] = []
    if word in MANUAL_DOMAIN_WORDS:
        priorities.append(MANUAL_DOMAIN_WORDS[word])
    if "gaokao" in sources:
        priorities.append(8.0)
    if "cet6" in sources:
        priorities.append(10.0)
    if "toefl" in sources:
        priorities.append(10.5)
    if "gre" in sources:
        priorities.append(10.8)

    rank = coca_rank_map.get(word)
    if rank is not None:
        priorities.append(coca_priority(rank))

    priority = max(priorities) if priorities else 10.0
    rank_for_score = rank if rank is not None else fallback_rank(priority)
    score = ALPHA * norm_stage(priority) + BETA * norm_rank(rank_for_score)

    root_difficulties = [
        float(base_entries[root]["difficulty"])
        for root in root_candidates(word)
        if root in base_entries and base_entries[root].get("difficulty") is not None
    ]
    if root_difficulties:
        anchor = max(root_difficulties) + 0.05
        if score > anchor + 0.10:
            score = 0.70 * score + 0.30 * min(0.99, anchor)
        elif score < anchor:
            score = min(0.99, 0.60 * score + 0.40 * anchor)

    return round(max(0.0, min(0.995, score)), 4), priority, rank_for_score


def source_labels(sources: set[str], category: str) -> list[str]:
    labels = {"vocab_expansion_phase1"}
    for source in sources:
        if source == "coca20000":
            labels.add("coca20000")
        elif source in {"gaokao", "cet6", "toefl", "gre"}:
            labels.add(f"exam_vocab:{source}")
    if category == "transparent_derived":
        labels.add("derived")
    if category == "modern_domain" or sources.intersection({"manual_domain"}):
        labels.add("manual_domain")
    return sorted(labels, key=lambda label: SOURCE_ORDER.index(label) if label in SOURCE_ORDER else 99)


def source_confidence(sources: set[str], category: str) -> str:
    if "coca20000" in sources and sources.intersection({"toefl", "gre", "cet6", "gaokao"}):
        return "high"
    if category in {"modern_domain", "transparent_derived"} and (
        "coca20000" in sources or sources.intersection({"toefl", "gre", "cet6", "gaokao"})
    ):
        return "high"
    if "coca20000" in sources or "toefl" in sources:
        return "medium"
    return "low"


def build_candidates(
    stage_vocab: dict[str, Any],
    exam_vocab: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Counter], dict[str, list[str]]]:
    base_entries: dict[str, dict[str, Any]] = stage_vocab["word_to_stage"]
    base_words = set(base_entries)
    sets = exam_vocab["sets"]
    variants = exam_vocab["variants"]
    coca_rank_map = exam_vocab["coca_rank"]

    candidate_sources: dict[str, set[str]] = defaultdict(set)
    for source, words in sets.items():
        for word in words:
            if word and word not in base_words:
                candidate_sources[word].add(source)
    for word in MANUAL_DOMAIN_WORDS:
        if word not in base_words:
            candidate_sources[word].add("manual_domain")

    new_entries: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, Counter] = {"missing_coca": Counter(), "included": Counter(), "excluded": Counter()}
    samples: dict[str, list[str]] = defaultdict(list)

    for word in sorted(candidate_sources):
        sources = candidate_sources[word]
        category = classify_word(word, source_variants=variants, base_words=base_words)
        if "coca20000" in sources:
            category_counts["missing_coca"][category] += 1
            if len(samples[f"missing_coca:{category}"]) < 20:
                samples[f"missing_coca:{category}"].append(word)

        if should_include_word(word, category, sources, base_words=base_words):
            difficulty, priority, rank_for_score = compute_new_difficulty(
                word, sources, coca_rank_map, base_entries
            )
            first_stage = priority_to_stage(priority)
            new_entries[word] = {
                "first_stage": first_stage,
                "all_stages": [first_stage],
                "sources": source_labels(sources, category),
                "source_confidence": source_confidence(sources, category),
                "difficulty": difficulty,
                "cluster_20": None,
                "cluster_100": None,
                "translation": "",
                "expansion": {
                    "phase": "article_estimation_optimization_phase1",
                    "category": category,
                    "priority": round(priority, 3),
                    "coca_rank": coca_rank_map.get(word),
                    "rank_for_score": rank_for_score,
                    "root_candidates": [root for root in root_candidates(word) if root in base_words],
                },
            }
            category_counts["included"][category] += 1
            if len(samples[f"included:{category}"]) < 20:
                samples[f"included:{category}"].append(word)
        else:
            category_counts["excluded"][category] += 1
            if len(samples[f"excluded:{category}"]) < 20:
                samples[f"excluded:{category}"].append(word)

    return new_entries, category_counts, samples


def rebuild_stages_and_clusters(stage_vocab: dict[str, Any]) -> None:
    word_to_stage = stage_vocab["word_to_stage"]
    for stage_info in stage_vocab["stages"].values():
        stage_info["words"] = []

    for word, info in word_to_stage.items():
        all_stages = info.get("all_stages") or [info.get("first_stage")]
        clean_stages = [stage for stage in all_stages if stage in stage_vocab["stages"]]
        if not clean_stages and info.get("first_stage") in stage_vocab["stages"]:
            clean_stages = [info["first_stage"]]
        info["all_stages"] = clean_stages
        if info.get("first_stage") not in stage_vocab["stages"] and clean_stages:
            info["first_stage"] = clean_stages[0]
        for stage in clean_stages:
            stage_vocab["stages"][stage]["words"].append(word)

    for stage_info in stage_vocab["stages"].values():
        stage_info["words"] = sorted(set(stage_info["words"]))
        stage_info["count"] = len(stage_info["words"])

    sorted_words = sorted(
        (
            float(info["difficulty"]),
            word,
        )
        for word, info in word_to_stage.items()
        if info.get("difficulty") is not None
    )
    total = len(sorted_words)
    for index, (_, word) in enumerate(sorted_words):
        word_to_stage[word]["cluster_20"] = min(19, int(index * 20 / total))
        word_to_stage[word]["cluster_100"] = min(99, int(index * 100 / total))

    stage_sets = {
        stage: set(info["words"])
        for stage, info in stage_vocab["stages"].items()
    }
    overlap_matrix: dict[str, dict[str, int]] = {}
    for stage_a, words_a in stage_sets.items():
        overlap_matrix[stage_a] = {}
        for stage_b, words_b in stage_sets.items():
            if stage_a != stage_b:
                overlap_matrix[stage_a][stage_b] = len(words_a & words_b)
    stage_vocab["overlap_matrix"] = overlap_matrix
    stage_vocab["word_to_stage"] = dict(sorted(word_to_stage.items()))


def coverage_for_word_list(words: set[str], entries: set[str]) -> dict[str, Any]:
    normalized = {word for word in words if word}
    present = normalized & entries
    missing = normalized - entries
    return {
        "unique": len(normalized),
        "present": len(present),
        "missing": len(missing),
        "coverage": len(present) / len(normalized) if normalized else 0.0,
    }


def article_metrics(path: Path, vocab_path: Path) -> dict[str, Any]:
    result = estimate_article(path.read_text(encoding="utf-8"), vocab_path)
    dist = result["coverage"]["difficulty_distribution"]
    stats = result["article_stats"]
    return {
        "estimated_vocab": result["estimated_vocab"],
        "level": result["level"],
        "coverage_token": result["coverage"]["stage_vocab"],
        "coverage_unique": stats["coverage_unique"],
        "matched_unique_words": stats["matched_unique_words"],
        "unique_content_words": stats["unique_content_words"],
        "difficulty_median": result["difficulty_median"],
        "difficulty_p75": dist["p75"],
        "difficulty_max": dist["max"],
        "unmatched_unique_words": stats["unmatched_unique_words"],
    }


def summarize_difficulties(entries: dict[str, dict[str, Any]]) -> dict[str, float]:
    values = sorted(float(info["difficulty"]) for info in entries.values())
    return {
        "min": values[0],
        "p25": quantile(values, 0.25),
        "median": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "max": values[-1],
        "mean": mean(values),
    }


def quantile(values: list[float], q: float) -> float:
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def write_report(
    path: Path,
    *,
    base_vocab: dict[str, Any],
    enhanced_vocab: dict[str, Any],
    new_entries: dict[str, dict[str, Any]],
    category_counts: dict[str, Counter],
    samples: dict[str, list[str]],
    exam_vocab: dict[str, Any],
    before_articles: dict[str, dict[str, Any]],
    after_articles: dict[str, dict[str, Any]],
) -> None:
    base_entries = base_vocab["word_to_stage"]
    enhanced_entries = enhanced_vocab["word_to_stage"]
    base_words = set(base_entries)
    enhanced_words = set(enhanced_entries)

    coverage_rows = []
    for name in ("gaokao", "cet6", "toefl", "gre", "coca20000"):
        before = coverage_for_word_list(exam_vocab["sets"][name], base_words)
        after = coverage_for_word_list(exam_vocab["sets"][name], enhanced_words)
        coverage_rows.append(
            [
                name,
                before["unique"],
                before["present"],
                before["missing"],
                f"{before['coverage']:.2%}",
                after["present"],
                after["missing"],
                f"{after['coverage']:.2%}",
            ]
        )

    category_rows = []
    all_categories = sorted(
        set(category_counts["missing_coca"])
        | set(category_counts["included"])
        | set(category_counts["excluded"])
    )
    for category in all_categories:
        category_rows.append(
            [
                category,
                category_counts["missing_coca"][category],
                category_counts["included"][category],
                category_counts["excluded"][category],
                ", ".join(samples.get(f"included:{category}", [])[:8]),
            ]
        )

    article_rows = []
    for label in ("C", "F", "P", "K"):
        before = before_articles[label]
        after = after_articles[label]
        article_rows.append(
            [
                label,
                f"{before['coverage_token']:.2%}",
                f"{after['coverage_token']:.2%}",
                f"{before['coverage_unique']:.2%}",
                f"{after['coverage_unique']:.2%}",
                before["estimated_vocab"],
                after["estimated_vocab"],
                before["difficulty_median"],
                after["difficulty_median"],
                after["difficulty_p75"],
                after["difficulty_max"],
            ]
        )

    source_counter = Counter()
    stage_counter = Counter()
    for info in new_entries.values():
        stage_counter[info["first_stage"]] += 1
        for source in info["sources"]:
            source_counter[source] += 1

    diff_summary = summarize_difficulties(new_entries) if new_entries else {}
    key_words = [
        "algorithmic",
        "societal",
        "geopolitical",
        "blockchain",
        "hegemony",
        "epistemology",
        "governance",
        "mitigation",
        "biodiversity",
        "neoliberal",
    ]
    key_rows = []
    for word in key_words:
        info = enhanced_entries.get(word)
        if info:
            expansion = info.get("expansion", {})
            key_rows.append(
                [
                    word,
                    "yes",
                    info["difficulty"],
                    info["cluster_20"],
                    info["cluster_100"],
                    info["first_stage"],
                    ", ".join(info["sources"]),
                    expansion.get("category", ""),
                ]
            )
        else:
            key_rows.append([word, "no", "", "", "", "", "", ""])

    content = [
        "# 词库扩展 Phase 1 报告",
        "",
        f"- 生成时间：{datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')}",
        f"- 原始词库：`{len(base_entries):,}` 个词条",
        f"- 增强词库：`{len(enhanced_entries):,}` 个词条",
        f"- 新增词条：`{len(new_entries):,}` 个",
        f"- 输出文件：`data/stage_vocab_enhanced.json`",
        f"- 测试语料：`{ARTICLE_DOCS['C'].parent}`",
        "",
        "## 外部词表覆盖率",
        "",
        markdown_table(
            ["词表", "unique", "扩展前命中", "扩展前缺失", "扩展前覆盖", "扩展后命中", "扩展后缺失", "扩展后覆盖"],
            coverage_rows,
        ),
        "",
        "## COCA 缺失词分类与纳入",
        "",
        markdown_table(
            ["类别", "COCA缺失数", "纳入数", "排除数", "纳入样例"],
            category_rows,
        ),
        "",
        "## 新增词来源与阶段",
        "",
        markdown_table(["来源", "词条数"], [[k, v] for k, v in sorted(source_counter.items())]),
        "",
        markdown_table(["first_stage", "新增词条数"], [[k, stage_counter[k]] for k in STAGE_PRIORITY if stage_counter[k]]),
        "",
        "## 新增词 difficulty 分布",
        "",
        markdown_table(
            ["min", "p25", "median", "p75", "max", "mean"],
            [[f"{diff_summary[k]:.4f}" for k in ("min", "p25", "median", "p75", "max", "mean")]],
        ),
        "",
        "## 重点缺失词验证",
        "",
        markdown_table(
            ["词", "已加入", "difficulty", "cluster_20", "cluster_100", "first_stage", "sources", "category"],
            key_rows,
        ),
        "",
        "## 四篇测试语料对比",
        "",
        markdown_table(
            [
                "文件",
                "token覆盖前",
                "token覆盖后",
                "unique覆盖前",
                "unique覆盖后",
                "estimated前",
                "estimated后",
                "median前",
                "median后",
                "p75后",
                "max后",
            ],
            article_rows,
        ),
        "",
        "## 增强后未匹配词样例",
        "",
    ]

    for label in ("C", "F", "P", "K"):
        words = after_articles[label]["unmatched_unique_words"]
        content.append(f"- {label}: {', '.join(words) if words else '(none)'}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def enhance_vocab(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> None:
    base_vocab = json.loads(input_path.read_text(encoding="utf-8"))
    exam_vocab = load_exam_vocab()
    new_entries, category_counts, samples = build_candidates(base_vocab, exam_vocab)

    enhanced_vocab = copy.deepcopy(base_vocab)
    enhanced_vocab["word_to_stage"].update(new_entries)
    meta = enhanced_vocab.setdefault("meta", {})
    meta["enhanced_from"] = str(input_path.relative_to(PROJECT_ROOT))
    meta["enhanced_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    meta["enhancement"] = {
        "name": "article_estimation_optimization_phase1",
        "new_entries": len(new_entries),
        "rules": [
            "include COCA missing words after proper-noun/noise filtering",
            "include TOEFL missing words after proper-noun/noise filtering",
            "include GRE missing words only when COCA/TOEFL-backed, transparent-derived, academic-abstract, or modern-domain",
            "include manual AI/climate/business/technology domain whitelist",
        ],
    }
    meta.setdefault("sources", {})["vocab_expansion_phase1"] = {
        "files": [
            "data/exam_vocab/coca20000.txt",
            "data/exam_vocab/toefl.txt",
            "data/exam_vocab/gre.txt",
            "data/exam_vocab/cet6.txt",
            "data/exam_vocab/gaokao.txt",
        ],
        "description": "Phase 1 article-estimation vocabulary expansion generated by scripts/expand_stage_vocab_phase1.py",
    }

    rebuild_stages_and_clusters(enhanced_vocab)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(enhanced_vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    before_articles = {
        label: article_metrics(path, input_path)
        for label, path in ARTICLE_DOCS.items()
    }
    after_articles = {
        label: article_metrics(path, output_path)
        for label, path in ARTICLE_DOCS.items()
    }
    write_report(
        report_path,
        base_vocab=base_vocab,
        enhanced_vocab=enhanced_vocab,
        new_entries=new_entries,
        category_counts=category_counts,
        samples=samples,
        exam_vocab=exam_vocab,
        before_articles=before_articles,
        after_articles=after_articles,
    )

    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)} with {len(enhanced_vocab['word_to_stage']):,} entries")
    print(f"Added {len(new_entries):,} entries")
    print(f"Wrote {report_path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_STAGE_VOCAB)
    parser.add_argument("--output", type=Path, default=DEFAULT_ENHANCED_VOCAB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    enhance_vocab(args.input.resolve(), args.output.resolve(), args.report.resolve())


if __name__ == "__main__":
    main()
