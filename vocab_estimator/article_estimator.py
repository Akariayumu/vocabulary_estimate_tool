"""仅基于 ``stage_vocab.json`` 的文章词汇量估算。"""

from __future__ import annotations

import bisect
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_VOCAB_PATH = PROJECT_ROOT / "data" / "stage_vocab_clean_v1.json"

WORD_RE = re.compile(r"[a-z]+(?:-[a-z]+)*")

STOPWORDS = {
    "a",
    "able",
    "about",
    "above",
    "according",
    "accordingly",
    "across",
    "actually",
    "after",
    "afterwards",
    "again",
    "against",
    "ago",
    "all",
    "allow",
    "allows",
    "almost",
    "alone",
    "along",
    "already",
    "also",
    "although",
    "always",
    "am",
    "among",
    "amongst",
    "an",
    "and",
    "another",
    "any",
    "anybody",
    "anyhow",
    "anyone",
    "anything",
    "anyway",
    "anyways",
    "anywhere",
    "apart",
    "appear",
    "appreciate",
    "appropriate",
    "are",
    "aren",
    "around",
    "as",
    "aside",
    "ask",
    "asking",
    "associated",
    "at",
    "available",
    "away",
    "awfully",
    "b",
    "be",
    "became",
    "because",
    "become",
    "becomes",
    "becoming",
    "been",
    "before",
    "beforehand",
    "behind",
    "being",
    "believe",
    "below",
    "beside",
    "besides",
    "between",
    "beyond",
    "both",
    "brief",
    "but",
    "by",
    "c",
    "came",
    "can",
    "cannot",
    "cant",
    "cause",
    "causes",
    "certain",
    "certainly",
    "changes",
    "clearly",
    "co",
    "com",
    "come",
    "comes",
    "concerning",
    "consequently",
    "consider",
    "considering",
    "contain",
    "containing",
    "contains",
    "corresponding",
    "could",
    "couldn",
    "course",
    "currently",
    "d",
    "definitely",
    "described",
    "despite",
    "did",
    "didn",
    "different",
    "do",
    "does",
    "doesn",
    "doing",
    "don",
    "done",
    "down",
    "downwards",
    "during",
    "e",
    "each",
    "edu",
    "eg",
    "eight",
    "either",
    "else",
    "elsewhere",
    "enough",
    "entirely",
    "especially",
    "et",
    "etc",
    "even",
    "ever",
    "every",
    "everybody",
    "everyone",
    "everything",
    "everywhere",
    "ex",
    "exactly",
    "example",
    "except",
    "f",
    "far",
    "few",
    "fifth",
    "first",
    "five",
    "followed",
    "following",
    "follows",
    "for",
    "former",
    "formerly",
    "forth",
    "four",
    "from",
    "further",
    "furthermore",
    "g",
    "get",
    "gets",
    "getting",
    "given",
    "gives",
    "go",
    "goes",
    "going",
    "gone",
    "got",
    "gotten",
    "great",
    "greetings",
    "h",
    "had",
    "hadn",
    "happens",
    "hardly",
    "has",
    "hasn",
    "have",
    "haven",
    "having",
    "he",
    "hello",
    "help",
    "hence",
    "her",
    "here",
    "hereafter",
    "hereby",
    "herein",
    "hereupon",
    "hers",
    "herself",
    "hi",
    "him",
    "himself",
    "his",
    "hither",
    "hopefully",
    "how",
    "howbeit",
    "however",
    "i",
    "ie",
    "if",
    "ignored",
    "immediate",
    "in",
    "inasmuch",
    "inc",
    "indeed",
    "indicate",
    "indicated",
    "indicates",
    "inner",
    "insofar",
    "instead",
    "into",
    "inward",
    "is",
    "isn",
    "it",
    "its",
    "itself",
    "j",
    "just",
    "k",
    "keep",
    "keeps",
    "kept",
    "know",
    "known",
    "knows",
    "l",
    "last",
    "lately",
    "later",
    "latter",
    "latterly",
    "least",
    "less",
    "lest",
    "let",
    "like",
    "liked",
    "likely",
    "little",
    "look",
    "looking",
    "looks",
    "long",
    "ltd",
    "m",
    "mainly",
    "many",
    "may",
    "maybe",
    "me",
    "mean",
    "meanwhile",
    "merely",
    "might",
    "more",
    "moreover",
    "most",
    "mostly",
    "much",
    "must",
    "mustn",
    "my",
    "myself",
    "n",
    "name",
    "namely",
    "nd",
    "near",
    "nearly",
    "necessary",
    "need",
    "needs",
    "neither",
    "never",
    "nevertheless",
    "new",
    "next",
    "nine",
    "no",
    "nobody",
    "non",
    "none",
    "noone",
    "nor",
    "normally",
    "not",
    "nothing",
    "novel",
    "now",
    "nowhere",
    "o",
    "obviously",
    "of",
    "off",
    "often",
    "oh",
    "ok",
    "okay",
    "old",
    "on",
    "once",
    "one",
    "ones",
    "only",
    "onto",
    "or",
    "other",
    "others",
    "otherwise",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "outside",
    "over",
    "overall",
    "own",
    "p",
    "particular",
    "particularly",
    "per",
    "perhaps",
    "placed",
    "please",
    "plus",
    "possible",
    "presumably",
    "probably",
    "provides",
    "put",
    "q",
    "que",
    "quite",
    "qv",
    "r",
    "rather",
    "rd",
    "re",
    "really",
    "reasonably",
    "regarding",
    "regardless",
    "regards",
    "relatively",
    "respectively",
    "right",
    "s",
    "said",
    "same",
    "saw",
    "say",
    "saying",
    "says",
    "second",
    "secondly",
    "see",
    "seeing",
    "seem",
    "seemed",
    "seeming",
    "seems",
    "seen",
    "self",
    "selves",
    "sensible",
    "sent",
    "serious",
    "seriously",
    "seven",
    "several",
    "shall",
    "she",
    "should",
    "shouldn",
    "since",
    "six",
    "so",
    "some",
    "somebody",
    "somehow",
    "someone",
    "something",
    "sometime",
    "sometimes",
    "somewhat",
    "somewhere",
    "soon",
    "sorry",
    "specified",
    "specify",
    "specifying",
    "still",
    "sub",
    "such",
    "sup",
    "sure",
    "t",
    "take",
    "taken",
    "tell",
    "tends",
    "th",
    "than",
    "thank",
    "thanks",
    "thanx",
    "that",
    "thats",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "thence",
    "there",
    "thereafter",
    "thereby",
    "therefore",
    "therein",
    "theres",
    "thereupon",
    "these",
    "they",
    "thing",
    "things",
    "think",
    "third",
    "this",
    "thorough",
    "thoroughly",
    "those",
    "though",
    "three",
    "through",
    "throughout",
    "thru",
    "thus",
    "to",
    "together",
    "too",
    "took",
    "toward",
    "towards",
    "tried",
    "tries",
    "truly",
    "try",
    "trying",
    "twice",
    "two",
    "u",
    "un",
    "under",
    "unfortunately",
    "unless",
    "unlikely",
    "until",
    "unto",
    "up",
    "upon",
    "us",
    "use",
    "used",
    "useful",
    "uses",
    "using",
    "usually",
    "v",
    "value",
    "various",
    "very",
    "via",
    "viz",
    "vs",
    "w",
    "want",
    "wants",
    "was",
    "wasn",
    "way",
    "we",
    "welcome",
    "well",
    "went",
    "were",
    "weren",
    "what",
    "whatever",
    "when",
    "whence",
    "whenever",
    "where",
    "whereafter",
    "whereas",
    "whereby",
    "wherein",
    "whereupon",
    "wherever",
    "whether",
    "which",
    "while",
    "whither",
    "who",
    "whoever",
    "whole",
    "whom",
    "whose",
    "why",
    "will",
    "willing",
    "wish",
    "with",
    "within",
    "without",
    "won",
    "wonder",
    "would",
    "wouldn",
    "x",
    "y",
    "yes",
    "yet",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "z",
    "zero",
}

IRREGULAR_LEMMAS = {
    "ate": "eat",
    "began": "begin",
    "begun": "begin",
    "bought": "buy",
    "brought": "bring",
    "built": "build",
    "came": "come",
    "caught": "catch",
    "children": "child",
    "did": "do",
    "done": "do",
    "drank": "drink",
    "driven": "drive",
    "drove": "drive",
    "eaten": "eat",
    "fallen": "fall",
    "fell": "fall",
    "felt": "feel",
    "fewer": "few",
    "found": "find",
    "gave": "give",
    "given": "give",
    "gone": "go",
    "grew": "grow",
    "grown": "grow",
    "had": "have",
    "heard": "hear",
    "held": "hold",
    "kept": "keep",
    "knew": "know",
    "known": "know",
    "left": "leave",
    "less": "little",
    "lost": "lose",
    "made": "make",
    "men": "man",
    "met": "meet",
    "paid": "pay",
    "ran": "run",
    "read": "read",
    "said": "say",
    "sang": "sing",
    "sat": "sit",
    "saw": "see",
    "seen": "see",
    "sent": "send",
    "slept": "sleep",
    "sold": "sell",
    "spent": "spend",
    "stood": "stand",
    "taken": "take",
    "taught": "teach",
    "teeth": "tooth",
    "thought": "think",
    "told": "tell",
    "took": "take",
    "went": "go",
    "women": "woman",
    "worse": "bad",
    "worst": "bad",
    "written": "write",
    "wrote": "write",
}

IRREGULAR_CONTENT_LEMMAS = {
    "better": "good",
    "best": "good",
}

VOWELS = set("aeiou")


def tokenize_article(
    text: str,
    entries: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """使用内置停用词表提取小写内容词 lemmas。"""

    tokens: list[str] = []
    for word in WORD_RE.findall(text.lower()):
        if word in STOPWORDS:
            continue
        lemma = lemmatize_word(word, entries)
        if lemma not in STOPWORDS:
            tokens.append(lemma)
    return tokens


def lemmatize_word(
    word: str,
    entries: dict[str, dict[str, Any]] | None = None,
) -> str:
    """返回轻量规则型 lemma，并优先匹配 stage-vocab 中的词。"""

    candidates = _lemma_candidates(word)
    if entries is not None:
        for candidate in candidates:
            if candidate in entries:
                return candidate
    return candidates[0]


def _lemma_candidates(word: str) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str) -> None:
        if len(candidate) >= 2 and candidate not in candidates:
            candidates.append(candidate)

    if word in IRREGULAR_CONTENT_LEMMAS:
        add(IRREGULAR_CONTENT_LEMMAS[word])
    if word in IRREGULAR_LEMMAS:
        add(IRREGULAR_LEMMAS[word])

    if word.endswith("ies") and len(word) > 4:
        add(word[:-3] + "y")
    if word.endswith("ves") and len(word) > 4:
        add(word[:-3] + "f")
        add(word[:-3] + "fe")
    if word.endswith(("ches", "shes", "sses", "xes", "zes", "oes")) and len(word) > 4:
        add(word[:-2])
    if word.endswith("es") and len(word) > 3:
        add(word[:-1])
        add(word[:-2])
    if word.endswith("s") and len(word) > 3 and not word.endswith(("ss", "us")):
        add(word[:-1])

    if word.endswith("ied") and len(word) > 4:
        add(word[:-3] + "y")
    if word.endswith("ed") and len(word) > 4:
        stem = word[:-2]
        add(stem)
        add(stem + "e")
        add(_undouble_final_consonant(stem))

    if word.endswith("ying") and len(word) > 5:
        add(word[:-4] + "ie")
    if word.endswith("ing") and len(word) > 5:
        stem = word[:-3]
        add(stem)
        add(stem + "e")
        add(_undouble_final_consonant(stem))

    if word.endswith("ically") and len(word) > 7:
        add(word[:-4])
        add(word[:-6] + "ic")
    if word.endswith("ily") and len(word) > 4:
        add(word[:-3] + "y")
    if word.endswith("ly") and len(word) > 4:
        add(word[:-2])

    if word.endswith("iest") and len(word) > 5:
        add(word[:-4] + "y")
    if word.endswith("ier") and len(word) > 4:
        add(word[:-3] + "y")
    if word.endswith("est") and len(word) > 5:
        stem = word[:-3]
        add(stem)
        add(stem + "e")
        add(_undouble_final_consonant(stem))
    if word.endswith("er") and len(word) > 4:
        stem = word[:-2]
        add(stem)
        add(stem + "e")
        add(_undouble_final_consonant(stem))

    if word.endswith(("tion", "sion", "ation", "ition")) and len(word) > 5:
        stem = word[:-3]
        add(stem)
        add(stem + "e")

    add(word)
    return candidates


def _undouble_final_consonant(stem: str) -> str:
    if (
        len(stem) >= 3
        and stem[-1] == stem[-2]
        and stem[-1] not in VOWELS
        and stem[-1] not in {"s", "z"}
    ):
        return stem[:-1]
    return stem


@lru_cache(maxsize=4)
def load_stage_vocab(stage_vocab_path: str | Path = DEFAULT_STAGE_VOCAB_PATH) -> dict[str, Any]:
    """从 ``stage_vocab.json`` 加载并归一化文章估算数据。"""

    path = Path(stage_vocab_path)
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)

    word_to_stage = raw.get("word_to_stage")
    if not isinstance(word_to_stage, dict):
        raise ValueError("stage_vocab.json must contain word_to_stage")

    entries: dict[str, dict[str, Any]] = {}
    difficulties: list[float] = []
    cluster_counts: Counter[int] = Counter()
    first_stage_counts: Counter[str] = Counter()

    for word, info in word_to_stage.items():
        if not isinstance(word, str) or not isinstance(info, dict):
            continue
        difficulty = info.get("difficulty")
        if difficulty is None:
            continue
        try:
            difficulty_float = float(difficulty)
        except (TypeError, ValueError):
            continue

        normalized_word = word.strip().lower()
        if not normalized_word:
            continue

        entry = {
            "difficulty": difficulty_float,
            "cluster_20": info.get("cluster_20"),
            "first_stage": info.get("first_stage"),
            "translation": info.get("translation"),
        }
        entries[normalized_word] = entry
        difficulties.append(difficulty_float)

        cluster_20 = entry["cluster_20"]
        if cluster_20 is not None:
            cluster_counts[int(cluster_20)] += 1

        first_stage = entry["first_stage"]
        if isinstance(first_stage, str):
            first_stage_counts[first_stage] += 1

    if not entries or not difficulties:
        raise ValueError("stage_vocab.json does not contain usable difficulty data")

    stages = raw.get("stages", {})
    stage_order = sorted(
        (
            int(stage_info.get("priority", 0)),
            stage_key,
            str(stage_info.get("label", stage_key)),
        )
        for stage_key, stage_info in stages.items()
        if isinstance(stage_info, dict)
    )

    cumulative_stage_sizes: list[dict[str, Any]] = []
    cumulative = 0
    for _, stage_key, label in stage_order:
        cumulative += first_stage_counts[stage_key]
        cumulative_stage_sizes.append(
            {"stage": stage_key, "label": label, "cumulative_size": cumulative}
        )

    return {
        "entries": entries,
        "sorted_difficulties": sorted(difficulties),
        "total_words": len(difficulties),
        "cluster_counts": dict(sorted(cluster_counts.items())),
        "stage_levels": cumulative_stage_sizes,
        "meta": raw.get("meta", {}),
    }


def estimate_article(
    article: str,
    stage_vocab_path: str | Path = DEFAULT_STAGE_VOCAB_PATH,
) -> dict[str, Any]:
    """估算阅读一篇文章所需的词汇量。

    估算值为文章 difficulty 中位数在阶段词库中的累计位置：
    ``P(vocab_difficulty <= article_median) * N``。
    """

    if not isinstance(article, str) or not article.strip():
        raise ValueError("article cannot be empty")

    vocab = load_stage_vocab(stage_vocab_path)
    entries: dict[str, dict[str, Any]] = vocab["entries"]

    tokens = tokenize_article(article, entries)
    if not tokens:
        raise ValueError("article does not contain analyzable English words")

    matched: list[tuple[str, dict[str, Any]]] = [
        (token, entries[token]) for token in tokens if token in entries
    ]
    if not matched:
        raise ValueError(
            "未能在词库中匹配到文章词汇。请确认输入是英文正文，且不是全数字、符号、"
            "代码片段或非英文文本。"
        )

    difficulties = sorted(item["difficulty"] for _, item in matched)
    difficulty_median = _quantile(difficulties, 0.50)
    p25 = _quantile(difficulties, 0.25)
    p75 = _quantile(difficulties, 0.75)

    sorted_vocab_difficulties: list[float] = vocab["sorted_difficulties"]
    cumulative_count = bisect.bisect_right(sorted_vocab_difficulties, difficulty_median)
    total_words = int(vocab["total_words"])
    estimated_vocab = max(1, min(total_words, int(round(cumulative_count))))

    cluster_distribution = Counter(
        int(item["cluster_20"])
        for _, item in matched
        if item.get("cluster_20") is not None
    )
    difficulty_distribution = _difficulty_histogram(difficulties)

    unique_tokens = set(tokens)
    matched_unique = {token for token in unique_tokens if token in entries}
    stage_vocab_coverage = len(matched) / len(tokens)
    coverage_unique = len(matched_unique) / len(unique_tokens)

    return {
        "difficulty_median": round(difficulty_median, 4),
        "estimated_vocab": estimated_vocab,
        "level": _map_level(estimated_vocab, vocab["stage_levels"]),
        "coverage": {
            "stage_vocab": round(stage_vocab_coverage, 4),
            "difficulty_distribution": {
                "p25": round(p25, 4),
                "median": round(difficulty_median, 4),
                "p75": round(p75, 4),
                "min": round(difficulties[0], 4),
                "max": round(difficulties[-1], 4),
                "mean": round(mean(difficulties), 4),
                "histogram": difficulty_distribution,
                "cluster_20": dict(sorted(cluster_distribution.items())),
            },
        },
        "article_stats": {
            "total_tokens": len(WORD_RE.findall(article.lower())),
            "content_tokens": len(tokens),
            "unique_content_words": len(unique_tokens),
            "matched_tokens": len(matched),
            "matched_unique_words": len(matched_unique),
            "unmatched_unique_words": sorted(unique_tokens - matched_unique)[:50],
            "coverage_unique": round(coverage_unique, 4),
        },
        "method": {
            "source": "stage_vocab",
            "stage_vocab_words": total_words,
            "cumulative_percentile": round(cumulative_count / total_words, 4),
            "stopword_count": len(STOPWORDS),
            "lemmatizer": "inline_rule_based",
        },
    }


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute quantile of empty values")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _difficulty_histogram(difficulties: list[float]) -> dict[str, int]:
    bins = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
    }
    for difficulty in difficulties:
        if difficulty < 0.2:
            bins["0.0-0.2"] += 1
        elif difficulty < 0.4:
            bins["0.2-0.4"] += 1
        elif difficulty < 0.6:
            bins["0.4-0.6"] += 1
        elif difficulty < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def _map_level(estimated_vocab: int, stage_levels: list[dict[str, Any]]) -> str:
    if not stage_levels:
        return "未知"

    for level in stage_levels:
        if estimated_vocab <= int(level["cumulative_size"]):
            return str(level["label"])
    return str(stage_levels[-1]["label"])
