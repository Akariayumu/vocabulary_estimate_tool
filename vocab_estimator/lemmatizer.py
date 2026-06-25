"""英语单词的 lemma 归一化。

主实现使用 spaCy 的 ``en_core_web_sm`` 模型。若 spaCy 或模型不可用，
模块会回退到保守的规则型归一化器，保证估算器在受限环境中仍可运行。
"""

from __future__ import annotations

import re
from functools import lru_cache

WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)?$")


class Lemmatizer:
    """将英语屈折词形归一化为 lemma。

    专有名词、数字字符串和疑似缩写会有意保留，不做归一化，
    因为合并它们会扭曲词汇量估算。
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp = self._load_spacy(model_name)

    @staticmethod
    def _load_spacy(model_name: str):
        try:
            import spacy

            return spacy.load(model_name, disable=["parser", "ner"])
        except Exception:
            return None

    @staticmethod
    def is_abbreviation(word: str) -> bool:
        """对 ``NASA`` 这类疑似缩写/acronym 返回 True。"""

        stripped = word.replace(".", "")
        return (
            len(stripped) >= 2
            and stripped.isupper()
            and any(ch.isalpha() for ch in stripped)
        )

    @staticmethod
    def is_proper_like(word: str) -> bool:
        """对不应合并的首字母大写词返回 True。"""

        return len(word) > 1 and word[0].isupper() and word[1:].islower()

    @staticmethod
    def is_valid_word(word: str) -> bool:
        """若 ``word`` 是纯字母英语 token，则返回 True。"""

        return bool(WORD_RE.match(word)) and not any(ch.isdigit() for ch in word)

    def should_preserve(self, word: str) -> bool:
        """当 token 应跳过 lemmatization 时返回 True。"""

        if not word or any(ch.isdigit() for ch in word):
            return True
        if self.is_abbreviation(word):
            return True
        if self.is_proper_like(word):
            return True
        return False

    @lru_cache(maxsize=100_000)
    def normalize(self, word: str) -> str:
        """返回 ``word`` 的归一化 lemma。

        示例：
            ``running`` -> ``run``; ``cars`` -> ``car``; ``NASA`` -> ``NASA``.
        """

        word = word.strip()
        if not word:
            return ""
        if self.should_preserve(word):
            return word

        lower = word.lower()
        if self._nlp is not None:
            doc = self._nlp(lower)
            if doc and doc[0].lemma_ and doc[0].lemma_ != "-PRON-":
                return doc[0].lemma_.lower()

        return self._rule_based_lemma(lower)

    @staticmethod
    def _rule_based_lemma(word: str) -> str:
        """用于常见屈折形式的保守 fallback lemmatizer。"""

        irregular = {
            "children": "child",
            "men": "man",
            "women": "woman",
            "people": "person",
            "mice": "mouse",
            "feet": "foot",
            "teeth": "tooth",
            "went": "go",
            "gone": "go",
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
        }
        if word in irregular:
            return irregular[word]

        if len(word) > 5 and word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 5 and word.endswith("ing"):
            base = word[:-3]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            if base.endswith("mak"):
                return base + "e"
            return base
        if len(word) > 4 and word.endswith("ed"):
            base = word[:-2]
            if len(base) >= 2 and base[-1] == base[-2]:
                base = base[:-1]
            if base.endswith("iz"):
                return base + "e"
            if not base.endswith("e"):
                return base
            return base
        if len(word) > 4 and word.endswith("es"):
            if word.endswith(("ses", "xes", "zes", "ches", "shes")):
                return word[:-2]
        if (
            len(word) > 3
            and word.endswith("s")
            and not word.endswith(("ss", "us", "is", "ous"))
        ):
            return word[:-1]
        return word

    def is_same_lemma(self, w1: str, w2: str) -> bool:
        """若两个表层词形归一化到同一 lemma，则返回 True。"""

        return self.normalize(w1) == self.normalize(w2)
