"""基于频率 rank 构建词库。

词库优先使用 ``wordfreq`` 提供 rank 数据；依赖不可用时回退到紧凑的内置列表。
词会先合并为 lemma families，经过过滤后再分配到 frequency-rank buckets。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DEFAULT_CONFIG, EstimatorConfig, bucket_label
from .lemmatizer import Lemmatizer


COMMON_PROPER_NAMES = {
    "john",
    "mary",
    "james",
    "robert",
    "michael",
    "william",
    "david",
    "richard",
    "thomas",
    "london",
    "paris",
    "china",
    "america",
    "england",
}


FALLBACK_WORDS = """
the of and to in a is that for it as was with be by on not he i this are or his
from at which but have an had they you were her one we all their there can has
more will if about who when what so up out them some into no time people year
your good would could other than then now only its over think also back after use
two how our work first well way even new want because any these give day most us
life man woman child school house hand world place while last still long great
small part large number group problem fact old high different right public sure
left few same tell young important every might never under again family own state
student country city question answer friend mother father water food room book
story idea light body name line end side home face head eye door car road air
money night morning job word hear play run walk read write speak learn study make
take come go see know find ask help feel need keep start call move live believe
turn show bring sit stand meet include continue set change follow stop create open
close begin grow happen provide pay lose win buy sell build choose explain develop
carry remember reach report decide receive return appear accept consider describe
expect improve compare control suggest notice protect discuss prepare require
contain support produce avoid imagine manage connect discover publish achieve
argue examine establish influence recognize maintain participate investigate
calculate demonstrate interpret evaluate negotiate emphasize transform coordinate
illustrate justify implement formulate synthesize
happy sad easy hard simple clear poor rich strong weak true false full empty hot
cold black white red blue green early late short tall fast slow kind serious
possible natural social political economic personal local national international
modern traditional common special major minor final central general specific
available necessary independent professional academic practical cultural physical
mental emotional successful difficult excellent obvious familiar ordinary
complex efficient accurate appropriate significant relevant reliable
cat dog bird fish tree flower river mountain sea sun moon star rain wind snow
music movie game sport art science history math computer phone internet energy
health medicine hospital doctor teacher worker farmer police soldier lawyer
market office company industry system service product research evidence theory
language grammar vocabulary sentence paragraph article document novel poem
library museum theater restaurant airport station hotel bank church kitchen
garden college university exam lesson course degree lecture homework
justice freedom peace power policy law crime court government election economy
trade tax price cost value profit budget investment contract
atom cell gene protein molecule bacteria virus climate planet universe galaxy
engine machine vehicle signal network database algorithm software hardware
analysis strategy method process structure function factor pattern variable
hypothesis experiment observation measurement conclusion
abandon ability absence abundant accelerate accommodation accomplish accumulate
acknowledge acquire adapt adequate adjacent adjust administer adolescent advocate
allocate ambiguous analogy analyze ancestor anecdote anticipate apparatus
applicant arbitrary articulate asset assumption atmosphere attribute authentic
beneficial bias capacity coherent collapse commodity compensate competent
compile complement comprehensive conceive concurrent confer consequence
constrain contemporary contradict controversy convene criterion crucial currency
deduce deficiency denote diminish discrete displace diverse duration elaborate
empirical enhance entity equivalent exceed exclude explicit facilitate fluctuate
framework hierarchy identical ideology implicit impose incentive incorporate
index inhibit initiate integrate intermediate intrinsic invoke isolate levy
mediate migrate minimal mutual norm objective offset orient paradigm passive
precede precise preliminary presume priority prohibit proportion qualitative
quantify radical rational reinforce replicate restore revenue rigid scenario
simulate spectrum subsequent supplement suspend sustain symbolic terminate
thereby transparent ultimate undergo valid violate voluntary whereas
abdicate aberration abrogate acumen adroit aesthetic affable affinity altruism
anachronism apathy archetype assiduous austere belligerent benevolent cacophony
capricious circumspect clandestine cogent complacent conundrum decorum defer
deleterious demagogue dichotomy didactic diffident dogmatic eclectic efface
egregious elicit enervate ephemeral equivocal esoteric fastidious fortuitous
gregarious hegemony idiosyncrasy ignominious immutable impetuous incongruous
ineffable intransigent laconic magnanimous meticulous obfuscate obstinate
paradox perfunctory perspicacious pragmatic prodigal quixotic reticent sanguine
scrupulous serendipity taciturn ubiquitous vacillate venerable vindicate zealot
""".split()


@dataclass(frozen=True)
class VocabItem:
    """词库中的一个 lemma-family 条目。"""

    word: str
    lemma: str
    rank: int
    bucket: str


class VocabBank:
    """构建并查询按频率 rank 排序的英语 lemma-family 词库。"""

    def __init__(
        self,
        config: EstimatorConfig = DEFAULT_CONFIG,
        lemmatizer: Lemmatizer | None = None,
        words_with_ranks: Iterable[tuple[str, int]] | None = None,
    ) -> None:
        self.config = config
        self.lemmatizer = lemmatizer or Lemmatizer()
        self._wordfreq_available = False
        source = list(words_with_ranks) if words_with_ranks is not None else self._load_ranked_words()
        self.items = self._build_items(source)
        self.rank_by_word = {item.word: item.rank for item in self.items}
        self.rank_by_lemma = {item.lemma: item.rank for item in self.items}
        self.item_by_word = {item.word: item for item in self.items}
        self.item_by_lemma = {item.lemma: item for item in self.items}
        self.words_by_bucket = self._index_buckets()
        self.used_fallback = words_with_ranks is None and not self._wordfreq_available

    def _load_ranked_words(self) -> list[tuple[str, int]]:
        """返回来自 wordfreq 或内置 fallback 的带 rank 词形。
        使用 pickle cache，避免重启时重复加载 200MB 的 wordfreq 字典。"""

        import pickle as _pickle
        import time as _time

        cache_path = Path("/tmp/vocab_bank_words.pkl")
        
        # 优先尝试 pickle cache
        if cache_path.exists():
            try:
                t0 = _time.time()
                with open(cache_path, "rb") as f:
                    words = _pickle.load(f)
                self._wordfreq_available = True
                print(f"  VocabBank loaded {len(words)} words from cache in {_time.time()-t0:.1f}s")
                return words
            except Exception:
                cache_path.unlink(missing_ok=True)

        self._wordfreq_available = False
        try:
            from wordfreq import top_n_list

            t0 = _time.time()
            words = top_n_list("en", self.config.vocab_size * 3)
            result = [(word, rank) for rank, word in enumerate(words, start=1)]
            print(f"  wordfreq loaded {len(result)} words in {_time.time()-t0:.1f}s")
            
            # 保存到 cache
            try:
                with open(cache_path, "wb") as f:
                    _pickle.dump(result, f)
                print(f"  Cache saved to {cache_path}")
            except Exception:
                pass
            
            self._wordfreq_available = True
            return result
        except Exception:
            return self._fallback_words()

    def _fallback_words(self) -> list[tuple[str, int]]:
        """返回一个小型人工整理频率列表，并附带合成 rank。"""

        seen: set[str] = set()
        unique_words: list[str] = []
        for word in FALLBACK_WORDS:
            clean = word.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                unique_words.append(clean)
        if len(unique_words) == 1:
            return [(unique_words[0], 1)]
        ranked: list[tuple[str, int]] = []
        for idx, clean in enumerate(unique_words):
            rank = 1 + round(idx * (self.config.vocab_size - 1) / (len(unique_words) - 1))
            ranked.append((clean, rank))
        return ranked

    def _build_items(self, words_with_ranks: Iterable[tuple[str, int]]) -> list[VocabItem]:
        """过滤、lemmatize 并将带 rank 的词放入 bucket。"""

        best_by_lemma: dict[str, tuple[str, int]] = {}
        for raw_word, raw_rank in words_with_ranks:
            word = raw_word.strip()
            if not self._passes_filter(word):
                continue
            lemma = self.lemmatizer.normalize(word.lower())
            if not lemma or not self._passes_filter(lemma):
                continue
            previous = best_by_lemma.get(lemma)
            rank = int(raw_rank)
            if previous is None or rank < previous[1]:
                best_by_lemma[lemma] = (word.lower(), rank)

        items: list[VocabItem] = []
        for lemma, (word, rank) in sorted(best_by_lemma.items(), key=lambda x: x[1][1]):
            if rank > self.config.vocab_size:
                continue
            bucket = self.bucket_for_rank(rank)
            if bucket is None:
                continue
            items.append(VocabItem(word=word, lemma=lemma, rank=rank, bucket=bucket))
        return items

    def _passes_filter(self, word: str) -> bool:
        """对普通英语词汇项返回 True。"""

        if len(word) < self.config.min_word_len:
            return False
        if not word.isalpha():
            return False
        if word.lower() in COMMON_PROPER_NAMES:
            return False
        if word.isupper() and len(word) <= self.config.abbreviation_max_len:
            return False
        return True

    def _index_buckets(self) -> dict[str, list[VocabItem]]:
        buckets = {bucket_label(boundary): [] for boundary in self.config.bucket_boundaries}
        for item in self.items:
            buckets[item.bucket].append(item)
        return buckets

    def bucket_for_rank(self, rank: int) -> str | None:
        """返回频率 rank 对应的 bucket 标签；超出范围则返回 None。"""

        for boundary in self.config.bucket_boundaries:
            if rank <= boundary:
                return bucket_label(boundary)
        return None

    def get_words_in_bucket(self, bucket: str | int) -> list[str]:
        """返回某个 bucket 中的代表词。

        Args:
            bucket: 可以是 ``"5k"`` 这样的紧凑标签，也可以是 ``5000``
                这样的数值上界。
        """

        label = bucket_label(bucket) if isinstance(bucket, int) else bucket.lower()
        return [item.word for item in self.words_by_bucket.get(label, [])]

    def get_items_in_bucket(self, bucket: str | int) -> list[VocabItem]:
        """返回某个 bucket 中的完整词库条目。"""

        label = bucket_label(bucket) if isinstance(bucket, int) else bucket.lower()
        return list(self.words_by_bucket.get(label, []))

    def get_rank(self, word: str) -> int | None:
        """返回 ``word`` 或其 lemma 的频率 rank。"""

        raw = word.strip()
        if not raw:
            return None
        lower = raw.lower()
        if lower in self.rank_by_word:
            return self.rank_by_word[lower]
        lemma = self.lemmatizer.normalize(raw)
        return self.rank_by_lemma.get(lemma.lower(), self.rank_by_word.get(lemma.lower()))

    def get_bucket(self, word: str) -> str | None:
        """若 ``word`` 存在于词库中，返回其频率 bucket。"""

        rank = self.get_rank(word)
        return self.bucket_for_rank(rank) if rank is not None else None

    def bucket_sizes(self) -> dict[str, int]:
        """返回每个 bucket 中 lemma-family 条目的数量。"""

        return {label: len(items) for label, items in self.words_by_bucket.items()}

    def ranks(self) -> list[int]:
        """返回所有保留下来的 lemma-family ranks。"""

        return [item.rank for item in self.items]

    def __len__(self) -> int:
        return len(self.items)
