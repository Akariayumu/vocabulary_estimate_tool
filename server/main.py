"""英语词汇量估算器的 FastAPI 应用。"""

from __future__ import annotations

import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.stratified_quiz import StratifiedQuiz
from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.article_estimator import estimate_article

from .database import (
    count_test_records,
    get_or_create_student,
    get_student,
    init_db,
    list_test_records,
    save_test_record,
)
from .translations import TRANSLATIONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "web"
VOCAB_VERSION_PATHS = {
    "v1": PROJECT_ROOT / "data" / "stage_vocab.json",
    "original": PROJECT_ROOT / "data" / "stage_vocab.json",
    "v2": PROJECT_ROOT / "data" / "stage_vocab_v2_clusterv1.json",
    "v2_clusterv1": PROJECT_ROOT / "data" / "stage_vocab_v2_clusterv1.json",
}
VOCAB_VERSION_CANONICAL = {
    "v1": "v1",
    "original": "v1",
    "v2": "v2_clusterv1",
    "v2_clusterv1": "v2_clusterv1",
}


class StudentPayload(BaseModel):
    """保存测试记录时使用的学生元数据。"""

    id: int | None = None
    name: str = "匿名学生"
    cet_score: int | None = Field(default=None, ge=0, le=750)


class SaveTestPayload(BaseModel):
    """持久化已完成词汇测试的 payload。"""

    student: StudentPayload = Field(default_factory=StudentPayload)
    responses: list[Any]
    result: dict[str, Any] | None = None


class ArticleEstimatePayload(BaseModel):
    """估算文章词汇需求的 payload。"""

    article: str = Field(..., min_length=1)
    vocab_version: str | None = None


@lru_cache(maxsize=1)
def get_vocab_bank():
    from vocab_estimator.vocab_bank import VocabBank

    return VocabBank(DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def get_estimator():
    from vocab_estimator.vocab_model import VocabEstimator

    return VocabEstimator(get_vocab_bank(), DEFAULT_CONFIG)


def get_sampler(seed: int | None = None) -> VocabularySampler:
    return VocabularySampler(get_vocab_bank(), DEFAULT_CONFIG, seed=seed)


def normalize_vocab_version(vocab_version: str | None = None) -> str:
    """返回规范的 stage-vocab 版本名。"""

    key = str(vocab_version or "v1").strip().lower()
    if key not in VOCAB_VERSION_CANONICAL:
        supported = ", ".join(sorted(VOCAB_VERSION_PATHS))
        raise HTTPException(
            status_code=400,
            detail=f"unsupported vocab_version '{vocab_version}'. Supported: {supported}",
        )
    return VOCAB_VERSION_CANONICAL[key]


def get_vocab_path(vocab_version: str = "v1") -> Path:
    """将 stage-vocab 版本名或别名映射到 JSON 文件路径。"""

    canonical = normalize_vocab_version(vocab_version)
    return VOCAB_VERSION_PATHS[canonical]


def payload_vocab_version(payload: Any, fallback: str | None = None) -> str:
    if isinstance(payload, dict):
        value = payload.get("vocab_version") or payload.get("vocabVersion") or fallback
    else:
        value = fallback
    return normalize_vocab_version(value)


@lru_cache(maxsize=8)
def get_stratified_quiz(
    phase1_question_count: int | None = None,
    vocab_version: str = "v1",
) -> StratifiedQuiz:
    # v2 StratifiedQuiz 是生产测验模型；旧版 v1 endpoints 为兼容性保留。
    question_count = phase1_question_count or DEFAULT_CONFIG.phase1_question_count
    canonical_version = normalize_vocab_version(vocab_version)
    return StratifiedQuiz(
        vocab_bank=None,
        config=DEFAULT_CONFIG,
        phase1_question_count=question_count,
        stage_vocab_path=get_vocab_path(canonical_version),
    )


app = FastAPI(
    title="英语词汇量估算工具",
    description="FastAPI API for vocabulary-size estimation, records and verification.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/estimate")
def estimate(payload: Any = Body(...)) -> dict[str, Any]:
    """根据词-作答对估算单个学习者的词汇量。"""

    responses = parse_response_payload(payload)
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    result = get_estimator().estimate_single(responses)
    return {
        "result": result,
        "input": {"response_count": len(responses)},
        "vocab_bank": vocabulary_summary(),
    }


@app.post("/api/estimate/groups")
def estimate_groups(payload: Any = Body(...)) -> dict[str, Any]:
    """估算 C/F/P/K 学习者组，并报告顺序一致性。"""

    groups = parse_group_payload(payload)
    if not groups:
        raise HTTPException(status_code=400, detail="groups cannot be empty")

    result = get_estimator().estimate_groups(groups)
    return {
        "result": result,
        "input": {"groups": {name: len(rows) for name, rows in groups.items()}},
        "vocab_bank": vocabulary_summary(),
    }


@app.get("/api/vocabulary/stats")
def vocabulary_stats() -> dict[str, Any]:
    """返回词库统计信息。"""

    bank = get_vocab_bank()
    return {
        **vocabulary_summary(),
        "bucket_boundaries": list(DEFAULT_CONFIG.bucket_boundaries),
        "levels": [
            {"name": name, "low": low, "high": high}
            for name, low, high in DEFAULT_CONFIG.levels
        ],
        "config": {
            "default_sample_per_bucket": DEFAULT_CONFIG.default_sample_per_bucket,
            "bootstrap_iterations": DEFAULT_CONFIG.bootstrap_iterations,
            "confidence_interval": DEFAULT_CONFIG.confidence_interval,
        },
        "first_words_by_bucket": {
            bucket: [item.word for item in items[:8]]
            for bucket, items in bank.words_by_bucket.items()
        },
    }


@app.get("/api/vocabulary/sample")
def vocabulary_sample(
    per_bucket: int = Query(default=4, ge=1, le=30),
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    """返回浏览器测试使用的均衡词表。"""

    items = get_sampler(seed).balanced_sample(per_bucket=per_bucket)
    return {
        "items": [
            {"word": word, "rank": rank, "bucket": bucket}
            for word, rank, bucket in items
        ],
        "count": len(items),
        "per_bucket": per_bucket,
    }


@app.get("/api/vocabulary/quiz")
def vocabulary_quiz(
    per_bucket: int = Query(default=4, ge=1, le=30),
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    """返回均衡的浏览器测验，并在可用时附带中文选项。"""

    effective_seed = seed if seed is not None else random.SystemRandom().randint(1, 1_000_000_000)
    rng = random.Random(effective_seed)
    items = get_sampler(effective_seed).balanced_sample(per_bucket=per_bucket)
    questions = [build_quiz_question(word, rank, bucket, rng) for word, rank, bucket in items]
    return {
        "questions": questions,
        "count": len(questions),
        "per_bucket": per_bucket,
        "seed": effective_seed,
    }


@app.post("/api/vocabulary/quiz-stage2")
def vocabulary_quiz_stage2(payload: Any = Body(...)) -> dict[str, Any]:
    """返回面向边界 buckets 的 Stage 2 精细测验题。

    接收 Stage 1（或 adaptive）responses，并为学习者已知率落在不确定区间的
    buckets 生成追加问题。提供 warmup_correct 时会缩小 Stage 2 范围：
    追加题更少，目标边界 buckets 也更少。
    """
    # 同时支持原始 response 列表和带 warmup 信息的 dict
    if isinstance(payload, dict) and "responses" in payload:
        raw_responses = payload["responses"]
        warmup_correct = payload.get("warmup_correct")
    else:
        raw_responses = payload
        warmup_correct = None

    responses = parse_response_payload({"responses": raw_responses} if isinstance(raw_responses, list) else raw_responses)
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    effective_seed = random.SystemRandom().randint(1, 1_000_000_000)
    rng = random.Random(effective_seed)
    sampler = get_sampler(effective_seed)

    # 有 warmup 信息时缩减 Stage 2
    if warmup_correct is not None:
        # 每个 bucket 使用更少追加题（3-4 道，而不是 8 道）
        extra_per_bucket = 3
        # 仅针对估计等级附近的 2-3 个 bucket
        items, boundary_buckets = sampler.stage2_refine_sample(
            responses, extra_per_bucket=extra_per_bucket
        )
        # 如果 bucket 过多，只保留最接近估计边界的 bucket
        if len(boundary_buckets) > 3:
            boundary_buckets = boundary_buckets[:3]
            # 重新采样限制在这些 bucket 内
            seen = {word.lower() for word, _ in responses}
            items = []
            for bucket in boundary_buckets:
                batch = sampler._sample_bucket(bucket, extra_per_bucket, exclude=seen)
                items.extend(batch)
                seen.update(word.lower() for word, _, _ in batch)
    else:
        extra_per_bucket = DEFAULT_CONFIG.stage2_extra_per_bucket
        items, boundary_buckets = sampler.stage2_refine_sample(
            responses, extra_per_bucket=extra_per_bucket
        )

    questions = [
        build_quiz_question(word, rank, bucket, rng)
        for word, rank, bucket in items
    ]

    return {
        "questions": questions,
        "count": len(questions),
        "bounds": [DEFAULT_CONFIG.stage2_boundary_low, DEFAULT_CONFIG.stage2_boundary_high],
        "boundary_buckets": boundary_buckets,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 分层测验 v2 endpoints
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/api/vocabulary/quiz-v2")
def vocabulary_quiz_v2(
    seed: int | None = Query(default=None),
    balanced: bool = Query(default=False, description="Non-adaptive balanced sampling"),
    question_count: int | None = Query(default=None, ge=1, le=40),
    vocab_version: str = Query(default="v1"),
) -> dict[str, Any]:
    """Phase 1：分层采样题目。

    默认返回 30 题，也可配置到旧版 40 题路径。默认使用流式分层采样；
    设置 ``balanced=True`` 可使用简单的每类 2 题采样。
    """
    # 基于时间的 seed 确保每次请求词不同，同时不修改共享 quiz。
    canonical_version = normalize_vocab_version(vocab_version)
    effective_seed = seed if seed is not None else int(time.time() * 1000) % (2**31)
    rng = random.Random(effective_seed)
    quiz = get_stratified_quiz(question_count, canonical_version)
    items = quiz.phase1_sample(adaptive=not balanced, rng=rng)
    questions = [build_v2_question(item, rng) for item in items]

    return {
        "questions": questions,
        "count": len(questions),
        "phase1_question_count": quiz.phase1_question_count,
        "seed": effective_seed,
        "quiz_id": str(effective_seed),
        "balanced": balanced,
        "vocab_version": canonical_version,
        "sampling_info": quiz.get_sampling_info(),
    }


@app.post("/api/vocabulary/quiz-v2/stream")
def vocabulary_quiz_v2_stream(
    payload: Any = Body(...),
    vocab_version: str | None = Query(default=None),
) -> dict[str, Any]:
    """基于已答 v2 responses 做流式估算，并给出接下来的 5 题。"""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")

    raw_responses = payload.get("responses", [])
    responses = parse_response_payload({"responses": raw_responses})
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    question_count = payload.get("phase1_question_count") or payload.get("question_count")
    if question_count is not None:
        try:
            question_count = int(question_count)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="phase1_question_count must be an integer") from exc
        if question_count < 1 or question_count > 40:
            raise HTTPException(status_code=400, detail="phase1_question_count must be between 1 and 40")

    canonical_version = payload_vocab_version(payload, vocab_version)
    quiz = get_stratified_quiz(question_count, canonical_version)
    quiz_id = payload.get("quiz_id")
    if quiz_id is not None:
        try:
            effective_seed = int(quiz_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="quiz_id must be the seed returned by quiz-v2") from exc
    else:
        effective_seed = int(time.time() * 1000) % (2**31)

    sample_rng = random.Random(effective_seed)
    question_rng = random.Random(effective_seed)
    phase1_items = quiz.phase1_sample(rng=sample_rng)
    result = quiz.stream_estimate(
        responses,
        next_count=5,
        phase1_items=phase1_items,
    )
    suggested_questions = [
        build_v2_question(item, question_rng)
        for item in result.get("suggested_items", [])
    ]

    return {
        "result": {
            key: value
            for key, value in result.items()
            if key != "suggested_items"
        },
        "suggested_words": [item["word"] for item in result.get("suggested_items", [])],
        "suggested_questions": suggested_questions,
        "quiz_id": str(effective_seed),
        "phase1_question_count": quiz.phase1_question_count,
        "vocab_version": canonical_version,
    }


@app.post("/api/vocabulary/quiz-v2-stage2")
def vocabulary_quiz_v2_stage2(
    payload: Any = Body(...),
    vocab_version: str | None = Query(default=None),
) -> dict[str, Any]:
    """Phase 2：针对低置信难度类别的精细问题。

    Payload 格式::

        {
            "responses": [{"word": "...", "known": true}, ...],
            "theta": null  # 可选，省略时自动计算
        }
    """
    raw_responses = payload.get("responses", [])
    forced_theta = payload.get("theta")
    canonical_version = payload_vocab_version(payload, vocab_version)

    if not raw_responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    responses = parse_response_payload({"responses": raw_responses})
    quiz = get_stratified_quiz(vocab_version=canonical_version)

    if forced_theta is not None:
        theta = float(forced_theta)
        theta_ci = (theta - 0.5, theta + 0.5)
    else:
        theta, theta_ci = quiz.fit_ability(responses)

    low_conf = quiz._identify_low_confidence(responses)

    # Bug 2：每次请求使用基于时间的 seed
    effective_seed = int(time.time() * 1000) % (2**31)
    rng = random.Random(effective_seed)

    # 提取全部 Phase 1 词，避免 Phase 2 重复测试
    phase1_words = {word.lower() for word, _ in responses}

    phase2_items = quiz.phase2_sample(
        theta,
        low_confidence_classes=low_conf,
        responses=responses,
        n_per_class=4,
        exclude=phase1_words,
    )
    questions = []
    for item in phase2_items:
        diff_label = f"cluster_{item['cluster_20']}"
        q = build_quiz_question(word=item["word"], rank=0, bucket=diff_label, rng=rng)
        q["difficulty"] = round(item["difficulty"], 4)
        q["cluster_20"] = item["cluster_20"]
        q["cluster_100"] = item["cluster_100"]
        questions.append(q)

    return {
        "questions": questions,
        "count": len(questions),
        "theta": round(theta, 4),
        "theta_ci_95": [round(theta_ci[0], 4), round(theta_ci[1], 4)],
        "low_confidence_classes": low_conf,
        "low_confidence_count": len(low_conf),
        "vocab_version": canonical_version,
    }


@app.post("/api/vocabulary/quiz-v2/estimate")
def vocabulary_quiz_v2_estimate(
    payload: Any = Body(...),
    vocab_version: str | None = Query(default=None),
) -> dict[str, Any]:
    """使用 Rasch model 根据 v2 测验 responses 估算词汇量。

    接收所有 Phase 1（以及可选 Phase 2）responses，返回基于 Rasch 的词汇量估算。
    """
    responses = parse_response_payload(payload)
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    canonical_version = payload_vocab_version(payload, vocab_version)
    quiz = get_stratified_quiz(vocab_version=canonical_version)
    result = quiz.estimate_with_ci(responses)

    return {
        "result": result,
        "input": {
            "response_count": len(responses),
            "vocab_version": canonical_version,
        },
        "vocab_version": canonical_version,
    }


@app.post("/api/v2/estimate/article")
def estimate_article_v2(
    payload: ArticleEstimatePayload,
    vocab_version: str | None = Query(default=None),
) -> dict[str, Any]:
    """根据 stage_vocab difficulty 数据估算文章词汇需求。"""

    canonical_version = normalize_vocab_version(vocab_version or payload.vocab_version)
    try:
        result = estimate_article(payload.article, stage_vocab_path=get_vocab_path(canonical_version))
        result["vocab_version"] = canonical_version
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/tests/save")
def save_test(payload: SaveTestPayload) -> dict[str, Any]:
    """将学生完成的测试记录保存到 SQLite。"""

    responses = parse_response_payload(payload.responses)
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    student_payload = payload.student
    if student_payload.id is not None:
        student = get_student(student_payload.id)
        if student is None:
            raise HTTPException(status_code=404, detail="student not found")
    else:
        student = get_or_create_student(student_payload.name, student_payload.cet_score)

    result = payload.result or get_estimator().estimate_single(responses)
    vocabulary_range = result.get("vocabulary_range") or result.get("confidence_interval_90")
    if not vocabulary_range or len(vocabulary_range) != 2:
        raise HTTPException(status_code=400, detail="result must contain vocabulary_range")

    try:
        record = save_test_record(
            student_id=int(student["id"]),
            estimate=int(result["point_estimate"]),
            level=str(result["level"]),
            confidence=str(result["confidence"]),
            range_low=int(vocabulary_range[0]),
            range_high=int(vocabulary_range[1]),
            responses=[{"word": word, "known": known} for word, known in responses],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"record": record}


def build_quiz_question(
    word: str,
    rank: int,
    bucket: str,
    rng: random.Random,
) -> dict[str, Any]:
    """构造一道选择题，或构造二元 fallback 题。

    约 30% 的题是“陷阱题”，四个选项全都错误。
    用户必须选择“没有正确答案”才算答对。
    """

    TRAP_PROBABILITY = 0.1

    answer_text = translation_for(word)
    if answer_text is None:
        return {
            "word": word,
            "rank": rank,
            "bucket": bucket,
            "mode": "binary",
            "options": [],
            "answer": None,
        }

    is_trap = rng.random() < TRAP_PROBABILITY
    distractor_count = 4 if is_trap else 3

    distractors = distractor_translations(word, bucket, answer_text, rng, count=distractor_count)
    if len(distractors) < distractor_count:
        return {
            "word": word,
            "rank": rank,
            "bucket": bucket,
            "mode": "binary",
            "options": [],
            "answer": None,
        }

    if is_trap:
        options = distractors[:4]
        rng.shuffle(options)
        return {
            "word": word,
            "rank": rank,
            "bucket": bucket,
            "mode": "multiple_choice",
            "options": options,
            "answer": None,
        }
    else:
        options = [answer_text, *distractors[:3]]
        rng.shuffle(options)
        return {
            "word": word,
            "rank": rank,
            "bucket": bucket,
            "mode": "multiple_choice",
            "options": options,
            "answer": options.index(answer_text),
        }


def build_v2_question(item: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """根据一个 StratifiedQuiz stage-vocab item 构造浏览器题目。"""

    diff_label = f"cluster_{item.get('cluster_20', 'unknown')}"
    question = build_quiz_question(
        word=str(item["word"]),
        rank=int(item.get("rank") or 0),
        bucket=diff_label,
        rng=rng,
    )
    if "difficulty" in item:
        question["difficulty"] = round(float(item["difficulty"]), 4)
    if "cluster_20" in item:
        question["cluster_20"] = item["cluster_20"]
    if "cluster_100" in item:
        question["cluster_100"] = item["cluster_100"]
    return question


def translation_for(word: str) -> str | None:
    """返回某个词或其归一化 lemma 的中文释义。"""

    lower = word.strip().lower()
    if lower in TRANSLATIONS:
        return TRANSLATIONS[lower]
    return None


def distractor_translations(
    word: str,
    bucket: str,
    answer_text: str,
    rng: random.Random,
    count: int = 3,
) -> list[str]:
    """挑选不重复的中文干扰项，并优先使用相同频率 bucket。

    对 v2 风格 buckets（``"cluster_*"``），此处完全跳过 ``VocabBank``，
    只使用 ``TRANSLATIONS``，避免加载 wordfreq。
    """

    exclude = {word.strip().lower()}

    def collect(candidates: list[str]) -> list[str]:
        values: list[str] = []
        seen = {answer_text}
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        for candidate in shuffled:
            candidate_lower = candidate.lower()
            if candidate_lower in exclude:
                continue
            text = translation_for(candidate)
            if text is None or text in seen:
                continue
            seen.add(text)
            values.append(text)
        return values

    if bucket.startswith("cluster_"):
        # v2 路径：完全跳过 VocabBank，只使用 TRANSLATIONS
        return collect(list(TRANSLATIONS))[:count]

    # v1 路径：使用 VocabBank 选择同 bucket 干扰项
    bank = get_vocab_bank()

    same_bucket_words = [item.word for item in bank.get_items_in_bucket(bucket)]
    all_bank_words = [item.word for item in bank.items]
    all_translation_words = list(TRANSLATIONS)

    distractors = collect(same_bucket_words)
    if len(distractors) < count:
        for text in collect(all_bank_words):
            if text not in distractors:
                distractors.append(text)
            if len(distractors) >= count:
                break
    if len(distractors) < count:
        for text in collect(all_translation_words):
            if text not in distractors:
                distractors.append(text)
            if len(distractors) >= count:
                break

    return distractors


@app.get("/api/tests/records")
def records(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    student_id: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    """查询历史测试记录。"""

    return {
        "records": list_test_records(limit=limit, offset=offset, student_id=student_id),
        "total": count_test_records(student_id=student_id),
        "limit": limit,
        "offset": offset,
    }


def parse_response_payload(payload: Any) -> list[tuple[str, bool]]:
    """将可接受的 response 形状归一化为 ``[(word, known), ...]``。"""

    raw = payload.get("responses") if isinstance(payload, dict) and "responses" in payload else payload
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="responses must be a list")

    responses: list[tuple[str, bool]] = []
    for idx, item in enumerate(raw):
        if isinstance(item, dict):
            word = item.get("word")
            known = item.get("known")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            word = item[0]
            known = item[1]
        else:
            raise HTTPException(status_code=400, detail=f"invalid response at index {idx}")

        if not isinstance(word, str) or not word.strip():
            raise HTTPException(status_code=400, detail=f"word at index {idx} must be a string")
        # 严格布尔校验：拒绝 "false" 这类字符串（非空 str → True 的 bug）
        if not isinstance(known, bool):
            raise HTTPException(status_code=400, detail=f"known at index {idx} must be a boolean (true/false), got {type(known).__name__}")
        responses.append((word.strip(), known))
    return responses


def _strict_bool(val):
    """解析 known 字段：只接受 True/False；'false'/'true' 字符串返回 400。"""
    if isinstance(val, bool):
        return val
    raise HTTPException(status_code=400, detail=f"known must be a boolean, got {type(val).__name__}")


def parse_group_payload(payload: Any) -> dict[str, list[tuple[str, bool]]]:
    """将组 payload 归一化为 ``{group_name: responses}``。"""

    if isinstance(payload, dict) and "groups" in payload:
        raw_groups = payload["groups"]
    elif isinstance(payload, dict) and "responses" in payload and isinstance(payload["responses"], dict):
        raw_groups = payload["responses"]
    else:
        raw_groups = payload
    if not isinstance(raw_groups, dict):
        raise HTTPException(status_code=400, detail="groups must be an object")

    groups: dict[str, list[tuple[str, bool]]] = {}
    for name, raw_responses in raw_groups.items():
        groups[str(name)] = parse_response_payload(raw_responses)
    return groups


def vocabulary_summary() -> dict[str, Any]:
    bank = get_vocab_bank()
    return {
        "size": len(bank),
        "used_fallback": bank.used_fallback,
        "bucket_sizes": bank.bucket_sizes(),
    }


if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")
