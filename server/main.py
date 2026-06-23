"""FastAPI application for the English vocabulary-size estimator."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vocab_estimator.config import DEFAULT_CONFIG
from vocab_estimator.sampler import VocabularySampler
from vocab_estimator.vocab_bank import VocabBank
from vocab_estimator.vocab_model import VocabEstimator

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


class StudentPayload(BaseModel):
    """Student metadata used when saving a test record."""

    id: int | None = None
    name: str = "匿名学生"
    cet_score: int | None = Field(default=None, ge=0, le=750)


class SaveTestPayload(BaseModel):
    """Payload for persisting a completed vocabulary test."""

    student: StudentPayload = Field(default_factory=StudentPayload)
    responses: list[Any]
    result: dict[str, Any] | None = None


@lru_cache(maxsize=1)
def get_vocab_bank() -> VocabBank:
    return VocabBank(DEFAULT_CONFIG)


@lru_cache(maxsize=1)
def get_estimator() -> VocabEstimator:
    return VocabEstimator(get_vocab_bank(), DEFAULT_CONFIG)


def get_sampler(seed: int | None = None) -> VocabularySampler:
    return VocabularySampler(get_vocab_bank(), DEFAULT_CONFIG, seed=seed)


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
    """Estimate one learner's vocabulary size from word-response pairs."""

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
    """Estimate C/F/P/K learner groups and report ordering consistency."""

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
    """Return vocabulary-bank statistics."""

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
    """Return a balanced word list for the browser test."""

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
    """Return a balanced browser quiz with Chinese options when available."""

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
    """Return Stage 2 refined quiz questions targeting boundary buckets.

    Accepts the Stage 1 responses and generates additional questions for
    buckets where the learner's known-rate is in the uncertain range.
    """
    responses = parse_response_payload(payload)
    if not responses:
        raise HTTPException(status_code=400, detail="responses cannot be empty")

    effective_seed = random.SystemRandom().randint(1, 1_000_000_000)
    rng = random.Random(effective_seed)
    sampler = get_sampler(effective_seed)
    extra_per_bucket = DEFAULT_CONFIG.stage2_extra_per_bucket

    items, boundary_buckets = sampler.stage2_refine_sample(
        responses, extra_per_bucket=extra_per_bucket
    )

    # Build quiz questions from the Stage 2 items
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


@app.post("/api/tests/save")
def save_test(payload: SaveTestPayload) -> dict[str, Any]:
    """Save a student's completed test record into SQLite."""

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
    """Build one multiple-choice question or a binary fallback item.

    ~30% of questions are "trap questions" where all four options are wrong.
    Users must pick "没有正确答案" to answer correctly.
    """

    TRAP_PROBABILITY = 0.3

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


def translation_for(word: str) -> str | None:
    """Return a Chinese gloss for a word or its normalized lemma."""

    lower = word.strip().lower()
    if lower in TRANSLATIONS:
        return TRANSLATIONS[lower]
    lemma = get_vocab_bank().lemmatizer.normalize(lower).lower()
    return TRANSLATIONS.get(lemma)


def distractor_translations(
    word: str,
    bucket: str,
    answer_text: str,
    rng: random.Random,
    count: int = 3,
) -> list[str]:
    """Pick unique Chinese distractors, preferring the same frequency bucket."""

    bank = get_vocab_bank()
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
    """Query historical test records."""

    return {
        "records": list_test_records(limit=limit, offset=offset, student_id=student_id),
        "total": count_test_records(student_id=student_id),
        "limit": limit,
        "offset": offset,
    }


def parse_response_payload(payload: Any) -> list[tuple[str, bool]]:
    """Normalize accepted response shapes into ``[(word, known), ...]``."""

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
        responses.append((word.strip(), bool(known)))
    return responses


def parse_group_payload(payload: Any) -> dict[str, list[tuple[str, bool]]]:
    """Normalize group payloads into ``{group_name: responses}``."""

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
