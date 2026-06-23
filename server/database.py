"""SQLite persistence for student vocabulary-test records."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "vocab_estimator.sqlite3"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with dictionary-like rows."""

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create database tables if they do not already exist."""

    with get_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                cet_score INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            );

            CREATE TABLE IF NOT EXISTS test_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                estimate INTEGER NOT NULL,
                level TEXT NOT NULL,
                confidence TEXT NOT NULL,
                range_low INTEGER NOT NULL,
                range_high INTEGER NOT NULL,
                responses_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_test_records_student_id
                ON test_records(student_id);
            CREATE INDEX IF NOT EXISTS idx_test_records_created_at
                ON test_records(created_at DESC);
            """
        )


def create_student(
    name: str,
    cet_score: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Insert a student and return the created row."""

    clean_name = name.strip()
    if not clean_name:
        raise ValueError("student name cannot be empty")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO students (name, cet_score) VALUES (?, ?)",
            (clean_name, cet_score),
        )
        student_id = int(cursor.lastrowid)
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return _row_to_dict(row)


def get_student(student_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return a student by id, or None."""

    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    return _row_to_dict(row) if row else None


def find_latest_student_by_name(
    name: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any] | None:
    """Return the most recently created student with this name."""

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM students
            WHERE name = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (name.strip(),),
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_or_create_student(
    name: str,
    cet_score: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Reuse a same-name student row when possible; otherwise create one."""

    existing = find_latest_student_by_name(name, db_path)
    if existing is not None:
        if cet_score is not None and existing.get("cet_score") != cet_score:
            with get_connection(db_path) as conn:
                conn.execute(
                    "UPDATE students SET cet_score = ? WHERE id = ?",
                    (cet_score, existing["id"]),
                )
            existing["cet_score"] = cet_score
        return existing
    return create_student(name, cet_score, db_path)


def save_test_record(
    student_id: int,
    estimate: int,
    level: str,
    confidence: str,
    range_low: int,
    range_high: int,
    responses: list[dict[str, Any]] | list[list[Any]] | list[tuple[Any, ...]],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> dict[str, Any]:
    """Insert a test record and return the saved row with student metadata."""

    if get_student(student_id, db_path) is None:
        raise ValueError(f"student_id {student_id} does not exist")

    responses_json = json.dumps(responses, ensure_ascii=False)
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO test_records (
                student_id, estimate, level, confidence,
                range_low, range_high, responses_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(student_id),
                int(estimate),
                str(level),
                str(confidence),
                int(range_low),
                int(range_high),
                responses_json,
            ),
        )
        record_id = int(cursor.lastrowid)
    return get_test_record(record_id, db_path) or {}


def get_test_record(record_id: int, db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Return one saved test record with joined student data."""

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                tr.id, tr.student_id, s.name AS student_name, s.cet_score,
                tr.estimate, tr.level, tr.confidence, tr.range_low, tr.range_high,
                tr.responses_json, tr.created_at
            FROM test_records tr
            JOIN students s ON s.id = tr.student_id
            WHERE tr.id = ?
            """,
            (record_id,),
        ).fetchone()
    return _record_row_to_dict(row) if row else None


def list_test_records(
    limit: int = 50,
    offset: int = 0,
    student_id: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[dict[str, Any]]:
    """Return recent test records, newest first."""

    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    params: list[Any] = []
    where = ""
    if student_id is not None:
        where = "WHERE tr.student_id = ?"
        params.append(int(student_id))

    params.extend([limit, offset])
    with get_connection(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT
                tr.id, tr.student_id, s.name AS student_name, s.cet_score,
                tr.estimate, tr.level, tr.confidence, tr.range_low, tr.range_high,
                tr.responses_json, tr.created_at
            FROM test_records tr
            JOIN students s ON s.id = tr.student_id
            {where}
            ORDER BY tr.created_at DESC, tr.id DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
    return [_record_row_to_dict(row) for row in rows]


def count_test_records(
    student_id: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> int:
    """Return the number of saved test records."""

    with get_connection(db_path) as conn:
        if student_id is None:
            row = conn.execute("SELECT COUNT(*) AS n FROM test_records").fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM test_records WHERE student_id = ?",
                (int(student_id),),
            ).fetchone()
    return int(row["n"]) if row else 0


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def _record_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = _row_to_dict(row)
    raw = data.pop("responses_json", "[]")
    try:
        data["responses"] = json.loads(raw)
    except json.JSONDecodeError:
        data["responses"] = []
    data["range"] = [data["range_low"], data["range_high"]]
    return data

