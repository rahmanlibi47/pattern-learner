from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("DATABASE_PATH", "/app/storage/pattern_learner.sqlite3"))


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    with connect() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                text TEXT NOT NULL,
                url TEXT,
                license TEXT,
                license_url TEXT,
                topic TEXT,
                split TEXT NOT NULL DEFAULT 'train',
                char_count INTEGER NOT NULL,
                collected_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                parent_id TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                epoch INTEGER DEFAULT 0,
                total_epochs INTEGER DEFAULT 0,
                train_loss REAL,
                validation_loss REAL,
                test_loss REAL,
                train_accuracy REAL,
                validation_accuracy REAL,
                test_accuracy REAL,
                parameter_count INTEGER,
                dataset_hash TEXT,
                source_count INTEGER DEFAULT 0,
                source_ids TEXT DEFAULT '[]',
                checkpoint_path TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS runtime (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
            CREATE INDEX IF NOT EXISTS idx_models_status ON models(status);
            """
        )
        article_columns = {row[1] for row in db.execute("PRAGMA table_info(articles)").fetchall()}
        if "split" not in article_columns:
            db.execute("ALTER TABLE articles ADD COLUMN split TEXT NOT NULL DEFAULT 'train'")


def _stable_split(article_id: str) -> str:
    bucket = int(hashlib.sha256(article_id.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _ensure_split_coverage(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT id, split FROM articles ORDER BY id").fetchall()
    if len(rows) < 3:
        return
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    for missing in (name for name, count in counts.items() if count == 0):
        donor = max(counts, key=counts.get)
        if counts[donor] <= 1:
            break
        candidate = next(row for row in rows if row["split"] == donor)
        db.execute("UPDATE articles SET split = ? WHERE id = ?", (missing, candidate["id"]))
        counts[donor] -= 1
        counts[missing] += 1


def emit(event_type: str, payload: dict[str, Any]) -> None:
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO events(event_type, payload, created_at) VALUES (?, ?, ?)",
            (event_type, json.dumps(payload), utcnow()),
        )
        if cursor.lastrowid and cursor.lastrowid % 500 == 0:
            db.execute("DELETE FROM events WHERE id < ?", (cursor.lastrowid - 5000,))


def set_runtime(**values: Any) -> None:
    now = utcnow()
    with connect() as db:
        for key, value in values.items():
            db.execute(
                "INSERT INTO runtime(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(value), now),
            )
    emit("status", values)


def get_runtime() -> dict[str, Any]:
    with connect() as db:
        rows = db.execute("SELECT key, value FROM runtime").fetchall()
    result: dict[str, Any] = {}
    for row in rows:
        try:
            result[row["key"]] = json.loads(row["value"])
        except json.JSONDecodeError:
            result[row["key"]] = row["value"]
    return result


def add_articles(articles: list[dict[str, Any]]) -> int:
    inserted = 0
    with connect() as db:
        for article in articles:
            text = article["text"]
            article_id = hashlib.sha256(
                (article.get("url", "") + "\n" + text).encode("utf-8")
            ).hexdigest()
            cursor = db.execute(
                """INSERT OR IGNORE INTO articles
                (id, title, text, url, license, license_url, topic, split, char_count, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article_id, article["title"], text, article.get("url"),
                    article.get("license"), article.get("license_url"),
                    article.get("topic"), _stable_split(article_id), len(text), utcnow(),
                ),
            )
            inserted += cursor.rowcount
        _ensure_split_coverage(db)
    return inserted


def list_articles(limit: int = 200) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT id, title, url, license, license_url, topic, split, char_count, collected_at "
            "FROM articles ORDER BY collected_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_articles_with_text() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM articles ORDER BY id").fetchall()
    return [dict(row) for row in rows]


def create_model(model_id: str, parent_id: str | None, epochs: int, source_ids: list[str], dataset_hash: str) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO models(id,status,parent_id,created_at,total_epochs,source_count,source_ids,dataset_hash)
            VALUES (?, 'training', ?, ?, ?, ?, ?, ?)""",
            (model_id, parent_id, utcnow(), epochs, len(source_ids), json.dumps(source_ids), dataset_hash),
        )


def update_model(model_id: str, **fields: Any) -> None:
    if not fields:
        return
    allowed = {
        "status", "completed_at", "epoch", "train_loss", "validation_loss", "test_loss",
        "train_accuracy", "validation_accuracy", "test_accuracy", "parameter_count",
        "checkpoint_path", "error",
    }
    fields = {key: value for key, value in fields.items() if key in allowed}
    assignments = ", ".join(f"{key} = ?" for key in fields)
    with connect() as db:
        db.execute(
            f"UPDATE models SET {assignments} WHERE id = ?",
            (*fields.values(), model_id),
        )


def list_models() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM models ORDER BY created_at DESC").fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["source_ids"] = json.loads(item.get("source_ids") or "[]")
        result.append(item)
    return result


def get_model(model_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["source_ids"] = json.loads(item.get("source_ids") or "[]")
    return item


def latest_ready_model() -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            "SELECT * FROM models WHERE status='ready' ORDER BY completed_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    item["source_ids"] = json.loads(item.get("source_ids") or "[]")
    return item


def latest_model() -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM models ORDER BY created_at DESC LIMIT 1").fetchone()
    if not row:
        return None
    item = dict(row)
    item["source_ids"] = json.loads(item.get("source_ids") or "[]")
    return item


def latest_event_id() -> int:
    with connect() as db:
        return int(db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0])


def get_events(after_id: int = 0, limit: int = 100) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute(
            "SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?", (after_id, limit)
        ).fetchall()
    return [
        {"id": row["id"], "type": row["event_type"], "data": json.loads(row["payload"]), "created_at": row["created_at"]}
        for row in rows
    ]


def counts() -> dict[str, int]:
    with connect() as db:
        article_count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        model_count = db.execute("SELECT COUNT(*) FROM models WHERE status='ready'").fetchone()[0]
        char_count = db.execute("SELECT COALESCE(SUM(char_count),0) FROM articles").fetchone()[0]
    return {"articles": article_count, "ready_models": model_count, "characters": char_count}
