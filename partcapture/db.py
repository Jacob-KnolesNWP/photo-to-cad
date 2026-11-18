"""SQLite job history. One shop, one queue."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone

from partcapture.config import DB_PATH, ensure_dirs


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class JobDB:
    def __init__(self, path=DB_PATH):
        ensure_dirs()
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                message TEXT NOT NULL DEFAULT '',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT
            )
            """
        )
        self._conn.commit()

    def create(self, job_id: str, mode: str) -> None:
        now = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs (id, mode, status, progress, message, created_at, updated_at) VALUES (?, ?, 'queued', 0, 'Waiting', ?, ?)",
                (job_id, mode, now, now),
            )
            self._conn.commit()

    def claim_next(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            cur = self._conn.execute(
                "UPDATE jobs SET status = 'running', message = 'Starting', updated_at = ? WHERE id = ? AND status = 'queued'",
                (_now(), row["id"]),
            )
            self._conn.commit()
            if cur.rowcount != 1:
                return None
            return row["id"]

    def update(self, job_id: str, *, status: str | None = None, progress: float | None = None, message: str | None = None, error: str | None = None, result_json: str | None = None) -> None:
        fields = ["updated_at = ?"]
        values: list = [_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)
        if message is not None:
            fields.append("message = ?")
            values.append(message)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if result_json is not None:
            fields.append("result_json = ?")
            values.append(result_json)
        values.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
            self._conn.commit()

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_recent(self, limit: int = 40) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, mode, status, progress, message, error, created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()


_db: JobDB | None = None


def get_db() -> JobDB:
    global _db
    if _db is None:
        _db = JobDB()
    return _db
