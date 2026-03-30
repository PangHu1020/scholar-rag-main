"""Session metadata storage using SQLite (file-based, zero config)."""

import sqlite3
import time
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).parent.parent / "db" / "sessions.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            page_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )"""
    )
    cols = {r[1] for r in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "content_hash" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''")

    conn.commit()
    return conn


def create_session(session_id: str, title: str = "") -> dict:
    now = time.time()
    conn = _get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, title, now, now),
    )
    conn.commit()
    conn.close()
    return {"session_id": session_id, "title": title, "created_at": now, "updated_at": now}


def update_session(session_id: str, title: Optional[str] = None) -> bool:
    conn = _get_conn()
    parts, vals = ["updated_at = ?"], [time.time()]
    if title is not None:
        parts.append("title = ?")
        vals.append(title)
    vals.append(session_id)
    cur = conn.execute(f"UPDATE sessions SET {', '.join(parts)} WHERE session_id = ?", vals)
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def list_sessions() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    ok = cur.rowcount > 0
    conn.close()
    return ok


def add_file(file_id: str, filename: str, paper_id: str, content_hash: str = "",
             size_bytes: int = 0, page_count: int = 0, chunk_count: int = 0) -> dict:
    now = time.time()
    conn = _get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO files (file_id, filename, paper_id, content_hash, size_bytes, page_count, chunk_count, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, filename, paper_id, content_hash, size_bytes, page_count, chunk_count, now),
    )
    conn.commit()
    conn.close()
    return {"file_id": file_id, "filename": filename, "paper_id": paper_id,
            "content_hash": content_hash, "size_bytes": size_bytes,
            "page_count": page_count, "chunk_count": chunk_count, "created_at": now}


def list_files() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM files ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_file(file_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_file_by_hash(content_hash: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM files WHERE content_hash = ?", (content_hash,)).fetchone()
    conn.close()
    return dict(row) if row else None


def delete_file_record(file_id: str) -> Optional[dict]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM files WHERE file_id = ?", (file_id,)).fetchone()
    if not row:
        conn.close()
        return None
    conn.execute("DELETE FROM files WHERE file_id = ?", (file_id,))
    conn.commit()
    conn.close()
    return dict(row)
