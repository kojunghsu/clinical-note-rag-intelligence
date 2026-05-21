import json
import os
import sqlite3
from typing import Any, Optional

from src.config import config


def init_cache() -> None:
    os.makedirs(os.path.dirname(config.CACHE_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            ragas_scores TEXT
        )
    """)

    cur.execute("PRAGMA table_info(cache)")
    columns = [row[1] for row in cur.fetchall()]
    if "ragas_scores" not in columns:
        cur.execute("ALTER TABLE cache ADD COLUMN ragas_scores TEXT")

    conn.commit()
    conn.close()


def get_cached_response(query: str) -> Optional[str]:
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT response FROM cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def get_cached_ragas_scores(query: str) -> Optional[dict[str, Any]]:
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT ragas_scores FROM cache WHERE query = ?", (query,))
    row = cur.fetchone()
    conn.close()

    if not row or not row[0]:
        return None

    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return None


def store_response(
    query: str,
    response: str,
    ragas_scores: Optional[dict[str, Any]] = None,
) -> None:
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO cache (query, response, ragas_scores)
        VALUES (?, ?, ?)
    """, (
        query,
        response,
        json.dumps(ragas_scores) if ragas_scores else None,
    ))

    conn.commit()
    conn.close()


def clear_response_cache() -> None:
    init_cache()
    conn = sqlite3.connect(config.CACHE_DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM cache")
    conn.commit()
    conn.close()
