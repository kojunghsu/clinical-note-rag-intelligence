"""
csv_to_sqlite.py — Load discharge notes from DATA_PATH into a local SQLite database.

Usage:
    python -m src.csv_to_sqlite
"""

import os
import sqlite3
import pandas as pd
from src.config import config


def build_sqlite_db():
    if not config.DATA_PATH:
        raise ValueError(
            "DATA_PATH is not set. Add it to your .env file:\n"
            "  DATA_PATH=/path/to/discharge.csv"
        )

    print(f"[CSV_TO_SQLITE] Reading {config.DATA_PATH} ...")
    df = pd.read_csv(
        config.DATA_PATH,
        usecols=["note_id", "subject_id", "hadm_id", "text", "charttime"],
        nrows=config.MAX_ROWS_LOAD,
    )
    print(f"[CSV_TO_SQLITE] Loaded {len(df)} rows.")

    os.makedirs(os.path.dirname(config.SQLITE_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    print(f"[CSV_TO_SQLITE] Writing to {config.SQLITE_DB_PATH} ...")
    df.to_sql("notes", conn, if_exists="replace", index=False)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_subject ON notes(subject_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_note    ON notes(note_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hadm    ON notes(hadm_id)")
    conn.commit()
    conn.close()

    print("[CSV_TO_SQLITE] Done.")


if __name__ == "__main__":
    build_sqlite_db()
