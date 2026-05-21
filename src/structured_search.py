import sqlite3
import pandas as pd
from src.config import config


def get_conn():
    conn = sqlite3.connect(config.SQLITE_DB_PATH)

    chk = conn.execute("""
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name='notes'
    """).fetchone()

    if chk is None:
        conn.close()
        raise RuntimeError(
            "SQLite table 'notes' not found. Run: python -m src.csv_to_sqlite"
        )

    return conn


def by_subject_id(subject_id):
    conn = get_conn()

    print(f"[STRUCTURED_SEARCH] subject_id lookup: {subject_id}")

    df = pd.read_sql_query(
        """
        SELECT note_id, subject_id, hadm_id, charttime, text
        FROM notes
        WHERE subject_id = ?
        ORDER BY charttime DESC
        LIMIT 4
        """,
        conn,
        params=(int(subject_id),)
    )

    conn.close()

    print(f"[STRUCTURED_SEARCH] Rows returned: {len(df)}")

    return df


def by_note_id(note_id):
    conn = get_conn()

    print(f"[STRUCTURED_SEARCH] note_id lookup: {note_id}")

    df = pd.read_sql_query(
        """
        SELECT note_id, subject_id, hadm_id, charttime, text
        FROM notes
        WHERE note_id = ?
        """,
        conn,
        params=(str(note_id),)
    )

    conn.close()

    print(f"[STRUCTURED_SEARCH] Rows returned: {len(df)}")

    return df


def by_hadm_id(hadm_id):
    conn = get_conn()

    print(f"[STRUCTURED_SEARCH] hadm_id lookup: {hadm_id}")

    df = pd.read_sql_query(
        """
        SELECT note_id, subject_id, hadm_id, charttime, text
        FROM notes
        WHERE hadm_id = ?
        ORDER BY charttime DESC
        """,
        conn,
        params=(int(hadm_id),)
    )

    conn.close()

    print(f"[STRUCTURED_SEARCH] Rows returned: {len(df)}")

    return df


def by_note_ids(note_ids):
    conn = get_conn()

    placeholders = ",".join(["?"] * len(note_ids))

    query = f"""
        SELECT note_id, subject_id, hadm_id, charttime, text
        FROM notes
        WHERE note_id IN ({placeholders})
        ORDER BY charttime DESC
    """

    df = pd.read_sql_query(
        query,
        conn,
        params=tuple(str(x) for x in note_ids)
    )

    conn.close()

    print(f"[STRUCTURED_SEARCH] Full notes fetched: {len(df)}")

    return df