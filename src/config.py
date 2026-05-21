import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
    OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # ── Data paths (all configurable via .env) ────────────────────────────────
    DATA_PATH         = os.getenv("DATA_PATH", "./data/sample/discharge_sample.csv")
    FAISS_INDEX_DIR   = os.getenv("FAISS_INDEX_DIR", "./storage/faiss_index")
    CACHE_DB_PATH     = os.getenv("CACHE_DB_PATH", "./storage/cache.sqlite")
    SQLITE_DB_PATH    = os.getenv("SQLITE_DB_PATH", "./storage/notes.db")
    CHECKPOINT_PATH   = os.getenv("CHECKPOINT_PATH", "./storage/embedding_checkpoint.pkl")

    # ── Retrieval / generation hyperparameters ────────────────────────────────
    TOP_K             = int(os.getenv("TOP_K", "8"))
    TOP_N             = int(os.getenv("TOP_N", "3"))
    COHORT_TOP_N      = int(os.getenv("COHORT_TOP_N", "4"))
    MAX_RETRIES       = int(os.getenv("MAX_RETRIES", "2"))
    MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "14000"))
    MAX_ROWS_LOAD     = int(os.getenv("MAX_ROWS_LOAD", "1000"))

    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.45"))
    RERANK_SCORE_THRESHOLD    = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.20"))


config = Config()
