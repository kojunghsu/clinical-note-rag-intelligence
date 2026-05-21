import os
import json
from datetime import datetime


LOG_DIR = "./logs"


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _safe_preview_docs(docs):
    out = []

    for d in docs:
        try:
            # dict style
            out.append({
                "metadata": d.get("metadata", {}),
                "preview": d.get("content", "")[:300]
            })
        except:
            try:
                # LangChain Document style
                out.append({
                    "metadata": getattr(d, "metadata", {}),
                    "preview": getattr(d, "page_content", "")[:300]
                })
            except:
                pass

    return out


def log_run(obj):
    ensure_log_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"log_{timestamp}.json"
    path = os.path.join(LOG_DIR, filename)

    # -----------------------------------
    # CASE 1: dict logs from pipeline.py
    # -----------------------------------
    if isinstance(obj, dict):
        log_data = obj

    # -----------------------------------
    # CASE 2: PipelineState object
    # -----------------------------------
    else:
        log_data = {
            "query": getattr(obj, "query", None),
            "rewritten_query": getattr(obj, "rewritten_query", None),
            "cache_hit": getattr(obj, "cache_hit", False),
            "retry_count": getattr(obj, "retry_count", 0),
            "used_fallback": getattr(obj, "used_fallback", False),
            "answer": getattr(obj, "answer", None),
            "retrieved_docs": _safe_preview_docs(
                getattr(obj, "retrieved_docs", [])
            ),
            "reranked_docs": _safe_preview_docs(
                getattr(obj, "reranked_docs", [])
            ),
            "debug": getattr(obj, "debug", {})
        }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"Log saved: {path}")