from typing import List, Dict, Any
from src.config import config


def evaluate_retrieval(docs: List[Dict[str, Any]]) -> bool:
    if not docs:
        return False

    # FAISS distance/similarity values differ by setup.
    # For project demo, simple heuristic:
    # at least one doc must exist and content must not be tiny.
    best = docs[0]
    return len(best["content"].strip()) > 80


def evaluate_rerank(docs: List[Dict[str, Any]]) -> bool:
    if not docs:
        return False

    best_score = docs[0].get("rerank_score", -999)
    return best_score >= config.RERANK_SCORE_THRESHOLD