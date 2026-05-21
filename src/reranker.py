from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

# Good small local reranker model
_MODEL_NAME = "BAAI/bge-reranker-base"
_model = None


def get_reranker():
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    if not docs:
        return []
    
    print(f"[RERANK] Re-ranking {len(docs)} docs...")

    model = get_reranker()
    pairs = [[query, d["content"]] for d in docs]
    scores = model.predict(pairs)

    rescored = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(score)
        rescored.append(item)

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    print(f"[RERANK] Returning top {top_n}")
    return rescored[:top_n]