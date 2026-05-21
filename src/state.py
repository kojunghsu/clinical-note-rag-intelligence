from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    cache_hit: bool = False
    cached_response: Optional[str] = None

    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    reranked_docs: List[Dict[str, Any]] = Field(default_factory=list)

    retrieval_good: bool = False
    rerank_good: bool = False
    retry_count: int = 0

    answer: Optional[str] = None
    used_fallback: bool = False
    ragas_scores: Optional[Dict[str, Any]] = None
    debug: Dict[str, Any] = Field(default_factory=dict)
