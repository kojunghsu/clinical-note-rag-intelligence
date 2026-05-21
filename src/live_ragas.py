"""Live RAGAS scoring for successful app responses.

The Streamlit app uses this module to score each successful RAG response
against the retrieved clinical-note context shown in the Evidence Vault.

UI-facing RAGAS metrics:
- faithfulness
- answer_relevancy
- overall

The overall label follows the original project rule:
High if both RAGAS scores are >= 0.75,
Medium if both are >= 0.50,
and Low otherwise.

Offline/batch evaluation remains available in ragas_evaluation.ipynb.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def _clean_answer(answer: Optional[str]) -> str:
    """Remove the model's self-reported confidence block before RAGAS scoring."""
    if not answer:
        return ""

    text = answer.strip()
    if text.startswith("Answer:"):
        text = text[len("Answer:"):].strip()

    marker = "\nConfidence:"
    if marker in text:
        text = text.split(marker, 1)[0].strip()

    return text


def _normalize_score(value: Any) -> Optional[float]:
    """Convert RAGAS/numpy/pandas scalar outputs into a rounded Python float."""
    if value is None:
        return None

    if isinstance(value, list):
        value = value[0] if value else None

    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _overall_level(faithfulness: float, answer_relevancy: float) -> str:
    if faithfulness >= 0.75 and answer_relevancy >= 0.75:
        return "High"
    if faithfulness >= 0.50 and answer_relevancy >= 0.50:
        return "Medium"
    return "Low"


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Handle common RAGAS result object formats across package versions."""
    if result is None:
        return {}

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        if len(df) > 0:
            return df.iloc[0].to_dict()

    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict):
            return payload

    if isinstance(result, dict):
        return result

    return {}


def docs_to_contexts(
    docs: Iterable[Dict[str, Any]],
    max_contexts: int = 4,
    max_chars: int = 4000,
) -> List[str]:
    """Convert retrieved/reranked docs into context strings for RAGAS."""
    contexts: List[str] = []
    seen = set()

    for doc in docs or []:
        content = str(doc.get("content", "")).strip()
        if not content:
            continue

        metadata = doc.get("metadata", {}) or {}
        note_id = metadata.get("note_id") or metadata.get("source") or "N/A"
        hadm_id = metadata.get("hadm_id") or "N/A"

        key = (note_id, content[:120])
        if key in seen:
            continue
        seen.add(key)

        content = re.sub(r"\s+", " ", content).strip()
        if len(content) > max_chars:
            content = content[:max_chars].rsplit(" ", 1)[0] + "..."

        contexts.append(f"note_id: {note_id}\nhadm_id: {hadm_id}\n\n{content}")
        if len(contexts) >= max_contexts:
            break

    return contexts


def df_to_contexts(
    df: Any,
    max_contexts: int = 4,
    max_chars: int = 4000,
) -> List[str]:
    """Convert direct SQLite lookup DataFrame rows into context strings."""
    contexts: List[str] = []
    if df is None or len(df) == 0:
        return contexts

    for _, row in df.head(max_contexts).iterrows():
        note_id = row.get("note_id", "N/A")
        hadm_id = row.get("hadm_id", "N/A")
        charttime = row.get("charttime", "N/A")
        text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        contexts.append(
            f"note_id: {note_id}\nhadm_id: {hadm_id}\ncharttime: {charttime}\n\n{text}"
        )

    return contexts


def df_to_docs(
    df: Any,
    max_docs: int = 4,
    max_chars: int = 6000,
) -> List[Dict[str, Any]]:
    """Convert direct SQLite lookup DataFrame rows into app Evidence Vault docs."""
    docs: List[Dict[str, Any]] = []
    if df is None or len(df) == 0:
        return docs

    for _, row in df.head(max_docs).iterrows():
        text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
        if len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + "..."
        docs.append(
            {
                "content": text,
                "metadata": {
                    "note_id": row.get("note_id", "N/A"),
                    "hadm_id": row.get("hadm_id", "N/A"),
                    "charttime": row.get("charttime", "N/A"),
                    "source": row.get("note_id", "N/A"),
                },
            }
        )

    return docs


def evaluate_live_ragas(
    question: str,
    answer: str,
    contexts: List[str],
) -> Optional[Dict[str, Any]]:
    """Run RAGAS on one successful response.

    Returns None when there is not enough information to evaluate.
    Raises on package/API errors so callers can log the failure and keep
    the app running.
    """
    cleaned_answer = _clean_answer(answer)
    if not question or not cleaned_answer or not contexts:
        return None

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [cleaned_answer],
            "contexts": [contexts],
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        raise_exceptions=False,
    )
    payload = _result_to_dict(result)

    faithfulness_score = _normalize_score(payload.get("faithfulness"))
    relevancy_score = _normalize_score(
        payload.get("answer_relevancy") or payload.get("answer_relevance")
    )

    if faithfulness_score is None or relevancy_score is None:
        return None

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "overall": _overall_level(faithfulness_score, relevancy_score),
        "source": "live_ragas",
    }


def attach_live_ragas_scores(
    state: Any,
    query: str,
    contexts: Optional[List[str]] = None,
) -> Any:
    """Attach live RAGAS scores to a PipelineState-like object.

    The function is intentionally fail-safe: RAGAS errors never block the normal
    answer or Evidence Vault display.
    """
    if not getattr(state, "answer", None):
        return state

    if contexts is None:
        docs = (
            getattr(state, "reranked_docs", None)
            or getattr(state, "retrieved_docs", None)
            or []
        )
        contexts = docs_to_contexts(docs)

    if not contexts:
        state.ragas_scores = None
        state.debug["ragas_status"] = "skipped_no_contexts"
        return state

    try:
        scores = evaluate_live_ragas(query, state.answer, contexts)
        state.ragas_scores = scores
        state.debug["ragas_status"] = "scored" if scores else "unavailable"
        if scores:
            state.debug["ragas_scores_source"] = scores.get("source")

    except Exception as exc:  # noqa: BLE001 - UI must remain available on evaluator failure.
        print(f"[RAGAS] Evaluation unavailable for this response: {exc}")
        state.ragas_scores = {"error": "RAGAS evaluation unavailable for this response"}
        state.debug["ragas_status"] = "error"
        state.debug["ragas_error"] = str(exc)

    return state
