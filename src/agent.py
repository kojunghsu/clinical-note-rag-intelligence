from langgraph.graph import StateGraph, END

from src.state import PipelineState
from src.cache import get_cached_response, get_cached_ragas_scores, store_response
from src.retriever import retrieve
from src.reranker import rerank
from src.llm import generate_answer, fallback_answer
from src.evaluator import evaluate_retrieval, evaluate_rerank
from src.config import config
from src.structured_search import by_note_ids
from src.live_ragas import attach_live_ragas_scores


ANSWER_CACHE_VERSION = "answer_prompt_v3"


def _cache_key(query: str) -> str:
    return f"{ANSWER_CACHE_VERSION}:{query.strip().lower()}"


def analyze_and_rewrite(state: PipelineState) -> PipelineState:
    state.rewritten_query = state.query
    state.debug["rewritten_query"] = state.query
    state.debug["rewrite_disabled"] = True
    return state


def check_cache(state: PipelineState) -> PipelineState:
    print("[AGENT] Cache check")

    key = _cache_key(state.rewritten_query or state.query)
    cached = get_cached_response(key)

    if cached:
        state.cache_hit = True
        state.answer = cached
        state.ragas_scores = get_cached_ragas_scores(key)

    state.debug["cache_lookup_key"] = key
    state.debug["cache_hit"] = state.cache_hit
    state.debug["cached_ragas_scores"] = bool(getattr(state, "ragas_scores", None))

    return state


def retrieve_docs(state: PipelineState) -> PipelineState:
    print("[AGENT] Retrieval step")

    query = state.rewritten_query or state.query
    k = config.TOP_K + state.retry_count * 2

    docs = retrieve(query, k=k)

    state.retrieved_docs = docs
    state.retrieval_good = evaluate_retrieval(docs)

    state.debug["retrieve_query"] = query
    state.debug["retrieved_k"] = k
    state.debug["retrieved_docs_count"] = len(docs)
    state.debug["retrieval_good"] = state.retrieval_good

    return state


def rerank_docs(state: PipelineState) -> PipelineState:
    print("[AGENT] Rerank step")

    query = state.rewritten_query or state.query
    reranked = rerank(query, state.retrieved_docs, config.TOP_N)

    state.reranked_docs = reranked
    state.rerank_good = evaluate_rerank(reranked)

    state.debug["reranked_docs_count"] = len(reranked)
    state.debug["rerank_good"] = state.rerank_good

    return state


def maybe_retry(state: PipelineState) -> PipelineState:
    print("[AGENT] Retry step")

    if (not state.retrieval_good or not state.rerank_good) and state.retry_count < config.MAX_RETRIES:
        state.retry_count += 1
        state.rewritten_query = state.query
        state.debug["retry_triggered"] = True

    return state


def generate(state: PipelineState) -> PipelineState:
    print("[AGENT] Generate step")

    note_ids = []
    seen = set()

    for d in state.reranked_docs:
        metadata = d.get("metadata", {})
        nid = metadata.get("note_id")

        if nid and nid not in seen:
            seen.add(nid)
            note_ids.append(nid)

        if len(note_ids) >= config.COHORT_TOP_N:
            break

    state.debug["parent_note_ids"] = note_ids

    if note_ids:
        df_notes = by_note_ids(note_ids)

        contexts = []
        for i, row in df_notes.iterrows():
            note_id = row.get("note_id", "N/A")
            hadm_id = row.get("hadm_id", "N/A")
            charttime = row.get("charttime", "N/A")
            text = str(row.get("text", ""))

            formatted_context = f"""Parent Note {i + 1}
note_id: {note_id}
hadm_id: {hadm_id}
charttime: {charttime}

{text}"""
            contexts.append(formatted_context)

        state.debug["generate_context_count"] = len(contexts)
        state.answer = generate_answer(state.query, contexts)

    else:
        contexts = []
        for i, d in enumerate(state.reranked_docs, start=1):
            metadata = d.get("metadata", {})
            note_id = metadata.get("note_id", "N/A")
            source = metadata.get("source", "N/A")
            content = d.get("content", "")

            formatted_context = f"""Document {i}
note_id: {note_id}
source: {source}

{content}"""
            contexts.append(formatted_context)

        state.debug["generate_context_count"] = len(contexts)
        state.answer = generate_answer(state.query, contexts)

    state = attach_live_ragas_scores(state, state.query, contexts)
    return state


def fallback(state: PipelineState) -> PipelineState:
    print("[AGENT] Fallback step")

    state.used_fallback = True
    state.debug["used_fallback"] = True
    state.answer = fallback_answer(state.query)

    state = attach_live_ragas_scores(state, state.query)
    return state


def save_cache(state: PipelineState) -> PipelineState:
    key = _cache_key(state.rewritten_query or state.query)

    if state.answer and not state.cache_hit:
        store_response(key, state.answer, state.ragas_scores)

    state.debug["cache_store_key"] = key

    return state


def should_continue_after_cache(state: PipelineState) -> str:
    return "end" if state.cache_hit else "retrieve"


def should_retry_or_generate(state: PipelineState) -> str:
    if state.rerank_good:
        return "generate"
    if state.retry_count < config.MAX_RETRIES:
        return "retry"
    return "fallback"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("rewrite", analyze_and_rewrite)
    graph.add_node("cache", check_cache)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("rerank", rerank_docs)
    graph.add_node("retry", maybe_retry)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)
    graph.add_node("save_cache", save_cache)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "cache")

    graph.add_conditional_edges(
        "cache",
        should_continue_after_cache,
        {
            "retrieve": "retrieve",
            "end": END,
        },
    )

    graph.add_edge("retrieve", "rerank")

    graph.add_conditional_edges(
        "rerank",
        should_retry_or_generate,
        {
            "generate": "generate",
            "retry": "retry",
            "fallback": "fallback",
        },
    )

    graph.add_edge("retry", "retrieve")
    graph.add_edge("generate", "save_cache")
    graph.add_edge("fallback", "save_cache")
    graph.add_edge("save_cache", END)

    return graph.compile()
