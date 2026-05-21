from src.state import PipelineState
from src.agent import build_graph
from src.logger import log_run
from src.parser import parse_query
from src.structured_search import by_subject_id, by_note_id, by_hadm_id
from src.config import config
from src.llm import (
    summarize_patient_notes,
    summarize_single_note,
    summarize_visit_notes
)
from src.live_ragas import attach_live_ragas_scores, df_to_contexts, df_to_docs

graph = build_graph()


def _join_text(df):
    print("[PIPELINE] Notes fetched:", len(df))

    texts = []
    for i, row in df.iterrows():
        txt = str(row["text"])
        note_id = row.get("note_id", "N/A")
        charttime = row.get("charttime", "N/A")
        print(f"[PIPELINE] note_id={note_id} charttime={charttime} chars={len(txt)}")

        texts.append(
            f"--- NOTE {i+1} | note_id={note_id} | charttime={charttime} ---\n{txt}"
        )

    joined = "\n\n".join(texts)

    print("[PIPELINE] Total joined chars:", len(joined))
    return joined


def run_pipeline(query: str):
    print(f"\n[PIPELINE] Query: {query}")
    print("[PIPELINE] Running parser...")

    log = {"query": query}

    parsed = parse_query(query)
    log["parsed_json"] = parsed

    # -------------------------------
    # DIRECT LOOKUPS
    # -------------------------------

    if parsed["subject_id"]:
        print("[PIPELINE] Route: subject_id lookup")
        df = by_subject_id(parsed["subject_id"])

        context = _join_text(df)
        print("[PIPELINE] Summarizing patient notes...")

        answer = summarize_patient_notes(context)

        state = PipelineState(
            query=query,
            answer=answer,
            retrieved_docs=df_to_docs(df)
        )

        log["route"] = "subject_id_lookup"
        log["rows_returned"] = len(df)
        log["answer_preview"] = answer[:1000]
        state = attach_live_ragas_scores(state, query, df_to_contexts(df))
        log["ragas_scores"] = state.ragas_scores
        log_run(log)

        return state

    if parsed["note_id"]:
        print("[PIPELINE] Route: note_id lookup")
        df = by_note_id(parsed["note_id"])

        context = _join_text(df)
        print("[PIPELINE] Summarizing note...")

        answer = summarize_single_note(context)

        state = PipelineState(
            query=query,
            answer=answer,
            retrieved_docs=df_to_docs(df)
        )

        log["route"] = "note_id_lookup"
        log["rows_returned"] = len(df)
        log["answer_preview"] = answer[:1000]
        state = attach_live_ragas_scores(state, query, df_to_contexts(df))
        log["ragas_scores"] = state.ragas_scores
        log_run(log)

        return state

    if parsed["hadm_id"]:
        print("[PIPELINE] Route: hadm_id lookup")
        df = by_hadm_id(parsed["hadm_id"])

        context = _join_text(df)
        print("[PIPELINE] Summarizing visit...")

        answer = summarize_visit_notes(context)

        state = PipelineState(
            query=query,
            answer=answer,
            retrieved_docs=df_to_docs(df)
        )

        log["route"] = "hadm_id_lookup"
        log["rows_returned"] = len(df)
        log["answer_preview"] = answer[:1000]
        state = attach_live_ragas_scores(state, query, df_to_contexts(df))
        log["ragas_scores"] = state.ragas_scores
        log_run(log)

        return state

    # -------------------------------
    # SINGLE NON-STRUCTURED ROUTE
    # -------------------------------

    print("[PIPELINE] Route: retrieval_search -> graph")

    initial_state = PipelineState(query=query)
    result = graph.invoke(initial_state)
    final_state = PipelineState(**result)

    log["route"] = "retrieval_search_graph"
    log["graph_debug"] = final_state.debug
    log["answer_preview"] = final_state.answer[:1000] if final_state.answer else ""
    if not getattr(final_state, "ragas_scores", None):
        final_state = attach_live_ragas_scores(final_state, query)
    log["ragas_scores"] = final_state.ragas_scores
    log_run(log)

    return final_state
