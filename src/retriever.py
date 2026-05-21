from typing import List, Dict, Any
from src.vectorstore import load_faiss_index
from src.config import config


def retrieve(query: str, k: int | None = None) -> List[Dict[str, Any]]:
    k = k or config.TOP_K

    print("[RETRIEVER] Loading FAISS index...")
    db = load_faiss_index()

    print(f"[RETRIEVER] Query: {query}")
    print(f"[RETRIEVER] similarity_search_with_score k={k}")

    results = db.similarity_search_with_score(query, k=k)

    docs = []
    for doc, score in results:
        docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": float(score),
        })

    print(f"[RETRIEVER] Retrieved docs: {len(docs)}")
    return docs


SECTION_MAP = {
    "history": [
        "Past Medical History",
        "History of Present Illness",
        "Chief Complaint",
        "Mental Status"
    ],
    "medications": [
        "Discharge Medications",
        "Medications on Admission"
    ],
    "results": [
        "Pertinent Results",
        "IMPRESSION"
    ],
    "hospital_course": [
        "Brief Hospital Course"
    ],
    "discharge": [
        "Discharge Condition",
        "Discharge Instructions",
        "Followup Instructions"
    ]
}


def retrieve_by_sections(query: str, groups: list[str], k: int = 20):
    print("[RETRIEVER_S] Loading FAISS index...")
    db = load_faiss_index()

    print(f"[RETRIEVER_S] Query: {query}")
    print(f"[RETRIEVER_S] similarity_search k={k}")

    docs = db.similarity_search(query, k=k)

    allowed = set()
    for g in groups:
        allowed.update(SECTION_MAP.get(g, []))

    if not allowed:
        print(f"[RETRIEVER_S] No section filter applied | Docs: {len(docs)}")
        return docs

    filtered = []
    for d in docs:
        sec = d.metadata.get("section_name", "")
        if sec in allowed:
            filtered.append(d)

    print(f"[RETRIEVER_S] Filtered docs: {len(filtered)} / {len(docs)}")
    return filtered if filtered else docs