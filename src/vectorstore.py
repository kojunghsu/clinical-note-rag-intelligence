import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from src.embeddings import get_embedding_model
from src.config import config


def save_faiss_index(documents: List[Document]) -> None:
    embeddings = get_embedding_model()

    print("[VS] Creating vector index...")
    db = FAISS.from_documents(documents, embeddings)

    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)

    print("[VS] Saving index...")
    db.save_local(config.FAISS_INDEX_DIR)


def load_faiss_index() -> FAISS:
    embeddings = get_embedding_model()

    print("[VS] Loading existing index...")
    return FAISS.load_local(
        config.FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )