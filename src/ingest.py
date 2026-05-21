"""
ingest.py — Data ingestion, chunking, and FAISS index construction.

Usage (one-time setup):
    python3 -m src.ingest
"""

import os
import pickle

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.data_sources.csv_loader import CSVLoader
from src.embeddings import get_embedding_model


def chunk_documents(documents):
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


def build_index():
    """
    Load discharge notes from DATA_PATH, chunk them, embed with OpenAI,
    and save a FAISS index to FAISS_INDEX_DIR.

    Supports incremental checkpointing: if interrupted, re-running resumes
    from the last saved batch rather than starting over.
    """
    if not config.DATA_PATH or not os.path.exists(config.DATA_PATH):
        raise FileNotFoundError(
            f"DATA_PATH not found: {config.DATA_PATH!r}\n"
            "Set DATA_PATH in your .env file to point to discharge.csv."
        )

    loader = CSVLoader(
        path=config.DATA_PATH,
        nrows=config.MAX_ROWS_LOAD,
        filters={},
    )

    print("[INGEST] Reading CSV...")
    docs = loader.load()
    print(f"[INGEST] Loaded {len(docs)} rows from {config.DATA_PATH}")

    print("[INGEST] Chunking documents...")
    chunks = chunk_documents(docs)
    print(f"[INGEST] Total chunks: {len(chunks)}")

    embedding_model = get_embedding_model()
    print(f"[INGEST] Embedding model: {embedding_model.model}")

    # ── Checkpointing: resume from last saved batch ──────────────────────────
    checkpoint_path = config.CHECKPOINT_PATH
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            saved = pickle.load(f)
        embedded_chunks = saved["chunks"]
        start_idx = saved["idx"]
        print(f"[INGEST] Resuming from chunk {start_idx} (checkpoint found)")
    else:
        embedded_chunks = []
        start_idx = 0

    batch_size = 50
    for i in range(start_idx, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        metadatas = [c.metadata for c in batch]
        embeddings = embedding_model.embed_documents(texts)

        for text, meta, emb in zip(texts, metadatas, embeddings):
            embedded_chunks.append((text, meta, emb))

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"chunks": embedded_chunks, "idx": i + batch_size}, f)

        print(f"[INGEST] Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    # ── Build and persist FAISS index ────────────────────────────────────────
    print("[INGEST] Building FAISS index...")
    texts = [x[0] for x in embedded_chunks]
    metas = [x[1] for x in embedded_chunks]
    embs  = [x[2] for x in embedded_chunks]

    db = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embs)),
        embedding=embedding_model,
        metadatas=metas,
        distance_strategy=DistanceStrategy.COSINE,
    )

    os.makedirs(config.FAISS_INDEX_DIR, exist_ok=True)
    db.save_local(config.FAISS_INDEX_DIR)

    # Clean up checkpoint after successful build
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(f"[INGEST] Index saved to {config.FAISS_INDEX_DIR}")

if __name__ == "__main__":
    build_index()
