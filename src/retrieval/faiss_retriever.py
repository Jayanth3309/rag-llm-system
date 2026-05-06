"""
Dense vector retriever backed by FAISS.
Supports building an index from documents and similarity search.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TOP_K           = 10


class FAISSRetriever:
    """
    Wraps a FAISS flat L2 index for dense semantic retrieval.

    Usage:
        retriever = FAISSRetriever()
        retriever.build_index(chunks)
        results = retriever.retrieve("What is the return policy?", top_k=5)
    """

    def __init__(self, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.embedding_model_name = embedding_model
        self._embedder = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
        self._index: faiss.Index | None = None
        self._documents: list[Document] = []

    # ── Index construction ──────────────────────────────────────

    def build_index(self, documents: list[Document]) -> None:
        """Embed all documents and build a FAISS flat L2 index."""
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        logger.info(f"Embedding {len(documents)} chunks…")
        texts      = [doc.page_content for doc in documents]
        embeddings = np.array(self._embedder.embed_documents(texts), dtype=np.float32)

        dim          = embeddings.shape[1]
        self._index  = faiss.IndexFlatIP(dim)   # inner product ≡ cosine (normalised)
        self._index.add(embeddings)
        self._documents = documents

        logger.info(f"FAISS index built: {self._index.ntotal} vectors, dim={dim}")

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[tuple[Document, float]]:
        """
        Return (document, score) pairs for the top-k most similar chunks.
        Scores are cosine similarities in [0, 1].
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = np.array(
            [self._embedder.embed_query(query)], dtype=np.float32
        )
        scores, indices = self._index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self._documents[idx], float(score)))

        return results

    # ── Persistence ──────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Save FAISS index and document store to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "faiss.index"))
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)
        logger.info(f"FAISS index saved to {path}")

    @classmethod
    def load(cls, directory: str | Path, embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> "FAISSRetriever":
        """Load a previously saved FAISS index from disk."""
        path     = Path(directory)
        instance = cls(embedding_model=embedding_model)
        instance._index = faiss.read_index(str(path / "faiss.index"))
        with open(path / "documents.pkl", "rb") as f:
            instance._documents = pickle.load(f)
        logger.info(f"FAISS index loaded from {path}: {instance._index.ntotal} vectors")
        return instance
