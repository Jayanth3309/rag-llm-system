"""
Sparse BM25 retriever using rank_bm25.
Complements FAISS dense search in the hybrid pipeline.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

from langchain.schema import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokeniser (lowercase)."""
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """
    Keyword-based retriever using BM25Okapi scoring.

    Pairs with FAISSRetriever in the HybridRetriever for complementary signals:
    dense search excels at semantic similarity; BM25 excels at exact keyword matches.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._documents: list[Document] = []

    # ── Index construction ──────────────────────────────────────

    def build_index(self, documents: list[Document]) -> None:
        """Tokenise documents and build a BM25 index."""
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        corpus = [_tokenize(doc.page_content) for doc in documents]
        self._bm25     = BM25Okapi(corpus)
        self._documents = documents
        logger.info(f"BM25 index built over {len(documents)} documents.")

    # ── Retrieval ────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 10) -> list[tuple[Document, float]]:
        """Return (document, bm25_score) pairs for the top-k keyword matches."""
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair with documents and sort descending
        scored = sorted(
            zip(self._documents, scores.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return scored[:top_k]

    # ── Persistence ──────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self._bm25, "documents": self._documents}, f)
        logger.info(f"BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        instance = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance._bm25      = data["bm25"]
        instance._documents = data["documents"]
        logger.info(f"BM25 index loaded from {path}: {len(instance._documents)} docs")
        return instance
