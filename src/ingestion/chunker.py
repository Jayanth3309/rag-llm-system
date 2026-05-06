"""
Text chunking strategies for RAG ingestion.

Supports:
- RecursiveCharacterTextSplitter  (default, fast)
- SemanticChunker                 (embedding-aware, higher quality)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    SEMANTIC  = "semantic"


DEFAULT_CHUNK_SIZE    = 512
DEFAULT_CHUNK_OVERLAP = 64


def chunk_documents(
    documents: Sequence[Document],
    strategy: ChunkStrategy = ChunkStrategy.RECURSIVE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
) -> list[Document]:
    """
    Split a list of LangChain Documents into chunks.

    Args:
        documents:       Source documents to chunk.
        strategy:        'recursive' (fast) or 'semantic' (embedding-aware).
        chunk_size:      Target chunk size in characters (recursive only).
        chunk_overlap:   Overlap between adjacent chunks (recursive only).
        embedding_model: HuggingFace model used for semantic chunking.

    Returns:
        List of chunked Document objects with updated metadata.
    """
    if not documents:
        return []

    if strategy == ChunkStrategy.RECURSIVE:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
        chunks = splitter.split_documents(list(documents))

    elif strategy == ChunkStrategy.SEMANTIC:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        chunks = splitter.split_documents(list(documents))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Tag each chunk with its position within the source document
    _add_chunk_metadata(chunks)

    logger.info(
        f"Chunked {len(documents)} docs → {len(chunks)} chunks "
        f"(strategy={strategy}, size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def _add_chunk_metadata(chunks: list[Document]) -> None:
    """Annotate each chunk with its index within its source document."""
    source_counter: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        idx = source_counter.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_size"]  = len(chunk.page_content)
        source_counter[src] = idx + 1
