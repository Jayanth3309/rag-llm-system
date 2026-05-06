"""
FastAPI application entry point.
Exposes /query and /ingest endpoints with Redis caching and rate limiting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.ingestion.document_loader import load_directory
from src.ingestion.chunker import chunk_documents, ChunkStrategy
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.rag_chain import RAGChain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ─────────────────────────────────────────────────
_rag_chain: RAGChain | None = None
_redis:     aioredis.Redis  | None = None

CACHE_TTL_SECONDS = 3600
INDEX_DIR         = os.getenv("INDEX_DIR", "./data/index")
DOCS_DIR          = os.getenv("DOCS_DIR",  "./data/docs")
REDIS_URL         = os.getenv("REDIS_URL", "redis://localhost:6379")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_chain, _redis

    logger.info("Loading RAG system…")
    faiss = FAISSRetriever.load(INDEX_DIR)
    bm25  = BM25Retriever.load(f"{INDEX_DIR}/bm25.pkl")
    hybrid = HybridRetriever(faiss, bm25)
    _rag_chain = RAGChain(
        retriever=hybrid,
        backend=os.getenv("LLM_BACKEND", "local"),
        model_path=os.getenv("MODEL_PATH", "./checkpoints/mistral-7b-qlora"),
    )

    _redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    logger.info("RAG system ready.")
    yield

    await _redis.aclose()


app = FastAPI(
    title="Domain-Adaptive RAG API",
    description="Production RAG system with Mistral-7B fine-tuning + hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k:    int = Field(default=5, ge=1, le=20)

class SourceDoc(BaseModel):
    content: str
    source:  str
    chunk:   int

class QueryResponse(BaseModel):
    answer:        str
    sources:       list[SourceDoc]
    retrieval_ms:  float
    generation_ms: float
    total_ms:      float
    cached:        bool

class IngestRequest(BaseModel):
    directory: str = Field(default=DOCS_DIR)
    strategy:  str = Field(default="recursive")

class IngestResponse(BaseModel):
    documents_loaded: int
    chunks_created:   int
    message:          str


# ── Endpoints ────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _rag_chain is not None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _rag_chain is None:
        raise HTTPException(503, "RAG system not initialized")

    # Check Redis cache
    cache_key = "rag:" + hashlib.sha256(f"{req.question}:{req.top_k}".encode()).hexdigest()
    cached_val = await _redis.get(cache_key)
    if cached_val:
        data = json.loads(cached_val)
        data["cached"] = True
        return QueryResponse(**data)

    # Run RAG pipeline
    result = _rag_chain.query(req.question)

    sources = [
        SourceDoc(
            content=doc.page_content[:300],
            source=doc.metadata.get("source", "unknown"),
            chunk=doc.metadata.get("chunk_index", 0),
        )
        for doc in result.source_docs
    ]

    response_data = {
        "answer":        result.answer,
        "sources":       [s.model_dump() for s in sources],
        "retrieval_ms":  round(result.retrieval_ms, 1),
        "generation_ms": round(result.generation_ms, 1),
        "total_ms":      round(result.total_ms, 1),
        "cached":        False,
    }

    # Cache for 1 hour
    await _redis.setex(cache_key, CACHE_TTL_SECONDS, json.dumps(response_data))

    return QueryResponse(**response_data)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    """Load, chunk, and index documents from a directory."""
    try:
        docs   = load_directory(req.directory)
        chunks = chunk_documents(docs, strategy=ChunkStrategy(req.strategy))

        faiss = FAISSRetriever()
        faiss.build_index(chunks)
        faiss.save(INDEX_DIR)

        bm25 = BM25Retriever()
        bm25.build_index(chunks)
        bm25.save(f"{INDEX_DIR}/bm25.pkl")

        return IngestResponse(
            documents_loaded=len(docs),
            chunks_created=len(chunks),
            message="Index built and saved successfully.",
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(500, str(e))
