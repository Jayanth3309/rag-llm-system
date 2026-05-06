"""
Document loaders for PDF, HTML, JSON, and plain text sources.
Supports single files and batch directory ingestion.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    BSHTMLLoader,
    TextLoader,
)

logger = logging.getLogger(__name__)


SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".json", ".jsonl"}


def load_document(file_path: str | Path) -> list[Document]:
    """Load a single document from disk. Returns a list of LangChain Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    logger.info(f"Loading {path.name} ({ext})")

    if ext == ".pdf":
        loader = PyPDFLoader(str(path))
        return loader.load()

    if ext in {".html", ".htm"}:
        loader = BSHTMLLoader(str(path))
        return loader.load()

    if ext == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()

    if ext == ".json":
        return _load_json(path)

    if ext == ".jsonl":
        return _load_jsonl(path)

    raise ValueError(f"Handler missing for extension: {ext}")


def _load_json(path: Path) -> list[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [Document(page_content=str(item), metadata={"source": str(path)}) for item in data]
    return [Document(page_content=json.dumps(data), metadata={"source": str(path)})]


def _load_jsonl(path: Path) -> list[Document]:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            content = item.get("text") or item.get("content") or json.dumps(item)
            meta = {k: v for k, v in item.items() if k not in {"text", "content"}}
            meta["source"] = str(path)
            meta["line"] = i
            docs.append(Document(page_content=content, metadata=meta))
    return docs


def iter_directory(
    directory: str | Path,
    recursive: bool = True,
) -> Iterator[Document]:
    """Yield documents from all supported files in a directory."""
    root = Path(directory)
    pattern = "**/*" if recursive else "*"

    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                for doc in load_document(path):
                    yield doc
            except Exception as e:
                logger.warning(f"Skipping {path.name}: {e}")


def load_directory(
    directory: str | Path,
    recursive: bool = True,
) -> list[Document]:
    """Load all documents from a directory into a flat list."""
    docs = list(iter_directory(directory, recursive=recursive))
    logger.info(f"Loaded {len(docs)} documents from {directory}")
    return docs
