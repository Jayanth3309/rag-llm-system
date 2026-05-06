"""
Prompt templates for RAG generation.
Mistral-7B uses [INST] ... [/INST] instruction format.
"""

from __future__ import annotations

from langchain.schema import Document

SYSTEM_PROMPT = """You are a precise, factual assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer confidently, say "I don't have enough information to answer that."
Never fabricate facts. Always cite the relevant source when possible."""


RAG_TEMPLATE = """[INST] {system_prompt}

Context:
{context}

Question: {question}

Answer concisely and accurately based on the context above. [/INST]"""


def build_rag_prompt(
    question: str,
    context_docs: list[Document],
    max_context_chars: int = 6000,
) -> str:
    """
    Assemble a RAG prompt for Mistral-7B-Instruct.

    Args:
        question:          The user's question.
        context_docs:      Retrieved document chunks.
        max_context_chars: Hard cap on total context length to stay within context window.

    Returns:
        Fully formatted prompt string ready for tokenisation.
    """
    context_parts = []
    total_chars   = 0

    for i, doc in enumerate(context_docs, start=1):
        source = doc.metadata.get("source", "unknown")
        chunk  = doc.metadata.get("chunk_index", "?")
        header = f"[Source {i}: {source}, chunk {chunk}]"
        body   = doc.page_content.strip()
        block  = f"{header}\n{body}"

        if total_chars + len(block) > max_context_chars:
            break

        context_parts.append(block)
        total_chars += len(block)

    context = "\n\n".join(context_parts) if context_parts else "No context available."

    return RAG_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context,
        question=question,
    )
