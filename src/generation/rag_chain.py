"""
End-to-end RAG chain: retrieval → prompt construction → LLM generation.
Supports both fine-tuned Mistral-7B (local) and OpenAI (fallback).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain.schema import Document

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_templates import build_rag_prompt

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer:         str
    source_docs:    list[Document]
    retrieval_ms:   float
    generation_ms:  float
    total_ms:       float
    model:          str
    tokens_used:    int = 0
    metadata:       dict = field(default_factory=dict)


class RAGChain:
    """
    Orchestrates the full RAG pipeline:
      query → hybrid retrieval → prompt assembly → LLM → RAGResponse

    Supports two backends:
      - "local":  fine-tuned Mistral-7B loaded via transformers (GPU required)
      - "openai": OpenAI GPT-4o (useful for baseline comparison / evaluation)
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        backend: str = "local",
        model_path: str = "./checkpoints/mistral-7b-qlora",
        top_k: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> None:
        self.retriever      = retriever
        self.backend        = backend
        self.top_k          = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature    = temperature
        self._llm           = self._load_llm(backend, model_path)

    # ── Public API ───────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline and return a structured response."""
        t0 = time.perf_counter()

        # 1. Retrieve
        t_ret = time.perf_counter()
        docs  = self.retriever.retrieve(question, top_k=self.top_k)
        retrieval_ms = (time.perf_counter() - t_ret) * 1000

        # 2. Build prompt
        prompt = build_rag_prompt(question=question, context_docs=docs)

        # 3. Generate
        t_gen = time.perf_counter()
        answer, tokens = self._generate(prompt)
        generation_ms = (time.perf_counter() - t_gen) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"RAG query complete | retrieval={retrieval_ms:.0f}ms "
            f"generation={generation_ms:.0f}ms total={total_ms:.0f}ms"
        )

        return RAGResponse(
            answer=answer,
            source_docs=docs,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            model=self.backend,
            tokens_used=tokens,
        )

    # ── LLM backends ─────────────────────────────────────────────

    def _load_llm(self, backend: str, model_path: str):
        if backend == "local":
            return self._load_local_model(model_path)
        if backend == "openai":
            return self._load_openai()
        raise ValueError(f"Unknown backend: {backend}. Choose 'local' or 'openai'.")

    def _load_local_model(self, model_path: str):
        """Load fine-tuned Mistral-7B with 4-bit quantization."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import PeftModel

            logger.info(f"Loading local model from {model_path}…")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            base_model = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer  = AutoTokenizer.from_pretrained(base_model)
            model      = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, model_path)
            model.eval()
            return {"model": model, "tokenizer": tokenizer, "type": "local"}

        except ImportError as e:
            raise ImportError(
                "transformers, peft, and bitsandbytes are required for local inference. "
                "Install with: pip install transformers peft bitsandbytes"
            ) from e

    def _load_openai(self):
        from langchain_openai import ChatOpenAI
        logger.info("Using OpenAI backend (gpt-4o)")
        return {"client": ChatOpenAI(model="gpt-4o", temperature=self.temperature), "type": "openai"}

    def _generate(self, prompt: str) -> tuple[str, int]:
        """Generate an answer; returns (answer_text, token_count)."""
        if self._llm["type"] == "local":
            return self._generate_local(prompt)
        return self._generate_openai(prompt)

    def _generate_local(self, prompt: str) -> tuple[str, int]:
        import torch
        model     = self._llm["model"]
        tokenizer = self._llm["tokenizer"]

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated  = output_ids[0][inputs["input_ids"].shape[1]:]
        answer     = tokenizer.decode(generated, skip_special_tokens=True).strip()
        tokens     = int(generated.shape[0])
        return answer, tokens

    def _generate_openai(self, prompt: str) -> tuple[str, int]:
        from langchain_core.messages import HumanMessage
        client   = self._llm["client"]
        response = client.invoke([HumanMessage(content=prompt)])
        return response.content.strip(), response.usage_metadata.get("total_tokens", 0)
