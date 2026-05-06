"""
QLoRA fine-tuning script for Mistral-7B-Instruct-v0.2.

Trains on domain-specific QA pairs to improve answer quality and
reduce hallucination compared to the base model.

Usage:
    python -m src.generation.finetuning.train \
        --base-model mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset    ./data/qa_pairs.jsonl \
        --output-dir ./checkpoints \
        --lora-rank  64 \
        --epochs     3
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA fine-tune Mistral-7B")
    p.add_argument("--base-model",  default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--dataset",     required=True, help="Path to .jsonl QA dataset")
    p.add_argument("--output-dir",  default="./checkpoints")
    p.add_argument("--lora-rank",   type=int, default=64)
    p.add_argument("--lora-alpha",  type=int, default=128)
    p.add_argument("--lora-dropout",type=float, default=0.05)
    p.add_argument("--epochs",      type=int, default=3)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--grad-accum",  type=int, default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--max-seq-len", type=int, default=2048)
    return p.parse_args()


def load_model_and_tokenizer(base_model: str):
    """Load Mistral-7B with 4-bit NF4 quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def apply_lora(model, rank: int, alpha: int, dropout: float):
    """Attach LoRA adapters to attention + MLP projection layers."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_qa_sample(sample: dict) -> str:
    """Format a QA pair into Mistral instruction format."""
    question = sample.get("question", "")
    context  = sample.get("context", "")
    answer   = sample.get("answer", "")
    return (
        f"[INST] Answer based on the context.\n\nContext: {context}\n\nQuestion: {question} [/INST] {answer}"
    )


def main() -> None:
    args = parse_args()
    logger.info(f"Fine-tuning {args.base_model} with QLoRA (rank={args.lora_rank})")

    model, tokenizer = load_model_and_tokenizer(args.base_model)
    model            = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    logger.info(f"Loaded {len(dataset)} training samples from {args.dataset}")

    output_dir = Path(args.output_dir) / "mistral-7b-qlora"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=format_qa_sample,
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    logger.info("Starting training…")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
