"""
RAGAS evaluation pipeline.
Measures: faithfulness, answer_relevancy, context_recall, context_precision.

Usage:
    python -m src.evaluation.ragas_eval \
        --testset ./data/testset.json \
        --output  ./results/ragas_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--testset", required=True, help="JSON testset: list of {question, ground_truth, contexts, answer}")
    p.add_argument("--output",  default="./results/ragas_report.json")
    return p.parse_args()


def load_testset(path: str) -> Dataset:
    with open(path) as f:
        data = json.load(f)

    # Validate expected keys
    required = {"question", "ground_truth", "contexts", "answer"}
    for i, item in enumerate(data):
        missing = required - item.keys()
        if missing:
            raise ValueError(f"Sample {i} missing keys: {missing}")

    return Dataset.from_list(data)


def run_evaluation(dataset: Dataset) -> dict:
    """Run RAGAS evaluation on the dataset and return a metrics dict."""
    logger.info(f"Running RAGAS evaluation on {len(dataset)} samples…")

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores = {
        "faithfulness":        round(float(result["faithfulness"]),        4),
        "answer_relevancy":    round(float(result["answer_relevancy"]),    4),
        "context_recall":      round(float(result["context_recall"]),      4),
        "context_precision":   round(float(result["context_precision"]),   4),
    }
    scores["mean_score"] = round(sum(scores.values()) / len(scores), 4)
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = load_testset(args.testset)
    scores  = run_evaluation(dataset)

    print("\n── RAGAS Evaluation Results ────────────────")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<25} {score:.4f}  {bar}")
    print("────────────────────────────────────────────\n")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"scores": scores, "n_samples": len(dataset)}, f, indent=2)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
