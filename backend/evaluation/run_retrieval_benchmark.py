"""
CortexRAG retrieval baseline benchmark.

Run from backend:
    python evaluation/run_retrieval_benchmark.py

This script:
- Loads the active snapshot and validated gold_questions.jsonl.
- Compares dense-only, BM25-only, hybrid RRF, and hybrid RRF + CrossEncoder.
- Evaluates exact gold chunk-ID matches at K=5, 10, and 20.
- Does not call an LLM or any generation API.
- Does not modify production configuration or retrieval files.

Metric convention:
- Precision@K uses K as the denominator, even if fewer than K results exist.
- Recall@K divides relevant retrieved chunks by the number of gold chunks.
- nDCG uses binary relevance.
- MRR is measured over the first 20 retrieved results.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure imports work when this file is run from the backend directory
# or directly by its absolute path.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from retrieval import (  # noqa: E402
    build_retrievers,
    dense_search,
    get_chunk_identity,
    reciprocal_rank_fusion,
    sparse_search,
)
from reranker import rerank_candidates, select_context  # noqa: E402
from snapshot_manager import load_active_snapshot  # noqa: E402
from config import RRF_K  # noqa: E402


EVALUATION_DIR = BACKEND_DIR / "evaluation"
GOLD_DATASET = EVALUATION_DIR / "gold_questions.jsonl"
RESULTS_ROOT = EVALUATION_DIR / "results"
K_VALUES = (5, 10, 20)
MAX_K = max(K_VALUES)
CONTEXT_TOP_K = 5
CONTEXT_TOKEN_BUDGET = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("cortexrag.evaluation")


def load_gold_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Gold dataset not found: {path}")

    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on gold dataset line {line_number}: {exc}"
                ) from exc

            required = {
                "question_id",
                "question",
                "gold_chunk_ids",
            }
            missing = required - set(item)
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing fields: {sorted(missing)}"
                )

            gold_ids = item["gold_chunk_ids"]
            if not isinstance(gold_ids, list) or not gold_ids:
                raise ValueError(
                    f"{item['question_id']} has no gold_chunk_ids"
                )

            item["gold_chunk_ids"] = [str(value) for value in gold_ids]
            questions.append(item)

    if not questions:
        raise ValueError("Gold dataset is empty")

    ids = [item["question_id"] for item in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id values found")

    return questions


def ranked_chunk_ids(results: list[Any]) -> list[str]:
    """Convert Document or RRF/reranker candidate results to chunk IDs."""
    ids: list[str] = []
    seen: set[str] = set()

    for result in results:
        if isinstance(result, dict):
            chunk_id = result.get("chunk_id")
            if not chunk_id and result.get("document") is not None:
                chunk_id = get_chunk_identity(result["document"])
        else:
            chunk_id = get_chunk_identity(result)

        if chunk_id is None:
            continue

        chunk_id = str(chunk_id)
        if chunk_id not in seen:
            ids.append(chunk_id)
            seen.add(chunk_id)

    return ids


def calculate_metrics(
    ranked_ids: list[str],
    gold_ids: set[str],
) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {}

    for k in K_VALUES:
        top_ids = ranked_ids[:k]
        relevant_count = sum(1 for chunk_id in top_ids if chunk_id in gold_ids)

        metrics[f"recall@{k}"] = relevant_count / len(gold_ids)
        metrics[f"precision@{k}"] = relevant_count / k

        dcg = sum(
            1.0 / __import__("math").log2(rank + 1)
            for rank, chunk_id in enumerate(top_ids, start=1)
            if chunk_id in gold_ids
        )
        ideal_relevant = min(len(gold_ids), k)
        idcg = sum(
            1.0 / __import__("math").log2(rank + 1)
            for rank in range(1, ideal_relevant + 1)
        )
        metrics[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0

    first_relevant_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(ranked_ids[:MAX_K], start=1)
            if chunk_id in gold_ids
        ),
        None,
    )
    metrics["first_relevant_rank"] = first_relevant_rank
    metrics["mrr@20"] = (
        1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
    )
    metrics["hit@20"] = 1 if first_relevant_rank is not None else 0

    return metrics


def evaluate_rankings(
    questions: list[dict[str, Any]],
    rankings: dict[str, dict[str, list[str]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_question_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []

    metric_names = [
        *(f"recall@{k}" for k in K_VALUES),
        *(f"precision@{k}" for k in K_VALUES),
        *(f"ndcg@{k}" for k in K_VALUES),
        "mrr@20",
        "hit@20",
    ]

    for strategy, question_rankings in rankings.items():
        metric_values: dict[str, list[float]] = defaultdict(list)

        for question in questions:
            question_id = question["question_id"]
            gold_ids = set(question["gold_chunk_ids"])
            ranked_ids = question_rankings[question_id]
            metrics = calculate_metrics(ranked_ids, gold_ids)

            row = {
                "question_id": question_id,
                "category": question.get("category", ""),
                "question": question["question"],
                "strategy": strategy,
                "gold_chunk_count": len(gold_ids),
                "retrieved_count": len(ranked_ids),
                "first_relevant_rank": metrics["first_relevant_rank"],
                "retrieved_chunk_ids_top20": json.dumps(ranked_ids[:MAX_K]),
                "gold_chunk_ids": json.dumps(sorted(gold_ids)),
            }
            for metric_name in metric_names:
                value = float(metrics[metric_name])
                row[metric_name] = value
                metric_values[metric_name].append(value)

            per_question_rows.append(row)

        aggregate = {
            "strategy": strategy,
            "question_count": len(questions),
        }
        for metric_name in metric_names:
            values = metric_values[metric_name]
            aggregate[metric_name] = statistics.mean(values) if values else 0.0

        aggregate_rows.append(aggregate)

    return per_question_rows, aggregate_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    path: Path,
    snapshot_id: str,
    question_count: int,
    aggregate_rows: list[dict[str, Any]],
    context_metrics: dict[str, float],
    elapsed_seconds: float,
) -> None:
    columns = [
        "strategy",
        "recall@5",
        "recall@10",
        "recall@20",
        "precision@5",
        "precision@10",
        "precision@20",
        "mrr@20",
        "ndcg@10",
        "ndcg@20",
    ]

    lines = [
        "# CortexRAG Retrieval Baseline",
        "",
        f"- Snapshot: `{snapshot_id}`",
        f"- Questions: {question_count}",
        f"- Evaluated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Runtime: {elapsed_seconds:.2f} seconds",
        "- Gold relevance: exact `chunk_id` match",
        "- Precision denominator: fixed K",
        "- LLM/API generation calls: none",
        "",
        "## Aggregate metrics (macro average across questions)",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for row in aggregate_rows:
        values = [row["strategy"]]
        values.extend(f"{float(row[column]):.4f}" for column in columns[1:])
        lines.append("| " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Production top-five context selection",
        "",
        f"- Questions evaluated: {question_count}",
        f"- Mean Recall@5 after CrossEncoder context selection: "
        f"{context_metrics.get('recall@5', 0.0):.4f}",
        f"- Mean Precision@5 after CrossEncoder context selection: "
        f"{context_metrics.get('precision@5', 0.0):.4f}",
        f"- Mean nDCG@5 after CrossEncoder context selection: "
        f"{context_metrics.get('ndcg@5', 0.0):.4f}",
        "",
        "## Interpretation notes",
        "",
        "- Results measure retrieval of manually labelled gold chunks, not answer quality.",
        "- A semantically equivalent neighbouring chunk is counted as non-relevant unless it is also labelled.",
        "- MRR@20 is the mean reciprocal rank of the first gold chunk within the first 20 results.",
        "- The reranked strategy uses the same RRF candidate pool as the hybrid strategy.",
        "- Top-five context metrics include the production duplicate-removal and token-budget selection logic.",
        "",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    start_time = time.perf_counter()

    logger.info("Loading validated gold dataset: %s", GOLD_DATASET)
    questions = load_gold_dataset(GOLD_DATASET)
    logger.info("Loaded %d gold questions", len(questions))

    logger.info("Loading active snapshot and local embedding model")
    snapshot = load_active_snapshot()
    snapshot_id = snapshot["snapshot_id"]

    if len(snapshot["chunks"]) != len(snapshot["vectors"]):
        raise ValueError("Snapshot chunk/vector count mismatch")

    logger.info(
        "Snapshot %s: %d chunks, vectors=%s",
        snapshot_id,
        len(snapshot["chunks"]),
        snapshot["vectors"].shape,
    )

    # Build BM25 once. Raise its output count to 20 for this benchmark only.
    logger.info("Building BM25 retriever")
    retrievers = build_retrievers(snapshot)
    retrievers["bm25_retriever"].k = MAX_K

    # Check all gold IDs exist in the loaded active snapshot before scoring.
    snapshot_ids = {
        get_chunk_identity(chunk)
        for chunk in snapshot["chunks"]
    }
    missing_gold = sorted({
        gold_id
        for question in questions
        for gold_id in question["gold_chunk_ids"]
        if gold_id not in snapshot_ids
    })
    if missing_gold:
        raise ValueError(
            f"{len(missing_gold)} gold chunk IDs are absent from the active "
            f"snapshot. First missing IDs: {missing_gold[:10]}"
        )

    rankings: dict[str, dict[str, list[str]]] = {
        "dense_only": {},
        "bm25_only": {},
        "hybrid_rrf": {},
        "hybrid_rrf_cross_encoder": {},
    }
    context_rows: list[dict[str, Any]] = []

    # The CrossEncoder is loaded lazily by rerank_candidates on first use.
    for index, question in enumerate(questions, start=1):
        question_id = question["question_id"]
        query = question["question"]

        logger.info(
            "[%d/%d] Evaluating %s",
            index,
            len(questions),
            question_id,
        )

        dense_results = dense_search(
            query=query,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=MAX_K,
        )
        sparse_results = sparse_search(
            query=query,
            bm25_retriever=retrievers["bm25_retriever"],
            top_k=MAX_K,
        )

        hybrid_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=RRF_K,
            top_k=MAX_K,
        )

        reranked = rerank_candidates(
            query=query,
            candidates=hybrid_candidates,
        )

        rankings["dense_only"][question_id] = ranked_chunk_ids(dense_results)
        rankings["bm25_only"][question_id] = ranked_chunk_ids(sparse_results)
        rankings["hybrid_rrf"][question_id] = ranked_chunk_ids(hybrid_candidates)
        rankings["hybrid_rrf_cross_encoder"][question_id] = ranked_chunk_ids(reranked)

        # Measure the actual production top-five context selection separately.
        selected_context = select_context(
            reranked_candidates=reranked,
            top_k=CONTEXT_TOP_K,
            max_tokens=CONTEXT_TOKEN_BUDGET,
        )
        selected_ids = ranked_chunk_ids(selected_context)
        context_rows.append({
            "question_id": question_id,
            "ranked_chunk_ids": selected_ids,
        })

        logger.info(
            "%s complete: dense=%d, BM25=%d, RRF=%d, reranked=%d, context=%d",
            question_id,
            len(dense_results),
            len(sparse_results),
            len(hybrid_candidates),
            len(reranked),
            len(selected_context),
        )

    per_question_rows, aggregate_rows = evaluate_rankings(
        questions=questions,
        rankings=rankings,
    )

    context_metric_values: dict[str, list[float]] = defaultdict(list)
    for question in questions:
        selected_ids = next(
            row["ranked_chunk_ids"]
            for row in context_rows
            if row["question_id"] == question["question_id"]
        )
        metrics = calculate_metrics(
            selected_ids,
            set(question["gold_chunk_ids"]),
        )
        for name in ("recall@5", "precision@5", "ndcg@5"):
            context_metric_values[name].append(float(metrics[name]))

    context_metrics = {
        name: statistics.mean(values) if values else 0.0
        for name, values in context_metric_values.items()
    }

    elapsed_seconds = time.perf_counter() - start_time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_ROOT / f"baseline_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    write_csv(output_dir / "per_question_metrics.csv", per_question_rows)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output_dir / "production_context_top5.csv", context_rows)

    result_json = {
        "snapshot_id": snapshot_id,
        "question_count": len(questions),
        "gold_chunk_reference_count": sum(
            len(question["gold_chunk_ids"]) for question in questions
        ),
        "k_values": list(K_VALUES),
        "metric_conventions": {
            "relevance": "exact gold chunk_id match",
            "precision_denominator": "fixed K",
            "mrr": "MRR@20",
            "aggregation": "macro average across questions",
        },
        "generation_api_calls": 0,
        "elapsed_seconds": elapsed_seconds,
        "aggregate_metrics": aggregate_rows,
        "production_context_top5": context_metrics,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    write_markdown_report(
        output_dir / "baseline_report.md",
        snapshot_id=snapshot_id,
        question_count=len(questions),
        aggregate_rows=aggregate_rows,
        context_metrics=context_metrics,
        elapsed_seconds=elapsed_seconds,
    )

    print("\n" + "=" * 72)
    print("CORTEXRAG RETRIEVAL BASELINE COMPLETE")
    print("=" * 72)
    print(f"Snapshot: {snapshot_id}")
    print(f"Questions: {len(questions)}")
    print(f"Gold chunk references: {result_json['gold_chunk_reference_count']}")
    print(f"Runtime: {elapsed_seconds:.2f} seconds")
    print(f"Results directory: {output_dir}")
    print("\nAggregate metrics:")
    for row in aggregate_rows:
        print(f"\n{row['strategy']}")
        for name in (
            "recall@5", "recall@10", "recall@20",
            "precision@5", "precision@10", "precision@20",
            "mrr@20", "ndcg@5", "ndcg@10", "ndcg@20",
        ):
            print(f"  {name:14s}: {row[name]:.4f}")

    print("\nProduction top-five context selection:")
    for name, value in context_metrics.items():
        print(f"  {name:14s}: {value:.4f}")

    print("\nGenerated files:")
    for filename in (
        "baseline_report.md",
        "aggregate_metrics.csv",
        "per_question_metrics.csv",
        "production_context_top5.csv",
        "summary.json",
    ):
        print(f"  - {output_dir / filename}")


if __name__ == "__main__":
    main()
