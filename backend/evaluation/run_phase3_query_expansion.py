from __future__ import annotations

import csv
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from config import RRF_K
from retrieval import (
    build_retrievers,
    dense_search,
    get_chunk_identity,
    reciprocal_rank_fusion,
    sparse_search,
)
from reranker import rerank_candidates, select_context
from snapshot_manager import load_active_snapshot


GOLD_PATH = Path("evaluation/gold_questions.jsonl")
RESULTS_ROOT = Path("evaluation/results")

DENSE_DEPTH = 100
SPARSE_DEPTH = 100
RRF_DEPTH = 100
CONTEXT_TOP_K = 5
CONTEXT_TOKEN_BUDGET = 5000

EXPERIMENTS = ("original", "expanded")

# Add these terms only to questions that appear to ask about
# an aircraft's departure point, destination, or route.
ROUTE_PATTERN = re.compile(
    r"\b(departure|destination|origin|arrival|"
    r"flying from|fly from|going to|headed to|"
    r"take.?off point|landing point)\b"
    r"|\bwhere\b.*\b(from|to|going)\b",
    re.IGNORECASE,
)

ROUTE_TERMS = (
    "departure point origin airport "
    "destination arrival point destination airport "
    "departure location arrival location"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase3_query_expansion")


def load_questions():
    questions = []

    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    if not questions:
        raise ValueError("Gold question dataset is empty")

    return sorted(questions, key=lambda x: x["question_id"])


def expanded_query(question):
    if ROUTE_PATTERN.search(question):
        return f"{question} {ROUTE_TERMS}"

    return question


def reciprocal_rank_metrics(ranked_ids, gold_ids, k):
    relevant = [
        1 if chunk_id in gold_ids else 0
        for chunk_id in ranked_ids[:k]
    ]

    recall = len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)
    precision = sum(relevant) / k

    dcg = sum(
        rel / __import__("math").log2(index + 2)
        for index, rel in enumerate(relevant)
    )

    ideal_relevant = min(len(gold_ids), k)
    ideal_dcg = sum(
        1 / __import__("math").log2(index + 2)
        for index in range(ideal_relevant)
    )

    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    return recall, precision, ndcg


def calculate_metrics(ranked_ids, gold_ids):
    result = {}

    for k in (5, 10, 20):
        recall, precision, ndcg = reciprocal_rank_metrics(
            ranked_ids, gold_ids, k
        )
        result[f"recall_at_{k}"] = recall
        result[f"precision_at_{k}"] = precision
        result[f"ndcg_at_{k}"] = ndcg

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in gold_ids:
            reciprocal_rank = 1.0 / rank
            break

    result["mrr_at_20"] = reciprocal_rank
    result["hit_at_20"] = float(
        bool(set(ranked_ids[:20]) & gold_ids)
    )

    return result


def main():
    started = time.perf_counter()
    questions = load_questions()

    logger.info("Loading active snapshot")
    snapshot = load_active_snapshot()

    logger.info(
        "Snapshot=%s | chunks=%d | vectors=%s",
        snapshot["snapshot_id"],
        len(snapshot["chunks"]),
        snapshot["vectors"].shape,
    )

    retrievers = build_retrievers(snapshot)
    retrievers["bm25_retriever"].k = SPARSE_DEPTH

    per_question_rows = []
    ranking_rows = []

    for question_data in questions:
        qid = question_data["question_id"]
        original_query = question_data["question"]
        gold_ids = set(question_data["gold_chunk_ids"])

        queries = {
            "original": original_query,
            "expanded": expanded_query(original_query),
        }

        logger.info("Processing %s", qid)

        for strategy in EXPERIMENTS:
            query = queries[strategy]

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=DENSE_DEPTH,
            )

            sparse = sparse_search(
                query=query,
                bm25_retriever=retrievers["bm25_retriever"],
                top_k=SPARSE_DEPTH,
            )

            candidates = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=RRF_DEPTH,
            )

            # Keep the original question for cross-encoder scoring.
            reranked = rerank_candidates(
                query=original_query,
                candidates=candidates,
            )

            selected = select_context(
                reranked_candidates=reranked,
                top_k=CONTEXT_TOP_K,
                max_tokens=CONTEXT_TOKEN_BUDGET,
            )

            ranked_ids = [
                item["chunk_id"] for item in reranked
            ]
            selected_ids = [
                item["chunk_id"] for item in selected
            ]

            metrics = calculate_metrics(ranked_ids, gold_ids)

            row = {
                "question_id": qid,
                "strategy": strategy,
                "query_was_expanded": query != original_query,
                "gold_chunk_count": len(gold_ids),
                "gold_in_context": bool(
                    set(selected_ids) & gold_ids
                ),
                "gold_chunks_in_context": len(
                    set(selected_ids) & gold_ids
                ),
                "context_chunk_count": len(selected_ids),
                **metrics,
            }
            per_question_rows.append(row)

            for rank, item in enumerate(reranked, start=1):
                document = item["document"]
                metadata = document.metadata

                ranking_rows.append({
                    "question_id": qid,
                    "strategy": strategy,
                    "rank": rank,
                    "chunk_id": item["chunk_id"],
                    "is_gold_chunk": item["chunk_id"] in gold_ids,
                    "document": metadata.get(
                        "file_name", metadata.get("source", "")
                    ),
                    "page": metadata.get("page", ""),
                    "rrf_score": item.get("rrf_score", ""),
                    "rerank_score": item.get("rerank_score", ""),
                    "text_preview": document.page_content[:350].replace(
                        "\n", " "
                    ),
                })

            logger.info(
                "%s | %s | expanded=%s | "
                "Recall@5=%.3f | Recall@10=%.3f | "
                "MRR@20=%.3f | context_gold=%s",
                qid,
                strategy,
                query != original_query,
                metrics["recall_at_5"],
                metrics["recall_at_10"],
                metrics["mrr_at_20"],
                bool(set(selected_ids) & gold_ids),
            )

    output_dir = RESULTS_ROOT / (
        "phase3_query_expansion_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    per_question_path = output_dir / "per_question_metrics.csv"
    rankings_path = output_dir / "candidate_rankings.csv"
    summary_path = output_dir / "summary.json"

    with per_question_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(per_question_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(per_question_rows)

    with rankings_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(ranking_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(ranking_rows)

    summary = {}

    for strategy in EXPERIMENTS:
        rows = [
            row for row in per_question_rows
            if row["strategy"] == strategy
        ]

        summary[strategy] = {
            key: round(
                sum(row[key] for row in rows) / len(rows), 4
            )
            for key in (
                "recall_at_5",
                "recall_at_10",
                "recall_at_20",
                "precision_at_5",
                "precision_at_10",
                "precision_at_20",
                "mrr_at_20",
                "ndcg_at_5",
                "ndcg_at_10",
                "ndcg_at_20",
                "hit_at_20",
            )
        }

        summary[strategy]["context_gold_coverage"] = round(
            sum(row["gold_in_context"] for row in rows) / len(rows),
            4,
        )

        summary[strategy]["questions_expanded"] = sum(
            row["query_was_expanded"] for row in rows
        )

    summary["snapshot_id"] = snapshot["snapshot_id"]
    summary["question_count"] = len(questions)
    summary["retrieval_depths"] = {
        "dense": DENSE_DEPTH,
        "sparse": SPARSE_DEPTH,
        "rrf": RRF_DEPTH,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPHASE 3C — QUERY EXPANSION COMPLETE")
    print("Snapshot:", snapshot["snapshot_id"])
    print("Questions:", len(questions))
    print("Questions expanded:", summary["expanded"]["questions_expanded"])
    print()

    metric_names = (
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "precision_at_5",
        "mrr_at_20",
        "ndcg_at_10",
        "context_gold_coverage",
    )

    print(f"{'Metric':<26} {'Original':>12} {'Expanded':>12}")
    print("-" * 52)

    for metric in metric_names:
        print(
            f"{metric:<26} "
            f"{summary['original'][metric]:>12.4f} "
            f"{summary['expanded'][metric]:>12.4f}"
        )

    print("\nPer-question changes:")
    for qid in sorted({row["question_id"] for row in per_question_rows}):
        original = next(
            r for r in per_question_rows
            if r["question_id"] == qid
            and r["strategy"] == "original"
        )
        expanded = next(
            r for r in per_question_rows
            if r["question_id"] == qid
            and r["strategy"] == "expanded"
        )

        if (
            original["gold_in_context"] != expanded["gold_in_context"]
            or original["recall_at_20"] != expanded["recall_at_20"]
        ):
            print(
                qid,
                "| context:",
                original["gold_in_context"],
                "->",
                expanded["gold_in_context"],
                "| Recall@20:",
                round(original["recall_at_20"], 3),
                "->",
                round(expanded["recall_at_20"], 3),
            )

    print("\nRuntime seconds:", round(time.perf_counter() - started, 2))
    print("Results directory:", output_dir)


if __name__ == "__main__":
    main()
