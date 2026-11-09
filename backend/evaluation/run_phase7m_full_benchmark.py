import csv
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
    RERANK_CANDIDATES,
)
from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
    get_chunk_identity,
)
from reranker import get_reranker


EVALUATION_DIR = BACKEND / "evaluation"
GOLD_FILE = EVALUATION_DIR / "gold_questions.jsonl"
RESULTS_DIR = EVALUATION_DIR / "results"


def load_questions():
    questions = []

    with GOLD_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            required = {
                "question_id",
                "question",
                "gold_chunk_ids",
            }

            missing = required - set(item)
            if missing:
                raise ValueError(
                    f"Line {line_number}: missing {missing}"
                )

            if not item["gold_chunk_ids"]:
                raise ValueError(
                    f"{item['question_id']} has no gold chunk IDs"
                )

            questions.append(item)

    ids = [q["question_id"] for q in questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs found")

    if len(questions) != 40:
        raise ValueError(
            f"Expected exactly 40 questions, found {len(questions)}"
        )

    return questions


def ranked_chunk_ids(results):
    output = []
    seen = set()

    for item in results:
        if isinstance(item, dict):
            document = item.get("document")
            chunk_id = item.get("chunk_id")

            if not chunk_id and document is not None:
                chunk_id = get_chunk_identity(document)
        else:
            chunk_id = get_chunk_identity(item)

        if chunk_id is None:
            continue

        chunk_id = str(chunk_id)

        if chunk_id not in seen:
            output.append(chunk_id)
            seen.add(chunk_id)

    return output


def calculate_metrics(ranked_ids, gold_ids):
    gold_ids = set(gold_ids)

    def recall_at(k):
        return (
            sum(x in gold_ids for x in ranked_ids[:k])
            / len(gold_ids)
        )

    def precision_at(k):
        return sum(
            x in gold_ids for x in ranked_ids[:k]
        ) / k

    def ndcg_at(k):
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, chunk_id in enumerate(
                ranked_ids[:k], start=1
            )
            if chunk_id in gold_ids
        )

        ideal_count = min(len(gold_ids), k)

        idcg = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )

        return dcg / idcg if idcg else 0.0

    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(
                ranked_ids[:20], start=1
            )
            if chunk_id in gold_ids
        ),
        None,
    )

    return {
        "recall_at_5": recall_at(5),
        "recall_at_10": recall_at(10),
        "recall_at_20": recall_at(20),
        "precision_at_5": precision_at(5),
        "ndcg_at_10": ndcg_at(10),
        "mrr_at_20": (
            1 / first_rank if first_rank is not None else 0.0
        ),
        "hit_at_20": int(first_rank is not None),
        "first_gold_rank": first_rank,
        "gold_ranks": [
            (
                ranked_ids.index(gold_id) + 1
                if gold_id in ranked_ids
                else None
            )
            for gold_id in gold_ids
        ],
    }


def rerank(model, query, candidates, use_metadata):
    pairs = []

    for item in candidates:
        document = item["document"]

        if use_metadata:
            filename = document.metadata.get(
                "file_name", "Unknown"
            )
            page = document.metadata.get("page", "Unknown")

            content = (
                f"Source document: {filename}\n"
                f"Page: {page}\n"
                f"Content:\n{document.page_content}"
            )
        else:
            content = document.page_content

        pairs.append((query, content))

    scores = model.predict(
        pairs,
        show_progress_bar=False,
        batch_size=16,
    )

    order = np.argsort(-np.asarray(scores))

    return [candidates[int(i)] for i in order]


def main():
    started = time.time()

    questions = load_questions()

    print("=" * 65)
    print("PHASE 7M — FULL 40-QUESTION RETRIEVAL BENCHMARK")
    print("=" * 65)

    print("\nLoading active snapshot...")
    snapshot = load_active_snapshot()

    snapshot_id = snapshot.get("snapshot_id", "unknown")

    print("Snapshot:", snapshot_id)

    retrievers = build_retrievers(snapshot)
    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    # Verify all gold chunk IDs exist in the active snapshot.
    indexed_ids = {
        str(get_chunk_identity(document))
        for document in retrievers["chunks"]
        if get_chunk_identity(document) is not None
    }

    all_gold_ids = {
        str(chunk_id)
        for question in questions
        for chunk_id in question["gold_chunk_ids"]
    }

    missing_gold_ids = all_gold_ids - indexed_ids

    print("Questions:", len(questions))
    print("Unique gold chunk IDs:", len(all_gold_ids))
    print("Gold IDs present in snapshot:", len(all_gold_ids & indexed_ids))
    print("Gold IDs missing from snapshot:", len(missing_gold_ids))

    if missing_gold_ids:
        raise RuntimeError(
            "Some gold chunk IDs are missing from the active snapshot. "
            "Benchmark stopped."
        )

    print("\nLoading CrossEncoder...")
    model = get_reranker()

    strategies = {
        "production_baseline": [],
        "metadata_experimental": [],
    }

    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "precision_at_5",
        "ndcg_at_10",
        "mrr_at_20",
        "hit_at_20",
    ]

    timings = {
        "production_baseline": [],
        "metadata_experimental": [],
    }

    original_bm25_k = bm25.k

    print("\nPRODUCTION CONFIGURATION")
    print("Dense top-k:", DENSE_TOP_K)
    print("BM25 top-k:", SPARSE_TOP_K)
    print("RRF candidates:", RERANK_CANDIDATES)
    print("RRF constant:", RRF_K)

    print("\nEXPERIMENTAL CONFIGURATION")
    print("Dense top-k: 100")
    print("BM25 top-k: 100")
    print("RRF candidates: 100")
    print("RRF constant:", RRF_K)

    print("\nStarting benchmark...\n")

    try:
        for number, question in enumerate(questions, 1):
            question_id = question["question_id"]
            query = question["question"]
            gold_ids = question["gold_chunk_ids"]

            # ---------------------------------------------
            # Strategy 1: Actual production configuration
            # ---------------------------------------------
            start = time.time()

            bm25.k = SPARSE_TOP_K

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=DENSE_TOP_K,
            )

            sparse = sparse_search(
                query,
                bm25,
                top_k=SPARSE_TOP_K,
            )

            production_candidates = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=RERANK_CANDIDATES,
            )

            production_ranked = rerank(
                model,
                query,
                production_candidates,
                use_metadata=False,
            )

            production_ids = ranked_chunk_ids(production_ranked)

            production_metrics = calculate_metrics(
                production_ids,
                gold_ids,
            )

            production_time = time.time() - start

            # ---------------------------------------------
            # Strategy 2: Metadata-aware experimental
            # ---------------------------------------------
            start = time.time()

            bm25.k = 100

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=100,
            )

            sparse = sparse_search(
                query,
                bm25,
                top_k=100,
            )

            experimental_candidates = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=100,
            )

            experimental_ranked = rerank(
                model,
                query,
                experimental_candidates,
                use_metadata=True,
            )

            experimental_ids = ranked_chunk_ids(
                experimental_ranked
            )

            experimental_metrics = calculate_metrics(
                experimental_ids,
                gold_ids,
            )

            experimental_time = time.time() - start

            timings["production_baseline"].append(production_time)
            timings["metadata_experimental"].append(
                experimental_time
            )

            strategies["production_baseline"].append({
                "question_id": question_id,
                "category": question.get("category", ""),
                "question": query,
                "gold_chunk_ids": gold_ids,
                "ranked_chunk_ids": production_ids[:20],
                "metrics": production_metrics,
            })

            strategies["metadata_experimental"].append({
                "question_id": question_id,
                "category": question.get("category", ""),
                "question": query,
                "gold_chunk_ids": gold_ids,
                "ranked_chunk_ids": experimental_ids[:20],
                "metrics": experimental_metrics,
            })

            print(
                f"{number:02d}/40 {question_id} | "
                f"Production rank: "
                f"{production_metrics['first_gold_rank']} | "
                f"Metadata rank: "
                f"{experimental_metrics['first_gold_rank']}"
            )

    finally:
        bm25.k = original_bm25_k

    # ---------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------
    summary = {
        "benchmark": "Phase 7M full regression",
        "timestamp": datetime.now().isoformat(),
        "question_count": len(questions),
        "snapshot_id": snapshot_id,
        "gold_chunk_count": len(all_gold_ids),
        "production_configuration": {
            "dense_top_k": DENSE_TOP_K,
            "bm25_top_k": SPARSE_TOP_K,
            "rrf_top_k": RERANK_CANDIDATES,
            "rrf_k": RRF_K,
            "reranking": "plain CrossEncoder",
        },
        "metadata_configuration": {
            "dense_top_k": 100,
            "bm25_top_k": 100,
            "rrf_top_k": 100,
            "rrf_k": RRF_K,
            "reranking": "metadata-aware CrossEncoder",
        },
        "production_baseline": {},
        "metadata_experimental": {},
        "runtime_seconds": {},
    }

    for strategy, records in strategies.items():
        for metric in metric_names:
            values = [
                record["metrics"][metric]
                for record in records
            ]

            summary[strategy][metric] = float(np.mean(values))

        summary["runtime_seconds"][strategy] = {
            "total": round(sum(timings[strategy]), 2),
            "mean_per_question": round(
                float(np.mean(timings[strategy])), 3
            ),
        }

    # ---------------------------------------------
    # Per-question comparison
    # ---------------------------------------------
    comparison_rows = []

    for baseline, experimental in zip(
        strategies["production_baseline"],
        strategies["metadata_experimental"],
    ):
        before = baseline["metrics"]
        after = experimental["metrics"]

        comparison_rows.append({
            "question_id": baseline["question_id"],
            "category": baseline["category"],
            "production_first_gold_rank": before["first_gold_rank"],
            "metadata_first_gold_rank": after["first_gold_rank"],
            "production_recall_at_5": before["recall_at_5"],
            "metadata_recall_at_5": after["recall_at_5"],
            "production_recall_at_10": before["recall_at_10"],
            "metadata_recall_at_10": after["recall_at_10"],
            "production_recall_at_20": before["recall_at_20"],
            "metadata_recall_at_20": after["recall_at_20"],
            "production_mrr_at_20": before["mrr_at_20"],
            "metadata_mrr_at_20": after["mrr_at_20"],
            "production_gold_ranks": json.dumps(
                before["gold_ranks"]
            ),
            "metadata_gold_ranks": json.dumps(
                after["gold_ranks"]
            ),
        })

    # ---------------------------------------------
    # Save results
    # ---------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = RESULTS_DIR / (
        "phase7m_full_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    with (output_dir / "per_question_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    with (output_dir / "rankings.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(strategies, file, indent=2)

    print("\n" + "=" * 65)
    print("PHASE 7M BENCHMARK COMPLETE")
    print("=" * 65)

    for strategy in strategies:
        print("\n" + strategy.upper())

        for metric in metric_names:
            print(
                f"{metric}: "
                f"{summary[strategy][metric]:.4f}"
            )

        print(
            "Total runtime:",
            summary["runtime_seconds"][strategy]["total"],
            "seconds",
        )

    print("\nRANK CHANGES")

    for row in comparison_rows:
        before = row["production_first_gold_rank"]
        after = row["metadata_first_gold_rank"]

        if before != after:
            print(
                f"{row['question_id']}: "
                f"{before} -> {after}"
            )

    print("\nSaved results:", output_dir)
    print("Total runtime:", round(time.time() - started, 2), "seconds")


if __name__ == "__main__":
    main()
