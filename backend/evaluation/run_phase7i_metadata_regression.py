import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)
from reranker import get_reranker


QUESTIONS_FILE = BACKEND / "evaluation" / "phase7_regression_questions.jsonl"
RESULTS_DIR = BACKEND / "evaluation" / "results"


def read_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def rerank(model, query, candidates, use_metadata):
    pairs = []

    for item in candidates:
        doc = item["document"]

        if use_metadata:
            filename = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            text = (
                f"Source document: {filename}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )
        else:
            text = doc.page_content

        pairs.append((query, text))

    scores = model.predict(pairs, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))

    return [candidates[int(i)] for i in order]


def ranking_metrics(ranked, gold_ids):
    gold_ids = set(gold_ids)
    ranked_ids = [
        item["document"].metadata.get("chunk_id")
        for item in ranked
    ]

    def recall_at(k):
        return (
            sum(x in gold_ids for x in ranked_ids[:k])
            / len(gold_ids)
            if gold_ids else 0.0
        )

    def precision_at(k):
        return sum(
            x in gold_ids for x in ranked_ids[:k]
        ) / k

    def ndcg_at(k):
        dcg = sum(
            1 / math.log2(i + 2)
            for i, chunk_id in enumerate(ranked_ids[:k])
            if chunk_id in gold_ids
        )
        ideal_count = min(len(gold_ids), k)
        ideal = sum(
            1 / math.log2(i + 2)
            for i in range(ideal_count)
        )
        return dcg / ideal if ideal else 0.0

    first_rank = next(
        (
            i + 1
            for i, chunk_id in enumerate(ranked_ids)
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
            1 / first_rank if first_rank and first_rank <= 20 else 0.0
        ),
        "hit_at_20": int(first_rank is not None and first_rank <= 20),
        "first_gold_rank": first_rank,
        "gold_ranks": [
            str(ranked_ids.index(gold_id) + 1)
            if gold_id in ranked_ids else "not_found"
            for gold_id in gold_ids
        ],
        "gold_found_count": sum(x in gold_ids for x in ranked_ids),
    }


def main():
    started = time.time()
    questions = read_questions()

    print("Loading active snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)
    model = get_reranker()

    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    rows = []
    metric_names = [
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "precision_at_5",
        "ndcg_at_10",
        "mrr_at_20",
        "hit_at_20",
    ]

    totals = {
        "baseline": {name: [] for name in metric_names},
        "metadata": {name: [] for name in metric_names},
    }

    print(f"Questions: {len(questions)}")
    print(f"Snapshot: {snapshot.get('snapshot_id', 'unknown')}")
    print("Retrieval: dense=100, BM25=100, RRF=100")
    print("Running baseline and metadata-aware reranking...\n")

    try:
        bm25.k = 100

        for number, question in enumerate(questions, 1):
            question_id = question["question_id"]
            query = question["question"]
            gold_ids = question["gold_chunk_ids"]

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=100,
            )

            sparse = sparse_search(query, bm25, top_k=100)

            candidates = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=60,
                top_k=100,
            )

            baseline = rerank(model, query, candidates, False)
            metadata = rerank(model, query, candidates, True)

            baseline_metrics = ranking_metrics(baseline, gold_ids)
            metadata_metrics = ranking_metrics(metadata, gold_ids)

            row = {
                "question_id": question_id,
                "category": question.get("category", ""),
                "phase7_focus": question.get("phase7_focus", ""),
                "candidate_count": len(candidates),
            }

            for strategy, result in (
                ("baseline", baseline_metrics),
                ("metadata", metadata_metrics),
            ):
                for name in metric_names:
                    totals[strategy][name].append(result[name])
                    row[f"{strategy}_{name}"] = result[name]

                row[f"{strategy}_gold_ranks"] = "|".join(
                    result["gold_ranks"]
                )
                row[f"{strategy}_first_gold_rank"] = (
                    result["first_gold_rank"]
                )
                row[f"{strategy}_gold_found_count"] = (
                    result["gold_found_count"]
                )

            rows.append(row)

            print(
                f"{number}/{len(questions)} {question_id} | "
                f"baseline rank={baseline_metrics['first_gold_rank']} | "
                f"metadata rank={metadata_metrics['first_gold_rank']}"
            )

    finally:
        bm25.k = original_bm25_k

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = RESULTS_DIR / (
        "phase7i_metadata_" + time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "per_question_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "question_count": len(rows),
        "snapshot_id": snapshot.get("snapshot_id"),
        "retrieval": {
            "dense_top_k": 100,
            "bm25_top_k": 100,
            "rrf_top_k": 100,
            "rrf_k": 60,
        },
        "baseline": {},
        "metadata": {},
    }

    for strategy in ("baseline", "metadata"):
        for name in metric_names:
            values = totals[strategy][name]
            summary[strategy][name] = (
                float(np.mean(values)) if values else 0.0
            )

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n========== PHASE 7I RESULTS ==========")
    print("Questions:", len(rows))

    for strategy in ("baseline", "metadata"):
        print(f"\n{strategy.upper()}")
        for name in metric_names:
            print(f"{name}: {summary[strategy][name]:.4f}")

    print("\n========== RANK CHANGES ==========")
    for row in rows:
        before = row["baseline_first_gold_rank"]
        after = row["metadata_first_gold_rank"]
        if before != after:
            print(
                f"{row['question_id']}: "
                f"{before} -> {after}"
            )

    print("\nSaved results:", output_dir)
    print("Runtime:", round(time.time() - started, 2), "seconds")


if __name__ == "__main__":
    main()
