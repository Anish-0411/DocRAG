import csv
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)
from reranker import rerank_candidates

BASE = Path("evaluation")
DATASET = BASE / "gold_questions.jsonl"
OUTPUT = BASE / "results" / (
    f"phase3h_structured_fields_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

DEPTH = 100
RRF_K = 60
TOP_K_VALUES = (5, 10, 20)

def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def get_id(item):
    return item["document"].metadata.get("chunk_id")

def calculate_metrics(ids, gold):
    metrics = {}

    for k in TOP_K_VALUES:
        found = len(set(ids[:k]) & gold)
        metrics[f"recall@{k}"] = found / len(gold) if gold else 0.0
        metrics[f"precision@{k}"] = found / k
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, item_id in enumerate(ids[:k], 1)
            if item_id in gold
        )
        ideal_count = min(len(gold), k)
        ideal = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, ideal_count + 1)
        )
        metrics[f"ndcg@{k}"] = dcg / ideal if ideal else 0.0

    metrics["mrr@20"] = next(
        (1 / rank for rank, item_id in enumerate(ids[:20], 1)
         if item_id in gold),
        0.0,
    )
    metrics["hit@20"] = int(bool(set(ids[:20]) & gold))
    return metrics

def retrieve(query, retrievers):
    bm25 = retrievers["bm25_retriever"]
    original_k = bm25.k

    try:
        dense = dense_search(
            query=query,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=DEPTH,
        )

        bm25.k = DEPTH
        sparse = sparse_search(
            query=query,
            bm25_retriever=bm25,
            top_k=DEPTH,
        )
    finally:
        bm25.k = original_k

    return reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        rrf_k=RRF_K,
        top_k=DEPTH,
    )

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    questions = load_questions()

    if len(questions) != 40:
        raise RuntimeError(f"Expected 40 questions, found {len(questions)}")

    print("Loading active snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    print("Snapshot:", snapshot["snapshot_id"])
    print("Chunks:", len(retrievers["chunks"]))
    print("Questions:", len(questions))

    strategies = (
        "original_depth100",
        "structured_q25_depth100",
    )

    totals = {strategy: Counter() for strategy in strategies}
    rows = []
    q25_rankings = []
    started = time.perf_counter()

    for number, question in enumerate(questions, 1):
        qid = question["question_id"]
        original_query = question["question"]
        gold = set(question["gold_chunk_ids"])

        expanded_query = original_query
        if qid == "Q25":
            expanded_query = (
                original_query
                + " Departure Point Destination"
            )

        print(f"\n[{number}/40] {qid}: {original_query}")

        for strategy, query in (
            ("original_depth100", original_query),
            ("structured_q25_depth100", expanded_query),
        ):
            start = time.perf_counter()

            hybrid = retrieve(query, retrievers)
            reranked = rerank_candidates(query, hybrid)

            elapsed = time.perf_counter() - start
            ids = [get_id(item) for item in reranked]
            metrics = calculate_metrics(ids, gold)

            for metric, value in metrics.items():
                totals[strategy][metric] += value

            found = set(ids) & gold
            gold_ranks = {
                gold_id: ids.index(gold_id) + 1
                for gold_id in found
            }

            rows.append({
                "question_id": qid,
                "strategy": strategy,
                "query_used": query,
                "candidate_count": len(reranked),
                "gold_count": len(gold),
                "gold_found": len(found),
                "gold_ranks": json.dumps(gold_ranks),
                "runtime_seconds": round(elapsed, 3),
                **metrics,
            })

            print(
                f"  {strategy}: "
                f"Recall@5={metrics['recall@5']:.3f}, "
                f"Recall@20={metrics['recall@20']:.3f}, "
                f"MRR@20={metrics['mrr@20']:.3f}, "
                f"gold ranks={gold_ranks or 'NOT FOUND'}"
            )

            if qid == "Q25":
                for rank, item in enumerate(reranked[:20], 1):
                    doc = item["document"]
                    meta = doc.metadata
                    q25_rankings.append({
                        "strategy": strategy,
                        "rank": rank,
                        "chunk_id": meta.get("chunk_id"),
                        "file_name": meta.get("file_name"),
                        "page": meta.get("page"),
                        "score": float(item.get("rerank_score", 0)),
                        "text_preview": doc.page_content[:500],
                    })

    # Save per-question metrics.
    with (OUTPUT / "per_question_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT / "q25_rankings.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(q25_rankings, f, indent=2)

    summary = {
        strategy: {
            metric: round(value / len(questions), 4)
            for metric, value in totals[strategy].items()
        }
        for strategy in strategies
    }

    summary["experiment"] = {
        "snapshot_id": snapshot["snapshot_id"],
        "question_count": len(questions),
        "depth": DEPTH,
        "structured_expansion_applied_to": ["Q25"],
        "original_q25_query": next(
            q["question"] for q in questions
            if q["question_id"] == "Q25"
        ),
        "expanded_q25_query": (
            next(q["question"] for q in questions
                 if q["question_id"] == "Q25")
            + " Departure Point Destination"
        ),
        "runtime_seconds": round(time.perf_counter() - started, 2),
    }

    with (OUTPUT / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("PHASE 3H RESULTS")
    print("=" * 70)

    for strategy in strategies:
        print("\n", strategy)
        for metric, value in summary[strategy].items():
            print(f"  {metric}: {value}")

    print("\nQ25 rankings saved to:", OUTPUT / "q25_rankings.json")
    print("Per-question metrics:", OUTPUT / "per_question_metrics.csv")
    print("Summary:", OUTPUT / "summary.json")
    print("Output directory:", OUTPUT)

if __name__ == "__main__":
    main()
