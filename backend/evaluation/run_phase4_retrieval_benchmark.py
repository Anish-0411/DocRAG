import csv
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# Ensure backend modules are importable when this script is
# executed directly from the evaluation directory.
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

OUTPUT = (
    BASE / "results" /
    f"phase4_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

TOP_K_VALUES = (5, 10, 20)
RRF_K = 60


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_id(item):
    document = item["document"]
    return document.metadata.get("chunk_id")


def ranked_ids(items):
    return [chunk_id(item) for item in items]


def recall_at_k(ids, gold, k):
    return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ids, gold, k):
    return len(set(ids[:k]) & gold) / k if k else 0.0


def reciprocal_rank(ids, gold, k=20):
    for rank, item_id in enumerate(ids[:k], start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ids, gold, k):
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(ids[:k], start=1)
        if item_id in gold
    )
    ideal_count = min(len(gold), k)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / ideal if ideal else 0.0


def calculate_metrics(ids, gold):
    result = {}
    for k in TOP_K_VALUES:
        result[f"recall@{k}"] = recall_at_k(ids, gold, k)
        result[f"precision@{k}"] = precision_at_k(ids, gold, k)
        result[f"ndcg@{k}"] = ndcg_at_k(ids, gold, k)

    result["mrr@20"] = reciprocal_rank(ids, gold, 20)
    result["hit@20"] = int(bool(set(ids[:20]) & gold))
    return result


def document_distribution(items):
    counts = Counter(
        item["document"].metadata.get("file_name", "Unknown")
        for item in items
    )
    return counts


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} gold questions")

    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    rows = []
    aggregate = {}

    strategies = [
        "baseline_hybrid_rrf",
        "baseline_hybrid_reranked",
        "depth100_hybrid_rrf",
        "depth100_hybrid_reranked",
    ]

    totals = {
        strategy: Counter()
        for strategy in strategies
    }

    try:
        for number, question in enumerate(questions, start=1):
            qid = question["question_id"]
            query = question["question"]
            gold = set(question["gold_chunk_ids"])

            print(f"[{number}/{len(questions)}] {qid}")

            # Existing production retrieval depth: 15 dense + 15 sparse.
            dense15 = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=15,
            )

            bm25.k = 15
            sparse15 = sparse_search(
                query=query,
                bm25_retriever=bm25,
                top_k=15,
            )

            hybrid15 = reciprocal_rank_fusion(
                dense_results=dense15,
                sparse_results=sparse15,
                rrf_k=RRF_K,
                top_k=20,
            )

            # Experimental retrieval depth: 100 dense + 100 sparse.
            dense100 = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=100,
            )

            bm25.k = 100
            sparse100 = sparse_search(
                query=query,
                bm25_retriever=bm25,
                top_k=100,
            )

            hybrid100 = reciprocal_rank_fusion(
                dense_results=dense100,
                sparse_results=sparse100,
                rrf_k=RRF_K,
                top_k=100,
            )

            # CrossEncoder uses the same existing implementation.
            reranked15 = rerank_candidates(query, hybrid15)
            reranked100 = rerank_candidates(query, hybrid100)

            result_sets = {
                "baseline_hybrid_rrf": hybrid15,
                "baseline_hybrid_reranked": reranked15,
                "depth100_hybrid_rrf": hybrid100,
                "depth100_hybrid_reranked": reranked100,
            }

            for strategy, candidates in result_sets.items():
                ids = ranked_ids(candidates)
                metrics = calculate_metrics(ids, gold)

                for metric, value in metrics.items():
                    totals[strategy][metric] += value

                rows.append({
                    "question_id": qid,
                    "strategy": strategy,
                    "candidate_count": len(candidates),
                    "gold_count": len(gold),
                    "gold_found": len(set(ids) & gold),
                    "gold_ranks": [
                        ids.index(gold_id) + 1
                        for gold_id in gold
                        if gold_id in ids
                    ],
                    "gold_document": question.get("document"),
                    "gold_page": question.get("page"),
                    **metrics,
                })

            # Record the full depth-100 reranked candidate list.
            ids100 = ranked_ids(reranked100)
            gold_ranks100 = {
                gold_id: ids100.index(gold_id) + 1
                for gold_id in gold
                if gold_id in ids100
            }

            aggregate[qid] = {
                "question": query,
                "gold_chunk_ids": list(gold),
                "depth100_gold_ranks": gold_ranks100,
                "depth100_document_distribution": dict(
                    document_distribution(reranked100)
                ),
                "depth100_candidate_count": len(reranked100),
            }

    finally:
        bm25.k = original_bm25_k

    # Write per-question metrics.
    csv_path = OUTPUT / "per_question_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write per-question diagnostics.
    with (OUTPUT / "candidate_diagnostics.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(aggregate, f, indent=2)

    # Calculate macro averages.
    summary = {}
    for strategy in strategies:
        summary[strategy] = {
            metric: round(value / len(questions), 4)
            for metric, value in totals[strategy].items()
        }

    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nPHASE 4 RETRIEVAL RESULTS")
    print("=" * 75)

    for strategy, metrics in summary.items():
        print(f"\n{strategy}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("\nOutput directory:", OUTPUT)
    print("CSV:", csv_path)
    print("Summary:", OUTPUT / "summary.json")


if __name__ == "__main__":
    main()
