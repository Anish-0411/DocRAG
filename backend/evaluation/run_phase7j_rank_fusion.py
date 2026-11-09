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

RRF_K = 60
CANDIDATE_COUNT = 100

# (metadata weight, original RRF weight)
FUSION_CONFIGS = {
    "fusion_50_50": (0.50, 0.50),
    "fusion_70_30": (0.70, 0.30),
    "fusion_30_70": (0.30, 0.70),
}


def read_questions():
    with QUESTIONS_FILE.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_chunk_id(item):
    return item["document"].metadata.get("chunk_id")


def metadata_rerank(model, query, candidates):
    pairs = []

    for item in candidates:
        doc = item["document"]
        filename = doc.metadata.get("file_name") or "Unknown source"
        page = doc.metadata.get("page")

        # Match Phase 5C's human-readable page numbering.
        page_label = (
            str(int(page) + 1)
            if page is not None
            else "Unknown page"
        )

        enriched = (
            f"Source: {filename}\n"
            f"Page: {page_label}\n"
            f"Content:\n{doc.page_content}"
        )
        pairs.append((query, enriched))

    scores = model.predict(
        pairs,
        show_progress_bar=False,
        batch_size=16,
    )

    ranked = []
    for item, score in zip(candidates, scores):
        copied = dict(item)
        copied["metadata_score"] = float(score)
        ranked.append(copied)

    ranked.sort(
        key=lambda x: x["metadata_score"],
        reverse=True,
    )
    return ranked


def fuse_rankings(rrf_ranked, metadata_ranked, metadata_weight, rrf_weight):
    rrf_positions = {
        get_chunk_id(item): rank
        for rank, item in enumerate(rrf_ranked, 1)
    }
    metadata_positions = {
        get_chunk_id(item): rank
        for rank, item in enumerate(metadata_ranked, 1)
    }

    fused = []

    for item in metadata_ranked:
        chunk_id = get_chunk_id(item)

        rrf_rank = rrf_positions.get(chunk_id)
        metadata_rank = metadata_positions.get(chunk_id)

        if rrf_rank is None or metadata_rank is None:
            continue

        score = (
            metadata_weight / (RRF_K + metadata_rank)
            + rrf_weight / (RRF_K + rrf_rank)
        )

        copied = dict(item)
        copied["fusion_score"] = score
        copied["original_rrf_rank"] = rrf_rank
        copied["metadata_rank"] = metadata_rank
        fused.append(copied)

    fused.sort(
        key=lambda x: (
            -x["fusion_score"],
            x["metadata_rank"],
            x["original_rrf_rank"],
        )
    )
    return fused


def evaluate(ranked, gold_ids):
    gold_ids = set(gold_ids)
    ranked_ids = [get_chunk_id(item) for item in ranked]

    def recall(k):
        return (
            sum(x in gold_ids for x in ranked_ids[:k]) / len(gold_ids)
            if gold_ids else 0.0
        )

    def precision(k):
        return sum(x in gold_ids for x in ranked_ids[:k]) / k

    def ndcg(k):
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

    ranks = [
        i + 1
        for i, chunk_id in enumerate(ranked_ids)
        if chunk_id in gold_ids
    ]
    first_rank = min(ranks) if ranks else None

    return {
        "recall_at_5": recall(5),
        "recall_at_10": recall(10),
        "recall_at_20": recall(20),
        "precision_at_5": precision(5),
        "ndcg_at_10": ndcg(10),
        "mrr_at_20": (
            1 / first_rank
            if first_rank is not None and first_rank <= 20
            else 0.0
        ),
        "hit_at_20": int(
            first_rank is not None and first_rank <= 20
        ),
        "first_gold_rank": first_rank,
        "gold_ranks": "|".join(str(x) for x in ranks),
    }


def main():
    started = time.time()
    questions = read_questions()

    print("Loading snapshot and local reranker...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)
    model = get_reranker()

    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    strategy_names = [
        "original_rrf",
        "metadata",
        *FUSION_CONFIGS.keys(),
    ]

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
        strategy: {metric: [] for metric in metric_names}
        for strategy in strategy_names
    }

    rows = []

    print("Questions:", len(questions))
    print("Snapshot:", snapshot.get("snapshot_id", "unknown"))
    print("Candidate depth:", CANDIDATE_COUNT)
    print("Running Phase 7J...\n")

    try:
        bm25.k = CANDIDATE_COUNT

        for number, question in enumerate(questions, 1):
            query = question["question"]
            gold_ids = question["gold_chunk_ids"]
            question_id = question["question_id"]

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=CANDIDATE_COUNT,
            )

            sparse = sparse_search(
                query,
                bm25,
                top_k=CANDIDATE_COUNT,
            )

            rrf_ranked = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=CANDIDATE_COUNT,
            )

            metadata_ranked = metadata_rerank(
                model,
                query,
                rrf_ranked,
            )

            rankings = {
                "original_rrf": rrf_ranked,
                "metadata": metadata_ranked,
            }

            for name, (metadata_weight, rrf_weight) in FUSION_CONFIGS.items():
                rankings[name] = fuse_rankings(
                    rrf_ranked,
                    metadata_ranked,
                    metadata_weight,
                    rrf_weight,
                )

            row = {
                "question_id": question_id,
                "phase7_focus": question.get("phase7_focus", ""),
                "candidate_count": len(rrf_ranked),
            }

            for strategy, ranking in rankings.items():
                result = evaluate(ranking, gold_ids)

                for metric in metric_names:
                    totals[strategy][metric].append(result[metric])
                    row[f"{strategy}_{metric}"] = result[metric]

                row[f"{strategy}_gold_ranks"] = result["gold_ranks"]
                row[f"{strategy}_first_gold_rank"] = result["first_gold_rank"]

            rows.append(row)

            print(
                f"{number}/{len(questions)} {question_id} | "
                f"RRF={row['original_rrf_first_gold_rank']} | "
                f"Metadata={row['metadata_first_gold_rank']} | "
                f"50/50={row['fusion_50_50_first_gold_rank']} | "
                f"70/30={row['fusion_70_30_first_gold_rank']} | "
                f"30/70={row['fusion_30_70_first_gold_rank']}"
            )

    finally:
        bm25.k = original_bm25_k

    output_dir = RESULTS_DIR / (
        "phase7j_rank_fusion_" + time.strftime("%Y%m%d_%H%M%S")
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
        "candidate_count": CANDIDATE_COUNT,
        "rrf_k": RRF_K,
        "strategies": {},
    }

    for strategy in strategy_names:
        summary["strategies"][strategy] = {
            metric: float(np.mean(totals[strategy][metric]))
            for metric in metric_names
        }

    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n========== PHASE 7J SUMMARY ==========")
    for strategy in strategy_names:
        print(f"\n{strategy.upper()}")
        for metric in metric_names:
            value = summary["strategies"][strategy][metric]
            print(f"{metric}: {value:.4f}")

    print("\nSaved:", output_dir)
    print("Runtime:", round(time.time() - started, 2), "seconds")


if __name__ == "__main__":
    main()
