import csv
import json
import sys
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


DATASET = Path("evaluation/phase7_regression_questions.jsonl")
TARGET_IDS = {"Q02", "Q03", "Q04", "Q06", "Q07"}

DENSE_K = 100
SPARSE_K = 100
RRF_K = 60
RRF_DEPTH = 200

OUTPUT = Path("evaluation/results") / (
    f"phase7g_stage_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]

    return [
        q for q in questions
        if q["question_id"] in TARGET_IDS
    ]


def document_id(document):
    return document.metadata.get("chunk_id")


def rank_map(items, get_id):
    result = {}
    for rank, item in enumerate(items, start=1):
        item_id = get_id(item)
        if item_id:
            result[item_id] = rank
    return result


def metadata_for_chunk(items, gold_id, get_id):
    for item in items:
        if get_id(item) == gold_id:
            return item.metadata
    return {}


def main():
    questions = load_questions()
    print(f"Loaded {len(questions)} target questions")

    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    print("Snapshot:", retrievers["snapshot_id"])
    print("Total indexed chunks:", len(retrievers["chunks"]))

    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    rows = []

    try:
        # BM25Retriever.invoke() itself is limited by bm25.k.
        bm25.k = SPARSE_K

        for question in questions:
            qid = question["question_id"]
            query = question["question"]
            gold_ids = set(question.get("gold_chunk_ids", []))

            print(f"\nProcessing {qid}: {query}")

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=DENSE_K,
            )

            sparse = sparse_search(
                query=query,
                bm25_retriever=bm25,
                top_k=SPARSE_K,
            )

            # Keep the full union (up to 200 candidates) so we can
            # observe whether RRF itself or a later cutoff loses gold.
            fused = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=RRF_DEPTH,
            )

            reranked = rerank_candidates(
                query=query,
                candidates=fused,
            )

            dense_ranks = rank_map(dense, document_id)
            sparse_ranks = rank_map(sparse, document_id)
            rrf_ranks = {
                item["chunk_id"]: rank
                for rank, item in enumerate(fused, start=1)
            }
            rerank_ranks = {
                item["chunk_id"]: rank
                for rank, item in enumerate(reranked, start=1)
            }

            print(
                f"  dense={len(dense)}, sparse={len(sparse)}, "
                f"RRF={len(fused)}, reranked={len(reranked)}, "
                f"gold_chunks={len(gold_ids)}"
            )

            for gold_id in sorted(gold_ids):
                metadata = metadata_for_chunk(
                    retrievers["chunks"],
                    gold_id,
                    document_id,
                )

                row = {
                    "question_id": qid,
                    "gold_chunk_id": gold_id,
                    "file_name": metadata.get("file_name"),
                    "page": metadata.get("page"),
                    "dense_rank": dense_ranks.get(gold_id),
                    "sparse_rank": sparse_ranks.get(gold_id),
                    "rrf_rank": rrf_ranks.get(gold_id),
                    "rrf_within_100": (
                        rrf_ranks[gold_id] <= 100
                        if gold_id in rrf_ranks else False
                    ),
                    "rerank_rank": rerank_ranks.get(gold_id),
                    "dense_found": gold_id in dense_ranks,
                    "sparse_found": gold_id in sparse_ranks,
                    "rrf_found": gold_id in rrf_ranks,
                    "rerank_found": gold_id in rerank_ranks,
                }

                rows.append(row)

                print(
                    f"  GOLD {gold_id[:12]} | "
                    f"dense={row['dense_rank']} | "
                    f"BM25={row['sparse_rank']} | "
                    f"RRF={row['rrf_rank']} | "
                    f"reranked={row['rerank_rank']} | "
                    f"file={row['file_name']} page={row['page']}"
                )

    finally:
        bm25.k = original_bm25_k

    output_csv = OUTPUT / "gold_stage_ranks.csv"

    fieldnames = [
        "question_id",
        "gold_chunk_id",
        "file_name",
        "page",
        "dense_rank",
        "sparse_rank",
        "rrf_rank",
        "rrf_within_100",
        "rerank_rank",
        "dense_found",
        "sparse_found",
        "rrf_found",
        "rerank_found",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nDiagnostic complete.")
    print("Gold chunks examined:", len(rows))
    print("CSV:", output_csv)


if __name__ == "__main__":
    main()
