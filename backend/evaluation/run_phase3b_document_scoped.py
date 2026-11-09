import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    build_bm25_retriever,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)
from reranker import rerank_candidates


BASE = Path("evaluation")
DATASET = BASE / "gold_questions.jsonl"
TARGET_IDS = {"Q02", "Q03", "Q04", "Q25"}

DOCUMENT_COUNT = 5
GLOBAL_DEPTH = 100
LOCAL_DENSE_K = 10
LOCAL_SPARSE_K = 10
LOCAL_RRF_K = 60
LOCAL_RRF_TOP_K = 20


def document_key(document):
    metadata = document.metadata
    return (
        metadata.get("document_id")
        or metadata.get("file_name")
        or metadata.get("source")
        or "unknown"
    )


def document_name(document):
    return document.metadata.get("file_name", "Unknown")


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return [q for q in questions if q["question_id"] in TARGET_IDS]


def aggregate_documents(hybrid_results):
    scores = defaultdict(list)
    names = {}

    for item in hybrid_results:
        document = item["document"]
        key = document_key(document)
        scores[key].append(float(item["rrf_score"]))
        names[key] = document_name(document)

    ranked = []
    for key, values in scores.items():
        score = sum(sorted(values, reverse=True)[:5])
        ranked.append({
            "document_id": key,
            "name": names[key],
            "score": score,
        })

    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def main():
    questions = load_questions()
    if len(questions) != len(TARGET_IDS):
        raise RuntimeError(
            f"Expected {len(TARGET_IDS)} questions, found {len(questions)}"
        )

    print(f"Loaded questions: {[q['question_id'] for q in questions]}")

    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    chunks = retrievers["chunks"]
    vectors = np.asarray(retrievers["vectors"])

    if len(chunks) != len(vectors):
        raise RuntimeError(
            f"Chunk/vector mismatch: {len(chunks)} chunks, {len(vectors)} vectors"
        )

    # Group every indexed chunk by its source document.
    document_indices = defaultdict(list)
    for index, chunk in enumerate(chunks):
        document_indices[document_key(chunk)].append(index)

    output_dir = (
        BASE / "results" /
        f"phase3b_document_scoped_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for number, question in enumerate(questions, start=1):
        qid = question["question_id"]
        query = question["question"]
        gold = set(question["gold_chunk_ids"])

        started = time.perf_counter()
        print("\n" + "=" * 75)
        print(f"{qid} | {query}")

        # Stage 1: global retrieval and document-level aggregation.
        dense_global = dense_search(
            query=query,
            vectors=vectors,
            embeddings=retrievers["embeddings"],
            chunks=chunks,
            top_k=GLOBAL_DEPTH,
        )

        global_bm25 = retrievers["bm25_retriever"]
        original_bm25_k = global_bm25.k
        global_bm25.k = GLOBAL_DEPTH

        try:
            sparse_global = sparse_search(
                query=query,
                bm25_retriever=global_bm25,
                top_k=GLOBAL_DEPTH,
            )
        finally:
            global_bm25.k = original_bm25_k

        global_hybrid = reciprocal_rank_fusion(
            dense_results=dense_global,
            sparse_results=sparse_global,
            rrf_k=LOCAL_RRF_K,
            top_k=GLOBAL_DEPTH,
        )

        ranked_documents = aggregate_documents(global_hybrid)
        selected_documents = ranked_documents[:DOCUMENT_COUNT]

        # Stage 2: retrieve passages independently inside each selected document.
        merged_candidates = {}
        document_candidate_counts = {}

        for selected in selected_documents:
            doc_key = selected["document_id"]
            indices = document_indices.get(doc_key, [])

            if not indices:
                continue

            local_chunks = [chunks[i] for i in indices]
            local_vectors = vectors[indices]

            local_dense = dense_search(
                query=query,
                vectors=local_vectors,
                embeddings=retrievers["embeddings"],
                chunks=local_chunks,
                top_k=LOCAL_DENSE_K,
            )

            local_bm25 = build_bm25_retriever(local_chunks)
            local_bm25.k = LOCAL_SPARSE_K

            local_sparse = sparse_search(
                query=query,
                bm25_retriever=local_bm25,
                top_k=LOCAL_SPARSE_K,
            )

            local_hybrid = reciprocal_rank_fusion(
                dense_results=local_dense,
                sparse_results=local_sparse,
                rrf_k=LOCAL_RRF_K,
                top_k=LOCAL_RRF_TOP_K,
            )

            document_candidate_counts[selected["name"]] = len(local_hybrid)

            for item in local_hybrid:
                chunk_id = item["document"].metadata.get("chunk_id")
                if chunk_id:
                    # Keep the strongest occurrence if a chunk appears twice.
                    previous = merged_candidates.get(chunk_id)
                    if (
                        previous is None
                        or item["rrf_score"] > previous["rrf_score"]
                    ):
                        merged_candidates[chunk_id] = item

        candidate_pool = list(merged_candidates.values())

        # Stage 3: rerank only the document-scoped candidate pool.
        reranked = rerank_candidates(query, candidate_pool)

        ranked_ids = [
            item["document"].metadata.get("chunk_id")
            for item in reranked
        ]

        gold_ranks = {
            gold_id: ranked_ids.index(gold_id) + 1
            for gold_id in gold
            if gold_id in ranked_ids
        }

        best_gold_rank = min(gold_ranks.values()) if gold_ranks else None
        elapsed = time.perf_counter() - started

        print(f"Selected documents: {len(selected_documents)}")
        print(f"Candidate chunks: {len(candidate_pool)}")
        print(f"Gold chunk rank: {best_gold_rank or 'NOT FOUND'}")
        print(f"Runtime: {elapsed:.2f}s")

        print("\nSELECTED DOCUMENTS")
        for rank, doc in enumerate(selected_documents, start=1):
            print(
                f"{rank}. {doc['name']} | "
                f"score={doc['score']:.5f} | "
                f"local_candidates="
                f"{document_candidate_counts.get(doc['name'], 0)}"
            )

        print("\nTOP 5 RERANKED CHUNKS")
        for rank, item in enumerate(reranked[:5], start=1):
            document = item["document"]
            chunk_id = document.metadata.get("chunk_id")
            print(
                f"{rank}. {document_name(document)} | "
                f"chunk={document.metadata.get('chunk_index')} | "
                f"gold={chunk_id in gold} | "
                f"score={float(item['rerank_score']):.5f}"
            )

        results.append({
            "question_id": qid,
            "question": query,
            "selected_documents": selected_documents,
            "document_candidate_counts": document_candidate_counts,
            "candidate_count": len(candidate_pool),
            "gold_ranks": gold_ranks,
            "best_gold_rank": best_gold_rank,
            "top5": [
                {
                    "file_name": document_name(item["document"]),
                    "chunk_id": item["document"].metadata.get("chunk_id"),
                    "chunk_index": item["document"].metadata.get("chunk_index"),
                    "rerank_score": float(item["rerank_score"]),
                }
                for item in reranked[:5]
            ],
            "runtime_seconds": round(elapsed, 3),
        })

    result_path = output_dir / "document_scoped_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 3B SUMMARY")
    print("=" * 75)

    for result in results:
        print(
            f"{result['question_id']}: "
            f"gold_rank={result['best_gold_rank'] or 'NOT FOUND'}, "
            f"candidates={result['candidate_count']}, "
            f"time={result['runtime_seconds']:.2f}s"
        )

    print(f"\nSaved results: {result_path}")


if __name__ == "__main__":
    main()
