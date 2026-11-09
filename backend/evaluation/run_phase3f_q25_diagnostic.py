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


OUTPUT = (
    Path("evaluation/results") /
    f"phase3f_q25_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

GLOBAL_DEPTH = 100
RRF_K = 60
LOCAL_DENSE_K = 10
LOCAL_SPARSE_K = 10
LOCAL_RRF_TOP_K = 20

TARGET_FILE = "20071229X02007.pdf"

QUERIES = {
    "original": "What were the aircraft's departure point and destination?",
    "identity": (
        "NTSB accident NYC08CA055 "
        "Piper PA-22-150 registration N5921D"
    ),
    "field": "Departure Point Destination",
}


def chunk_id(item):
    return item["document"].metadata.get("chunk_id")


def document_name(item):
    return item["document"].metadata.get("file_name", "Unknown")


def document_key(document):
    return (
        document.metadata.get("document_id")
        or document.metadata.get("file_name")
        or document.metadata.get("source")
        or "unknown"
    )


def gold_rank(items, gold_ids):
    ids = [chunk_id(item) for item in items]
    ranks = [
        ids.index(gid) + 1
        for gid in gold_ids
        if gid in ids
    ]
    return min(ranks) if ranks else None


def aggregate_documents(items):
    scores = defaultdict(list)
    names = {}

    for item in items:
        doc = item["document"]
        key = document_key(doc)
        scores[key].append(float(item["rrf_score"]))
        names[key] = document_name(item)

    ranked = []

    for key, values in scores.items():
        ranked.append({
            "document_id": key,
            "name": names[key],
            "score": sum(sorted(values, reverse=True)[:5]),
        })

    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )


def retrieve_global(query, retrievers):
    dense = dense_search(
        query=query,
        vectors=retrievers["vectors"],
        embeddings=retrievers["embeddings"],
        chunks=retrievers["chunks"],
        top_k=GLOBAL_DEPTH,
    )

    bm25 = retrievers["bm25_retriever"]
    original_k = bm25.k

    try:
        bm25.k = GLOBAL_DEPTH
        sparse = sparse_search(
            query=query,
            bm25_retriever=bm25,
            top_k=GLOBAL_DEPTH,
        )
    finally:
        bm25.k = original_k

    return reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        rrf_k=RRF_K,
        top_k=GLOBAL_DEPTH,
    )


def retrieve_local(query, chunks, vectors, embeddings):
    dense = dense_search(
        query=query,
        vectors=vectors,
        embeddings=embeddings,
        chunks=chunks,
        top_k=LOCAL_DENSE_K,
    )

    bm25 = build_bm25_retriever(chunks)
    bm25.k = LOCAL_SPARSE_K

    sparse = sparse_search(
        query=query,
        bm25_retriever=bm25,
        top_k=LOCAL_SPARSE_K,
    )

    return reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        rrf_k=RRF_K,
        top_k=LOCAL_RRF_TOP_K,
    )


def print_ranked(label, items, gold_ids):
    print(f"\n{label}: {len(items)} candidates")
    print(f"Gold rank: {gold_rank(items, gold_ids) or 'NOT FOUND'}")

    for rank, item in enumerate(items[:5], start=1):
        doc = item["document"]
        cid = chunk_id(item)
        print(
            f"{rank}. {document_name(item)} | "
            f"chunk={doc.metadata.get('chunk_index')} | "
            f"gold={cid in gold_ids}"
        )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("Loading snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    chunks = retrievers["chunks"]
    vectors = np.asarray(retrievers["vectors"])
    embeddings = retrievers["embeddings"]

    gold_question = None

    with Path("evaluation/gold_questions.jsonl").open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item["question_id"] == "Q25":
                    gold_question = item
                    break

    if gold_question is None:
        raise RuntimeError("Q25 not found in gold dataset")

    gold_ids = set(gold_question["gold_chunk_ids"])

    target_indices = [
        i for i, chunk in enumerate(chunks)
        if chunk.metadata.get("file_name") == TARGET_FILE
    ]

    if not target_indices:
        raise RuntimeError(
            f"Target report not found: {TARGET_FILE}"
        )

    target_chunks = [chunks[i] for i in target_indices]
    target_vectors = vectors[target_indices]

    target_gold_ids = {
        chunk.metadata.get("chunk_id")
        for chunk in target_chunks
        if chunk.metadata.get("chunk_id") in gold_ids
    }

    if target_gold_ids != gold_ids:
        raise RuntimeError(
            f"Gold chunk mismatch. Dataset={gold_ids}, "
            f"target report={target_gold_ids}"
        )

    print(f"Target report chunks: {len(target_chunks)}")
    print(f"Gold chunk IDs: {sorted(gold_ids)}")

    results = []

    # Experiment A: global retrieval for each query.
    global_results = {}

    for label, query in QUERIES.items():
        print("\n" + "=" * 75)
        print(f"GLOBAL QUERY: {label}")
        print(query)

        started = time.perf_counter()
        hybrid = retrieve_global(query, retrievers)
        docs = aggregate_documents(hybrid)

        target_rank = next(
            (
                rank
                for rank, doc in enumerate(docs, start=1)
                if doc["name"] == TARGET_FILE
            ),
            None,
        )

        global_results[label] = hybrid

        print(
            f"Target document rank: "
            f"{target_rank or 'NOT FOUND'}"
        )
        print_ranked("Global hybrid", hybrid, gold_ids)

        print("\nTOP 5 DOCUMENTS")
        for rank, doc in enumerate(docs[:5], start=1):
            print(
                f"{rank}. {doc['name']} | "
                f"score={doc['score']:.5f}"
            )

        results.append({
            "experiment": "global",
            "query_type": label,
            "query": query,
            "target_document_rank": target_rank,
            "gold_passage_rank": gold_rank(hybrid, gold_ids),
            "top_documents": docs[:10],
            "runtime_seconds": round(
                time.perf_counter() - started, 3
            ),
        })

    # Experiment B: oracle-restricted local passage retrieval.
    print("\n" + "=" * 75)
    print("ORACLE-RESTRICTED LOCAL PASSAGE RETRIEVAL")
    print(f"Known report: {TARGET_FILE}")

    local_results = {}

    for label, query in QUERIES.items():
        started = time.perf_counter()

        local_hybrid = retrieve_local(
            query,
            target_chunks,
            target_vectors,
            embeddings,
        )

        reranked = rerank_candidates(query, local_hybrid)
        local_results[label] = reranked

        print(f"\nLOCAL QUERY: {label}")
        print_ranked("Local hybrid", local_hybrid, gold_ids)
        print_ranked("Local reranked", reranked, gold_ids)

        results.append({
            "experiment": "oracle_local",
            "query_type": label,
            "query": query,
            "hybrid_gold_rank": gold_rank(local_hybrid, gold_ids),
            "reranked_gold_rank": gold_rank(reranked, gold_ids),
            "runtime_seconds": round(
                time.perf_counter() - started, 3
            ),
        })

    result_path = OUTPUT / "q25_diagnostic.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("PHASE 3F SUMMARY")
    print("=" * 75)

    for item in results:
        if item["experiment"] == "global":
            print(
                f"GLOBAL {item['query_type']}: "
                f"document_rank="
                f"{item['target_document_rank'] or 'NOT FOUND'} | "
                f"gold_rank="
                f"{item['gold_passage_rank'] or 'NOT FOUND'}"
            )
        else:
            print(
                f"LOCAL {item['query_type']}: "
                f"hybrid_rank="
                f"{item['hybrid_gold_rank'] or 'NOT FOUND'} | "
                f"reranked_rank="
                f"{item['reranked_gold_rank'] or 'NOT FOUND'}"
            )

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
