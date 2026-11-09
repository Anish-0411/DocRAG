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
OUTPUT = (
    BASE / "results" /
    f"phase3e_document_passage_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

TARGET_IDS = {"Q02", "Q03", "Q04", "Q25"}

DOCUMENT_COUNT = 5
GLOBAL_DEPTH = 100
LOCAL_DENSE_K = 10
LOCAL_SPARSE_K = 10
LOCAL_RRF_K = 60
LOCAL_RRF_TOP_K = 20

# Separate field-oriented query: do not fuse it with the original query.
FIELD_QUERY = "Departure Point Destination"


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


def get_chunk_id(item):
    return item["document"].metadata.get("chunk_id")


def gold_rank(items, gold_ids):
    ids = [get_chunk_id(item) for item in items]
    ranks = [
        ids.index(gold_id) + 1
        for gold_id in gold_ids
        if gold_id in ids
    ]
    return min(ranks) if ranks else None


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
        ranked.append({
            "document_id": key,
            "name": names[key],
            "score": sum(sorted(values, reverse=True)[:5]),
            "retrieved_chunks": len(values),
        })

    return sorted(
        ranked,
        key=lambda item: item["score"],
        reverse=True,
    )


def local_retrieve(query, chunks, vectors, embeddings):
    if not chunks:
        return []

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
        rrf_k=LOCAL_RRF_K,
        top_k=LOCAL_RRF_TOP_K,
    )


def merge_candidates(*candidate_lists):
    merged = {}

    for candidates in candidate_lists:
        for item in candidates:
            cid = get_chunk_id(item)
            if not cid:
                continue

            previous = merged.get(cid)
            if (
                previous is None
                or item["rrf_score"] > previous["rrf_score"]
            ):
                merged[cid] = item

    return list(merged.values())


def serialize_top_items(items, gold_ids, limit=10):
    output = []

    for rank, item in enumerate(items[:limit], start=1):
        document = item["document"]
        cid = get_chunk_id(item)

        output.append({
            "rank": rank,
            "file_name": document_name(document),
            "chunk_id": cid,
            "chunk_index": document.metadata.get("chunk_index"),
            "is_gold": cid in gold_ids,
            "rerank_score": (
                float(item["rerank_score"])
                if "rerank_score" in item
                else None
            ),
        })

    return output


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        questions = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    selected = [
        question for question in questions
        if question["question_id"] in TARGET_IDS
    ]

    selected.sort(key=lambda question: question["question_id"])

    if len(selected) != len(TARGET_IDS):
        raise RuntimeError(
            f"Expected {len(TARGET_IDS)} questions; found {len(selected)}"
        )

    return selected


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    questions = load_questions()

    print("Loading active snapshot and retrievers...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    chunks = retrievers["chunks"]
    vectors = np.asarray(retrievers["vectors"])
    embeddings = retrievers["embeddings"]

    if len(chunks) != len(vectors):
        raise RuntimeError(
            f"Chunk/vector mismatch: {len(chunks)} vs {len(vectors)}"
        )

    # Build document -> chunk-index mapping once.
    document_indices = defaultdict(list)

    for index, chunk in enumerate(chunks):
        document_indices[document_key(chunk)].append(index)

    results = []

    for question in questions:
        qid = question["question_id"]
        query = question["question"]
        gold_ids = set(question["gold_chunk_ids"])

        print("\n" + "=" * 78)
        print(f"{qid}: {query}")

        started = time.perf_counter()

        # ---------------------------------------------------------
        # Stage 1: Global document identification.
        # ---------------------------------------------------------
        dense_global = dense_search(
            query=query,
            vectors=vectors,
            embeddings=embeddings,
            chunks=chunks,
            top_k=GLOBAL_DEPTH,
        )

        global_bm25 = retrievers["bm25_retriever"]
        original_bm25_k = global_bm25.k

        try:
            global_bm25.k = GLOBAL_DEPTH
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

        selected_keys = {
            item["document_id"] for item in selected_documents
        }

        gold_document_keys = {
            document_key(chunk)
            for chunk in chunks
            if chunk.metadata.get("chunk_id") in gold_ids
        }

        gold_document_ranks = {
            gold_key: next(
                (
                    rank
                    for rank, item in enumerate(
                        ranked_documents, start=1
                    )
                    if item["document_id"] == gold_key
                ),
                None,
            )
            for gold_key in gold_document_keys
        }

        # ---------------------------------------------------------
        # Stage 2: Passage retrieval inside each selected document.
        # ---------------------------------------------------------
        original_candidates = []
        field_candidates = []
        per_document = []

        for selected in selected_documents:
            doc_key = selected["document_id"]
            indices = document_indices.get(doc_key, [])

            local_chunks = [chunks[i] for i in indices]
            local_vectors = vectors[indices]

            original_local = local_retrieve(
                query,
                local_chunks,
                local_vectors,
                embeddings,
            )
            original_candidates.extend(original_local)

            field_local_count = 0

            if qid == "Q25":
                field_local = local_retrieve(
                    FIELD_QUERY,
                    local_chunks,
                    local_vectors,
                    embeddings,
                )
                field_candidates.extend(field_local)
                field_local_count = len(field_local)

            per_document.append({
                "file_name": selected["name"],
                "document_score": selected["score"],
                "document_rank": ranked_documents.index(selected) + 1,
                "local_chunk_count": len(local_chunks),
                "original_candidate_count": len(original_local),
                "field_candidate_count": field_local_count,
                "gold_document": doc_key in gold_document_keys,
            })

        # Deduplicate each strategy independently.
        original_candidates = merge_candidates(original_candidates)

        if qid == "Q25":
            field_candidates = merge_candidates(field_candidates)
        else:
            field_candidates = []

        # ---------------------------------------------------------
        # Stage 3: Compare independent passage strategies.
        # ---------------------------------------------------------
        original_reranked = rerank_candidates(
            query,
            original_candidates,
        )

        original_rank = gold_rank(original_reranked, gold_ids)

        field_reranked = []
        field_rank = None
        union_reranked = []
        union_rank = None

        if qid == "Q25":
            field_reranked = rerank_candidates(
                query,
                field_candidates,
            )
            field_rank = gold_rank(field_reranked, gold_ids)

            # Union, not RRF fusion. Keep the best local occurrence
            # of each chunk, then rerank with the original question.
            union_candidates = merge_candidates(
                original_candidates,
                field_candidates,
            )

            union_reranked = rerank_candidates(
                query,
                union_candidates,
            )
            union_rank = gold_rank(union_reranked, gold_ids)

        elapsed = time.perf_counter() - started

        print("\nDOCUMENT IDENTIFICATION")
        print(
            "Gold document rank(s):",
            gold_document_ranks or "No gold document identified",
        )
        print(
            "Gold document in selected top five:",
            bool(gold_document_keys & selected_keys),
        )

        for rank, item in enumerate(selected_documents, start=1):
            print(
                f"{rank}. {item['name']} | "
                f"score={item['score']:.5f} | "
                f"chunks={item['retrieved_chunks']}"
            )

        print("\nPASSAGE RETRIEVAL")
        print(
            f"Original query: candidates={len(original_candidates)}, "
            f"gold_rank={original_rank or 'NOT FOUND'}"
        )

        if qid == "Q25":
            print(
                f"Field query: candidates={len(field_candidates)}, "
                f"gold_rank={field_rank or 'NOT FOUND'}"
            )
            print(
                f"Union + original-question reranker: "
                f"candidates={len(union_candidates)}, "
                f"gold_rank={union_rank or 'NOT FOUND'}"
            )

        print("\nTOP 5 — ORIGINAL QUERY")
        for item in serialize_top_items(
            original_reranked, gold_ids, limit=5
        ):
            print(
                f"{item['rank']}. {item['file_name']} | "
                f"chunk={item['chunk_index']} | "
                f"gold={item['is_gold']} | "
                f"score={item['rerank_score']:.5f}"
            )

        if qid == "Q25":
            print("\nTOP 5 — FIELD QUERY")
            for item in serialize_top_items(
                field_reranked, gold_ids, limit=5
            ):
                print(
                    f"{item['rank']}. {item['file_name']} | "
                    f"chunk={item['chunk_index']} | "
                    f"gold={item['is_gold']} | "
                    f"score={item['rerank_score']:.5f}"
                )

            print("\nTOP 5 — UNION")
            for item in serialize_top_items(
                union_reranked, gold_ids, limit=5
            ):
                print(
                    f"{item['rank']}. {item['file_name']} | "
                    f"chunk={item['chunk_index']} | "
                    f"gold={item['is_gold']} | "
                    f"score={item['rerank_score']:.5f}"
                )

        print(f"\nRuntime: {elapsed:.2f}s")

        results.append({
            "question_id": qid,
            "question": query,
            "gold_chunk_ids": sorted(gold_ids),
            "gold_document_ranks": gold_document_ranks,
            "gold_document_in_top5": bool(
                gold_document_keys & selected_keys
            ),
            "selected_documents": per_document,
            "original_candidate_count": len(original_candidates),
            "original_gold_rank": original_rank,
            "original_top10": serialize_top_items(
                original_reranked, gold_ids
            ),
            "field_query": FIELD_QUERY if qid == "Q25" else None,
            "field_candidate_count": len(field_candidates),
            "field_gold_rank": field_rank,
            "field_top10": serialize_top_items(
                field_reranked, gold_ids
            ),
            "union_candidate_count": (
                len(union_candidates) if qid == "Q25" else None
            ),
            "union_gold_rank": union_rank,
            "union_top10": serialize_top_items(
                union_reranked, gold_ids
            ),
            "runtime_seconds": round(elapsed, 3),
        })

    result_path = OUTPUT / "phase3e_results.json"

    with result_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print("PHASE 3E SUMMARY")
    print("=" * 78)

    for item in results:
        print(
            f"{item['question_id']}: "
            f"doc_top5={item['gold_document_in_top5']} | "
            f"original_rank={item['original_gold_rank'] or 'NOT FOUND'}"
            + (
                f" | field_rank={item['field_gold_rank'] or 'NOT FOUND'}"
                f" | union_rank={item['union_gold_rank'] or 'NOT FOUND'}"
                if item["question_id"] == "Q25"
                else ""
            )
        )

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
