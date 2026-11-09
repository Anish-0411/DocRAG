import json
import sys
import time
from collections import defaultdict
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

DATASET = Path("evaluation/gold_questions.jsonl")
OUTPUT = (
    Path("evaluation/results") /
    f"phase3d_multi_query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

RRF_K = 60
DEPTH = 100

EXPANSIONS = {
    "Q03": "NTSB probable cause determination findings",
    "Q25": "Departure Point Destination",
}


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    return {
        q["question_id"]: q
        for q in questions
        if q["question_id"] in EXPANSIONS
    }


def chunk_id(item):
    return item["document"].metadata.get("chunk_id")


def document_name(item):
    return item["document"].metadata.get("file_name", "Unknown")


def multi_query_fusion(ranked_lists):
    scores = defaultdict(float)
    documents = {}

    for results in ranked_lists:
        for rank, item in enumerate(results, start=1):
            cid = chunk_id(item)
            if not cid:
                continue

            scores[cid] += 1.0 / (RRF_K + rank)
            documents[cid] = item["document"]

    fused = [
        {
            "chunk_id": cid,
            "document": documents[cid],
            "rrf_score": score,
        }
        for cid, score in scores.items()
    ]

    return sorted(
        fused,
        key=lambda item: item["rrf_score"],
        reverse=True,
    )


def gold_rank(results, gold_ids):
    ids = [chunk_id(item) for item in results]
    ranks = [
        ids.index(gid) + 1
        for gid in gold_ids
        if gid in ids
    ]
    return min(ranks) if ranks else None


def document_ranking(results):
    scores = defaultdict(list)
    names = {}

    for item in results:
        name = document_name(item)
        scores[name].append(item["rrf_score"])
        names[name] = name

    ranked = [
        {
            "name": names[name],
            "score": sum(sorted(values, reverse=True)[:5]),
        }
        for name, values in scores.items()
    ]

    return sorted(ranked, key=lambda x: x["score"], reverse=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    all_results = []

    try:
        for qid, question in questions.items():
            original_query = question["question"]
            expanded_query = EXPANSIONS[qid]
            gold_ids = set(question["gold_chunk_ids"])

            print("\n" + "=" * 75)
            print(f"{qid}: {original_query}")
            print(f"Expansion: {expanded_query}")

            query_results = {}

            for label, query in [
                ("original", original_query),
                ("expanded", expanded_query),
            ]:
                started = time.perf_counter()

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

                hybrid = reciprocal_rank_fusion(
                    dense_results=dense,
                    sparse_results=sparse,
                    rrf_k=RRF_K,
                    top_k=DEPTH,
                )

                query_results[label] = hybrid

                print(
                    f"{label}: candidates={len(hybrid)}, "
                    f"gold_rank={gold_rank(hybrid, gold_ids) or 'NOT FOUND'}, "
                    f"time={time.perf_counter() - started:.2f}s"
                )

            # Combine rankings from both queries.
            fused = multi_query_fusion([
                query_results["original"],
                query_results["expanded"],
            ])

            # Rerank the combined top 100 candidates.
            fused = fused[:DEPTH]
            reranked = rerank_candidates(original_query, fused)

            fused_rank = gold_rank(fused, gold_ids)
            reranked_rank = gold_rank(reranked, gold_ids)

            print(f"\nMulti-query fused gold rank: {fused_rank or 'NOT FOUND'}")
            print(f"Multi-query reranked gold rank: {reranked_rank or 'NOT FOUND'}")

            print("\nTOP 5 RERANKED")
            for rank, item in enumerate(reranked[:5], start=1):
                doc = item["document"]
                cid = doc.metadata.get("chunk_id")
                print(
                    f"{rank}. {doc.metadata.get('file_name')} | "
                    f"chunk={doc.metadata.get('chunk_index')} | "
                    f"gold={cid in gold_ids} | "
                    f"score={float(item['rerank_score']):.5f}"
                )

            docs = document_ranking(fused)

            print("\nTOP 5 DOCUMENTS")
            for rank, doc in enumerate(docs[:5], start=1):
                print(
                    f"{rank}. {doc['name']} | "
                    f"score={doc['score']:.5f}"
                )

            all_results.append({
                "question_id": qid,
                "original_query": original_query,
                "expanded_query": expanded_query,
                "original_gold_rank": gold_rank(
                    query_results["original"], gold_ids
                ),
                "expanded_gold_rank": gold_rank(
                    query_results["expanded"], gold_ids
                ),
                "multi_query_gold_rank": fused_rank,
                "reranked_gold_rank": reranked_rank,
                "candidate_count": len(fused),
                "top_documents": docs[:5],
            })

    finally:
        bm25.k = original_bm25_k

    result_path = OUTPUT / "multi_query_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved: {result_path}")


if __name__ == "__main__":
    main()
