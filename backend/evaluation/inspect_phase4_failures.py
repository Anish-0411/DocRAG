import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)
from reranker import rerank_candidates

DATASET = Path("evaluation/gold_questions.jsonl")

TARGETS = ["Q02", "Q03", "Q04", "Q06", "Q07", "Q25", "Q36", "Q37"]


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    return {item["question_id"]: item for item in records}


def get_document(item):
    """Support both Document and dictionary retrieval results."""
    if isinstance(item, dict):
        return item["document"]
    return item


def print_candidates(label, candidates, gold_ids, limit=12):
    print(f"\\n--- {label} ---")

    found = False

    for rank, item in enumerate(candidates[:limit], start=1):

        # Dense and sparse return Document objects.
        # Hybrid and reranked retrieval return dictionaries.
        doc = get_document(item)
        score = (
            item.get("rerank_score", item.get("rrf_score", "N/A"))
            if isinstance(item, dict)
            else "N/A"
        )

        metadata = doc.metadata
        chunk_id = metadata.get("chunk_id")
        is_gold = chunk_id in gold_ids
        found = found or is_gold

        print(
            f"\\n{rank:02d}. {'[GOLD] ' if is_gold else ''}"
            f"{metadata.get('file_name', 'Unknown')} | "
            f"page={metadata.get('page')} | "
            f"chunk={metadata.get('chunk_index')} | "
            f"score={score}"
        )

        print("    chunk_id:", chunk_id)
        print(
            "    text:",
            " ".join(doc.page_content.split())[:220]
        )

    all_gold_ids = {
        (
            get_document(item).metadata.get("chunk_id")
            if isinstance(item, dict)
            else item.metadata.get("chunk_id")
        )
        for item in candidates
    }

    if not found and gold_ids.intersection(all_gold_ids):
        print("\\nGold evidence exists beyond the displayed limit.")


def main():
    questions = load_questions()
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    bm25 = retrievers["bm25_retriever"]
    original_k = bm25.k

    try:
        for qid in TARGETS:
            question = questions[qid]
            query = question["question"]
            gold_ids = set(question["gold_chunk_ids"])

            print("\n" + "=" * 90)
            print(qid, "-", query)
            print("Expected document:", question["document"])
            print("Expected page:", question["page"])
            print("Expected answer:", question["expected_answer"])

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=100,
            )

            bm25.k = 100
            sparse = sparse_search(
                query=query,
                bm25_retriever=bm25,
                top_k=100,
            )

            hybrid = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=60,
                top_k=100,
            )

            reranked = rerank_candidates(query, hybrid)

            print_candidates("DENSE TOP 12", dense, gold_ids)
            print_candidates("BM25 TOP 12", sparse, gold_ids)
            print_candidates("HYBRID RRF TOP 12", hybrid, gold_ids)
            print_candidates("RERANKED TOP 25", reranked, gold_ids, 25)

            for label, candidates in [
                ("DENSE", dense),
                ("BM25", sparse),
                ("HYBRID", hybrid),
                ("RERANKED", reranked),
            ]:
                ranks = [
                    index + 1
                    for index, item in enumerate(candidates)
                    if get_document(item).metadata.get("chunk_id") in gold_ids
                ]
                print(f"{label} gold ranks:", ranks or "NOT RETRIEVED")

    finally:
        bm25.k = original_k


if __name__ == "__main__":
    main()
