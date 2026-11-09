import sys
from pathlib import Path

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


RRF_K = 60

TARGET_DOCUMENT = "20071229X02007.pdf"

TESTS = [
    {
        "id": "Q03",
        "gold_id": "6eee78",  # Prefix used only for display
        "queries": [
            "What was the NTSB's probable cause?",
            "Probable cause pilot misjudged clearance over obstacles during final approach",
            "NTSB probable cause pilot misjudged clearance obstacles",
        ],
    },
    {
        "id": "Q25",
        "gold_id": "c8c383",  # Prefix used only for display
        "queries": [
            "What were the aircraft's departure point and destination?",
            "For NTSB accident NYC08CA055 involving Piper PA-22-150 registration N5921D, what were the aircraft's departure point and destination?",
            "Departure Point Destination",
        ],
    },
]


def main():
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    all_chunks = retrievers["chunks"]
    all_vectors = retrievers["vectors"]

    indices = [
        i for i, chunk in enumerate(all_chunks)
        if chunk.metadata.get("file_name") == TARGET_DOCUMENT
    ]

    if not indices:
        raise RuntimeError(f"Document not found: {TARGET_DOCUMENT}")

    chunks = [all_chunks[i] for i in indices]
    vectors = all_vectors[indices]

    print(f"Target document: {TARGET_DOCUMENT}")
    print(f"Chunks in target document: {len(chunks)}")

    for test in TESTS:
        print("\n" + "=" * 80)
        print(test["id"])

        # Resolve the exact gold chunk using the dataset.
        import json
        dataset = Path("evaluation/gold_questions.jsonl")

        with dataset.open(encoding="utf-8") as f:
            questions = [
                json.loads(line)
                for line in f if line.strip()
            ]

        question = next(
            q for q in questions
            if q["question_id"] == test["id"]
        )
        gold_ids = set(question["gold_chunk_ids"])

        for query_number, query in enumerate(test["queries"], start=1):
            print("\n" + "-" * 70)
            print(f"Query {query_number}: {query}")

            dense = dense_search(
                query=query,
                vectors=vectors,
                embeddings=retrievers["embeddings"],
                chunks=chunks,
                top_k=len(chunks),
            )

            bm25 = build_bm25_retriever(chunks)
            bm25.k = len(chunks)

            sparse = sparse_search(
                query=query,
                bm25_retriever=bm25,
                top_k=len(chunks),
            )

            hybrid = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=len(chunks),
            )

            reranked = rerank_candidates(query, hybrid)

            for label, results in [
                ("DENSE", dense),
                ("BM25", sparse),
                ("HYBRID RRF", hybrid),
                ("CROSS ENCODER", reranked),
            ]:
                ids = [
                    (
                        item["document"].metadata.get("chunk_id")
                        if isinstance(item, dict)
                        else item.metadata.get("chunk_id")
                    )
                    for item in results
                ]

                ranks = [
                    ids.index(gold_id) + 1
                    for gold_id in gold_ids
                    if gold_id in ids
                ]

                print(
                    f"\n{label}: "
                    f"gold_rank={min(ranks) if ranks else 'NOT FOUND'}"
                )

                for rank, item in enumerate(results[:5], start=1):
                    document = (
                        item["document"]
                        if isinstance(item, dict)
                        else item
                    )

                    metadata = document.metadata
                    score = (
                        item.get("rerank_score")
                        if isinstance(item, dict)
                        else item.get("rrf_score")
                        if isinstance(item, dict)
                        else None
                    )

                    print(
                        f"  {rank}. chunk={metadata.get('chunk_index')} "
                        f"page={metadata.get('page')} "
                        f"gold={metadata.get('chunk_id') in gold_ids} "
                        f"score={score}"
                    )

                    print(
                        "     ",
                        " ".join(document.page_content.split())[:220]
                    )


if __name__ == "__main__":
    main()
