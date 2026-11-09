import csv
import json
import sys
from datetime import datetime
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

DATASET = Path("evaluation/phase7_regression_questions.jsonl")
TARGET_IDS = {"Q02", "Q03", "Q04", "Q06", "Q07"}

OUTPUT = Path("evaluation/results") / (
    f"phase7h_competitor_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
            and json.loads(line)["question_id"] in TARGET_IDS
        ]


def describe(item, rank, stage, qid, gold_ids):
    if stage == "RRF":
        doc = item["document"]
        chunk_id = item["chunk_id"]
        score = item["rrf_score"]
        sources = ",".join(item["retrieved_by"])
        rerank_score = None
    else:
        doc = item["document"]
        chunk_id = item["chunk_id"]
        score = item["rerank_score"]
        sources = ",".join(item["retrieved_by"])
        rerank_score = score

    metadata = doc.metadata

    return {
        "question_id": qid,
        "stage": stage,
        "rank": rank,
        "is_gold": chunk_id in gold_ids,
        "chunk_id": chunk_id,
        "file_name": metadata.get("file_name"),
        "page": metadata.get("page"),
        "document_id": metadata.get("document_id"),
        "retrieved_by": sources,
        "rrf_score": item.get("rrf_score"),
        "rerank_score": rerank_score,
        "text_excerpt": doc.page_content[:1200].replace("\n", " "),
    }


def main():
    questions = load_questions()
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    bm25 = retrievers["bm25_retriever"]
    original_k = bm25.k
    bm25.k = 100

    rows = []

    try:
        for q in questions:
            qid = q["question_id"]
            gold_ids = set(q.get("gold_chunk_ids", []))

            dense = dense_search(
                q["question"],
                retrievers["vectors"],
                retrievers["embeddings"],
                retrievers["chunks"],
                top_k=100,
            )

            sparse = sparse_search(
                q["question"],
                bm25,
                top_k=100,
            )

            fused = reciprocal_rank_fusion(
                dense,
                sparse,
                rrf_k=60,
                top_k=200,
            )

            reranked = rerank_candidates(
                q["question"],
                fused,
            )

            for stage, items in (
                ("RRF", fused[:10]),
                ("RERANKED", reranked[:10]),
            ):
                for rank, item in enumerate(items, start=1):
                    row = describe(
                        item, rank, stage, qid, gold_ids
                    )
                    rows.append(row)

                    marker = " GOLD" if row["is_gold"] else ""
                    print(
                        f"\n{qid} {stage} #{rank}{marker}"
                        f"\n  {row['file_name']} | page={row['page']}"
                        f"\n  chunk={row['chunk_id'][:16]}"
                        f"\n  retrieved_by={row['retrieved_by']}"
                        f"\n  RRF={row['rrf_score']}"
                        f" | rerank={row['rerank_score']}"
                        f"\n  {row['text_excerpt'][:450]}"
                    )

            # Also print gold passages wherever they appear in reranked list.
            for rank, item in enumerate(reranked, start=1):
                if item["chunk_id"] in gold_ids and rank > 10:
                    row = describe(
                        item, rank, "RERANKED_GOLD", qid, gold_ids
                    )
                    rows.append(row)
                    print(
                        f"\n{qid} GOLD RERANKED #{rank}"
                        f"\n  {row['file_name']} | page={row['page']}"
                        f"\n  RRF={row['rrf_score']}"
                        f" | rerank={row['rerank_score']}"
                        f"\n  {row['text_excerpt'][:450]}"
                    )

    finally:
        bm25.k = original_k

    output_file = OUTPUT / "competitor_audit.csv"
    fields = list(rows[0].keys()) if rows else []

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAudit saved:", output_file)


if __name__ == "__main__":
    main()
