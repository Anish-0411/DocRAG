from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the backend root importable when this file is run directly.
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from retrieval import (
    build_retrievers,
    dense_search,
    get_chunk_identity,
    reciprocal_rank_fusion,
    sparse_search,
)
from reranker import rerank_candidates, select_context
from snapshot_manager import load_active_snapshot
from config import RRF_K


TARGET_IDS = None
GOLD_PATH = Path("evaluation/gold_questions.jsonl")
RESULTS_ROOT = Path("evaluation/results")

EXPERIMENTS = [
    {"name": "expanded_100_100_100", "dense": 100, "sparse": 100, "rrf": 100},
    {"name": "expanded_200_200_200", "dense": 200, "sparse": 200, "rrf": 200},
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("phase3")


def load_questions():
    questions = []
    with GOLD_PATH.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = __import__("json").loads(line)
                if TARGET_IDS is None or item["question_id"] in TARGET_IDS:
                    questions.append(item)

    if not questions:
        raise ValueError("No questions found in gold dataset")

    return sorted(questions, key=lambda item: item["question_id"])


def metadata_value(document, *keys):
    for key in keys:
        value = document.metadata.get(key)
        if value is not None:
            return value
    return ""


def main():
    started = time.perf_counter()
    questions = load_questions()

    logger.info("Loading active snapshot")
    snapshot = load_active_snapshot()

    logger.info(
        "Snapshot=%s chunks=%d vectors=%s",
        snapshot["snapshot_id"],
        len(snapshot["chunks"]),
        snapshot["vectors"].shape,
    )

    retrievers = build_retrievers(snapshot)
    rows = []

    for experiment in EXPERIMENTS:
        name = experiment["name"]
        logger.info("Starting experiment: %s", name)

        retrievers["bm25_retriever"].k = experiment["sparse"]

        for question in questions:
            qid = question["question_id"]
            query = question["question"]
            gold_ids = set(question["gold_chunk_ids"])

            dense = dense_search(
                query=query,
                vectors=retrievers["vectors"],
                embeddings=retrievers["embeddings"],
                chunks=retrievers["chunks"],
                top_k=experiment["dense"],
            )

            sparse = sparse_search(
                query=query,
                bm25_retriever=retrievers["bm25_retriever"],
                top_k=experiment["sparse"],
            )

            candidates = reciprocal_rank_fusion(
                dense_results=dense,
                sparse_results=sparse,
                rrf_k=RRF_K,
                top_k=experiment["rrf"],
            )

            reranked = rerank_candidates(
                query=query,
                candidates=candidates,
            )

            selected = select_context(
                reranked_candidates=reranked,
                top_k=5,
                max_tokens=5000,
            )

            for stage, results in (
                ("dense", dense),
                ("bm25", sparse),
                ("rrf", candidates),
                ("reranked", reranked),
                ("context_top5", selected),
            ):
                for rank, result in enumerate(results, start=1):
                    if stage in ("dense", "bm25"):
                        document = result
                        chunk_id = get_chunk_identity(result)
                        rerank_score = ""
                        rrf_score = ""
                    else:
                        document = result["document"]
                        chunk_id = result["chunk_id"]
                        rerank_score = result.get("rerank_score", "")
                        rrf_score = result.get("rrf_score", "")

                    metadata = document.metadata

                    rows.append({
                        "experiment": name,
                        "question_id": qid,
                        "stage": stage,
                        "rank": rank,
                        "chunk_id": chunk_id,
                        "is_gold_chunk": chunk_id in gold_ids,
                        "document": metadata_value(
                            document, "file_name", "source"
                        ),
                        "page": metadata_value(document, "page"),
                        "rerank_score": rerank_score,
                        "rrf_score": rrf_score,
                        "text_preview": document.page_content[:350].replace(
                            "\n", " "
                        ),
                    })

            gold_ranks = [
                rank
                for rank, result in enumerate(reranked, start=1)
                if result["chunk_id"] in gold_ids
            ]

            selected_gold = sum(
                result["chunk_id"] in gold_ids for result in selected
            )

            logger.info(
                "%s | %s | candidates=%d | gold reranked ranks=%s | "
                "gold in context=%d",
                name,
                qid,
                len(candidates),
                gold_ranks or "NOT FOUND",
                selected_gold,
            )

    output_dir = RESULTS_ROOT / (
        "phase3_candidates_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    output_file = output_dir / "candidate_rankings.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.perf_counter() - started

    print("\nPHASE 3A CANDIDATE EXPERIMENT COMPLETE")
    print("Snapshot:", snapshot["snapshot_id"])
    print("Questions:", len(questions))
    print("Experiments:", len(EXPERIMENTS))
    print("Rows:", len(rows))
    print("Runtime seconds:", round(elapsed, 2))
    print("Results:", output_file)


if __name__ == "__main__":
    main()
