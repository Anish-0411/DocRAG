import csv
import json
import math
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
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


BASE = Path("evaluation")
DATASET = BASE / "gold_questions.jsonl"

OUTPUT = BASE / "results" / (
    f"phase3g_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

RRF_K = 60
GLOBAL_DEPTH = 100
DOCUMENT_COUNT = 5
LOCAL_DENSE_K = 10
LOCAL_SPARSE_K = 10
LOCAL_RRF_TOP_K = 20

TOP_K_VALUES = (5, 10, 20)

STRATEGIES = (
    "baseline_hybrid_reranked",
    "depth100_hybrid_reranked",
    "document_scoped_reranked",
)


def load_questions():
    with DATASET.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def document_key(document):
    metadata = document.metadata
    return (
        str(metadata.get("document_id", "unknown")),
        str(metadata.get("document_version", "unknown")),
    )


def document_name(document):
    return document.metadata.get("file_name", "Unknown")


def chunk_id(item):
    return item["document"].metadata.get("chunk_id")


def ranked_ids(items):
    return [chunk_id(item) for item in items]


def recall_at_k(ids, gold, k):
    return len(set(ids[:k]) & gold) / len(gold) if gold else 0.0


def precision_at_k(ids, gold, k):
    return len(set(ids[:k]) & gold) / k if k else 0.0


def reciprocal_rank(ids, gold, k=20):
    for rank, item_id in enumerate(ids[:k], start=1):
        if item_id in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ids, gold, k):
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item_id in enumerate(ids[:k], start=1)
        if item_id in gold
    )

    ideal_count = min(len(gold), k)
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / ideal if ideal else 0.0


def calculate_metrics(ids, gold):
    metrics = {}

    for k in TOP_K_VALUES:
        metrics[f"recall@{k}"] = recall_at_k(ids, gold, k)
        metrics[f"precision@{k}"] = precision_at_k(ids, gold, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ids, gold, k)

    metrics["mrr@20"] = reciprocal_rank(ids, gold, 20)
    metrics["hit@20"] = int(bool(set(ids[:20]) & gold))

    return metrics


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
            "document_id": key[0],
            "document_version": key[1],
            "name": names[key],
            "score": score,
        })

    return sorted(
        ranked,
        key=lambda item: item["score"],
        reverse=True,
    )


def gold_document_keys(gold_ids, chunk_lookup):
    result = set()

    for gold_id in gold_ids:
        document = chunk_lookup.get(gold_id)

        if document is not None:
            result.add(document_key(document))

    return result


def retrieve_global(query, retrievers, depth, rrf_top_k):
    bm25 = retrievers["bm25_retriever"]
    original_k = bm25.k

    try:
        dense = dense_search(
            query=query,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=depth,
        )

        bm25.k = depth

        sparse = sparse_search(
            query=query,
            bm25_retriever=bm25,
            top_k=depth,
        )

    finally:
        bm25.k = original_k

    return reciprocal_rank_fusion(
        dense_results=dense,
        sparse_results=sparse,
        rrf_k=RRF_K,
        top_k=rrf_top_k,
    )


def retrieve_document_scoped(
    query,
    retrievers,
    document_indices,
):
    chunks = retrievers["chunks"]
    vectors = retrievers["vectors"]

    # Stage 1: identify likely source documents globally.
    global_hybrid = retrieve_global(
        query,
        retrievers,
        depth=GLOBAL_DEPTH,
        rrf_top_k=GLOBAL_DEPTH,
    )

    ranked_documents = aggregate_documents(global_hybrid)
    selected_documents = ranked_documents[:DOCUMENT_COUNT]

    # Stage 2: retrieve passages independently in each selected document.
    merged_candidates = {}

    for selected in selected_documents:
        key = (
            selected["document_id"],
            selected["document_version"],
        )

        indices = document_indices.get(key, [])

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
            rrf_k=RRF_K,
            top_k=LOCAL_RRF_TOP_K,
        )

        for item in local_hybrid:
            identifier = chunk_id(item)

            if not identifier:
                continue

            previous = merged_candidates.get(identifier)

            if (
                previous is None
                or item["rrf_score"] > previous["rrf_score"]
            ):
                merged_candidates[identifier] = item

    candidate_pool = list(merged_candidates.values())

    # Stage 3: rerank the combined local candidate pool.
    reranked = rerank_candidates(query, candidate_pool)

    return ranked_documents, selected_documents, candidate_pool, reranked


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    questions = load_questions()

    if len(questions) != 40:
        raise RuntimeError(
            f"Expected 40 questions, found {len(questions)}"
        )

    print("Loading active snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    chunks = retrievers["chunks"]
    vectors = retrievers["vectors"]

    if len(chunks) != len(vectors):
        raise RuntimeError("Snapshot chunk/vector count mismatch")

    print("Snapshot:", snapshot["snapshot_id"])
    print("Chunks:", len(chunks))
    print("Questions:", len(questions))

    # Build document-to-chunk index once.
    document_indices = defaultdict(list)
    chunk_lookup = {}

    for index, chunk in enumerate(chunks):
        document_indices[document_key(chunk)].append(index)

        identifier = chunk.metadata.get("chunk_id")
        if identifier:
            chunk_lookup[identifier] = chunk

    totals = {
        strategy: Counter()
        for strategy in STRATEGIES
    }

    rows = []
    document_rows = []
    started_all = time.perf_counter()

    for number, question in enumerate(questions, start=1):
        qid = question["question_id"]
        query = question["question"]
        gold = set(question["gold_chunk_ids"])

        missing = gold - set(chunk_lookup)

        if missing:
            raise RuntimeError(
                f"{qid}: gold chunks missing from active snapshot: "
                f"{sorted(missing)}"
            )

        gold_docs = gold_document_keys(gold, chunk_lookup)

        print(f"\n[{number}/{len(questions)}] {qid}: {query}")

        # --------------------------------------------------
        # A: Existing production retrieval configuration.
        # --------------------------------------------------
        start = time.perf_counter()

        baseline_hybrid = retrieve_global(
            query,
            retrievers,
            depth=15,
            rrf_top_k=20,
        )

        baseline_reranked = rerank_candidates(
            query,
            baseline_hybrid,
        )

        baseline_time = time.perf_counter() - start

        # --------------------------------------------------
        # B: Global depth-100 retrieval.
        # --------------------------------------------------
        start = time.perf_counter()

        depth100_hybrid = retrieve_global(
            query,
            retrievers,
            depth=100,
            rrf_top_k=100,
        )

        depth100_reranked = rerank_candidates(
            query,
            depth100_hybrid,
        )

        depth100_time = time.perf_counter() - start

        # --------------------------------------------------
        # C: Document-scoped retrieval.
        # --------------------------------------------------
        start = time.perf_counter()

        ranked_documents, selected_documents, local_pool, scoped_reranked = (
            retrieve_document_scoped(
                query,
                retrievers,
                document_indices,
            )
        )

        scoped_time = time.perf_counter() - start

        result_sets = {
            "baseline_hybrid_reranked": (
                baseline_reranked,
                baseline_time,
            ),
            "depth100_hybrid_reranked": (
                depth100_reranked,
                depth100_time,
            ),
            "document_scoped_reranked": (
                scoped_reranked,
                scoped_time,
            ),
        }

        # Document-level diagnostic.
        document_ranks = {
            (item["document_id"], item["document_version"]): rank
            for rank, item in enumerate(ranked_documents, start=1)
        }

        best_gold_document_rank = min(
            (
                document_ranks[key]
                for key in gold_docs
                if key in document_ranks
            ),
            default=None,
        )

        selected_doc_keys = {
            (item["document_id"], item["document_version"])
            for item in selected_documents
        }

        gold_document_selected = bool(
            gold_docs & selected_doc_keys
        )

        document_rows.append({
            "question_id": qid,
            "gold_document_rank": best_gold_document_rank,
            "gold_document_selected_top5": int(
                gold_document_selected
            ),
            "selected_documents": " | ".join(
                item["name"] for item in selected_documents
            ),
            "candidate_count": len(local_pool),
        })

        # Passage-level metrics.
        for strategy, (candidates, elapsed) in result_sets.items():
            ids = ranked_ids(candidates)
            metrics = calculate_metrics(ids, gold)

            for metric, value in metrics.items():
                totals[strategy][metric] += value

            found_gold = set(ids) & gold

            rows.append({
                "question_id": qid,
                "strategy": strategy,
                "candidate_count": len(candidates),
                "gold_count": len(gold),
                "gold_found": len(found_gold),
                "gold_ranks": ";".join(
                    str(ids.index(gold_id) + 1)
                    for gold_id in sorted(found_gold)
                ),
                "runtime_seconds": round(elapsed, 3),
                **metrics,
            })

            print(
                f"  {strategy}: "
                f"Recall@20={metrics['recall@20']:.3f}, "
                f"MRR@20={metrics['mrr@20']:.3f}, "
                f"candidates={len(candidates)}, "
                f"time={elapsed:.2f}s"
            )

        print(
            "  Gold document rank:",
            best_gold_document_rank or "NOT FOUND",
            "| Selected in top 5:",
            gold_document_selected,
        )

    # ------------------------------------------------------
    # Save results.
    # ------------------------------------------------------
    csv_path = OUTPUT / "per_question_metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    document_csv = OUTPUT / "document_selection_metrics.csv"

    with document_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(document_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(document_rows)

    summary = {}

    for strategy in STRATEGIES:
        summary[strategy] = {
            metric: round(value / len(questions), 4)
            for metric, value in totals[strategy].items()
        }

    selected_count = sum(
        row["gold_document_selected_top5"]
        for row in document_rows
    )

    summary["document_selection"] = {
        "questions": len(document_rows),
        "gold_document_selected_top5": selected_count,
        "gold_document_selection_rate": round(
            selected_count / len(document_rows), 4
        ),
        "mean_gold_document_rank": round(
            sum(
                row["gold_document_rank"] or 0
                for row in document_rows
            ) / len(document_rows),
            4,
        ),
    }

    summary["experiment"] = {
        "snapshot_id": snapshot["snapshot_id"],
        "question_count": len(questions),
        "document_count": DOCUMENT_COUNT,
        "global_depth": GLOBAL_DEPTH,
        "local_dense_k": LOCAL_DENSE_K,
        "local_sparse_k": LOCAL_SPARSE_K,
        "local_rrf_top_k": LOCAL_RRF_TOP_K,
        "total_runtime_seconds": round(
            time.perf_counter() - started_all, 2
        ),
    }

    summary_path = OUTPUT / "summary.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE 3G RESULTS")
    print("=" * 80)

    for strategy in STRATEGIES:
        print(f"\n{strategy}")

        for metric, value in summary[strategy].items():
            print(f"  {metric}: {value}")

    print("\nDocument selection:")
    print(json.dumps(summary["document_selection"], indent=2))

    print("\nTotal runtime:", summary["experiment"]["total_runtime_seconds"])
    print("\nOutput directory:", OUTPUT)
    print("Summary:", summary_path)
    print("Passage metrics:", csv_path)
    print("Document metrics:", document_csv)


if __name__ == "__main__":
    main()
