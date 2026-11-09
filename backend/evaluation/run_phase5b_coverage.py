import json
import math
import re
import time
import csv
from collections import Counter
from pathlib import Path
import sys

# Make backend modules importable when this script is run from evaluation/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from snapshot_manager import load_active_snapshot
from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)
from reranker import get_reranker

STOPWORDS = {
    "the", "and", "for", "from", "with", "that", "this", "what",
    "when", "where", "which", "who", "whom", "how", "why",
    "was", "were", "are", "is", "has", "have", "had", "does",
    "did", "can", "could", "would", "should", "about", "into",
    "over", "under", "their", "there", "they", "them", "its",
    "give", "list", "describe", "explain", "identify", "according",
    "document", "documents", "following", "provided"
}

def tokens(text):
    return {
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    }

def metrics(selected_ids, gold_ids):
    gold_ids = set(gold_ids)
    found = sum(x in gold_ids for x in selected_ids)

    dcg = sum(
        1 / math.log2(i + 2)
        for i, x in enumerate(selected_ids)
        if x in gold_ids
    )
    ideal = sum(
        1 / math.log2(i + 2)
        for i in range(min(len(gold_ids), 5))
    )

    return {
        "recall": found / len(gold_ids),
        "precision": found / 5,
        "ndcg": dcg / ideal if ideal else 0,
        "hit": int(found > 0),
    }

def coverage_select(ranked_candidates, query, k=5):
    query_terms = tokens(query)
    selected = []
    covered_terms = set()
    remaining = list(range(len(ranked_candidates)))

    while remaining and len(selected) < k:
        best_index = None
        best_score = float("-inf")

        for index in remaining:
            doc = ranked_candidates[index]["document"]
            content_terms = tokens(doc.page_content)

            new_terms = (content_terms & query_terms) - covered_terms
            gain = (
                len(new_terms) / len(query_terms)
                if query_terms else 0
            )

            # Preserve relevance while rewarding new query-term coverage.
            relevance = 1 / (index + 1)
            score = 0.85 * relevance + 0.15 * gain

            if score > best_score:
                best_score = score
                best_index = index

        selected.append(ranked_candidates[best_index])
        doc = ranked_candidates[best_index]["document"]
        covered_terms |= tokens(doc.page_content) & query_terms
        remaining.remove(best_index)

    return selected

def rank_candidates(model, query, candidates, use_metadata):
    pairs = []

    for item in candidates:
        doc = item["document"]

        if use_metadata:
            filename = doc.metadata.get("file_name", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            text = (
                f"Source document: {filename}\n"
                f"Page: {page}\n"
                f"Content:\n{doc.page_content}"
            )
        else:
            text = doc.page_content

        pairs.append((query, text))

    scores = model.predict(pairs, show_progress_bar=False)
    order = np.argsort(-np.asarray(scores))

    return [candidates[int(i)] for i in order]

# --------------------------------------------------
# Load benchmark and active snapshot
# --------------------------------------------------

with open("evaluation/gold_questions.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

snapshot = load_active_snapshot()
retrievers = build_retrievers(snapshot)
model = get_reranker()

bm25 = retrievers["bm25_retriever"]
original_bm25_k = bm25.k

totals = {
    "baseline": Counter(),
    "metadata": Counter(),
    "coverage": Counter(),
}

rows = []
start = time.time()

try:
    bm25.k = 100

    for number, q in enumerate(questions, 1):
        query = q["question"]
        gold_ids = set(q["gold_chunk_ids"])

        dense = dense_search(
            query=query,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=100,
        )

        sparse = sparse_search(query, bm25, top_k=100)

        candidates = reciprocal_rank_fusion(
            dense_results=dense,
            sparse_results=sparse,
            rrf_k=60,
            top_k=100,
        )

        if not candidates:
            continue

        baseline_ranked = rank_candidates(
            model, query, candidates, False
        )
        metadata_ranked = rank_candidates(
            model, query, candidates, True
        )

        selections = {
            "baseline": baseline_ranked[:5],
            "metadata": metadata_ranked[:5],
            "coverage": coverage_select(metadata_ranked, query, 5),
        }

        row = {"question_id": q["question_id"]}

        for strategy, selected in selections.items():
            ids = [
                item["document"].metadata.get("chunk_id")
                for item in selected
            ]

            result = metrics(ids, gold_ids)

            for key, value in result.items():
                totals[strategy][key] += value

            row[f"{strategy}_gold_chunks"] = sum(
                chunk_id in gold_ids for chunk_id in ids
            )

            row[f"{strategy}_selected_ids"] = "|".join(
                str(x) for x in ids
            )

        rows.append(row)

        if number % 5 == 0:
            print(f"Completed {number}/{len(questions)}")

finally:
    bm25.k = original_bm25_k

# --------------------------------------------------
# Save results
# --------------------------------------------------

output_dir = Path("evaluation/results") / (
    "phase5b_coverage_" + time.strftime("%Y%m%d_%H%M%S")
)
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "per_question.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== PHASE 5B RESULTS ==========")
print("Questions:", len(rows))

for strategy, values in totals.items():
    print(f"\n{strategy.upper()}")
    for key in ("recall", "precision", "ndcg", "hit"):
        print(f"{key}@5: {values[key] / len(rows):.4f}")

print("\n========== QUESTIONS WHERE COVERAGE CHANGED ==========")

for row in rows:
    if (
        row["metadata_selected_ids"]
        != row["coverage_selected_ids"]
    ):
        print(
            row["question_id"],
            "| metadata gold:", row["metadata_gold_chunks"],
            "| coverage gold:", row["coverage_gold_chunks"]
        )

print("\nSaved to:", output_dir)
print("Runtime:", round(time.time() - start, 2), "seconds")
