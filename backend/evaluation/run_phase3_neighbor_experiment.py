import csv
import json
from pathlib import Path
from collections import defaultdict

import tiktoken

SNAPSHOT = Path(
    "index_store/snapshots/snapshot_20261101T075233_ddc94958"
)
GOLD_PATH = Path("evaluation/gold_questions.jsonl")
RANKINGS_PATH = Path(
    "evaluation/results/phase3_candidates_20261103_171559/candidate_rankings.csv"
)
OUTPUT_DIR = Path("evaluation/results/phase3_neighbor_experiment")

EXPERIMENT = "expanded_100_100_100"
TOKEN_BUDGET = 5000
NEIGHBOR_OFFSETS = (-1, 1)

encoder = tiktoken.get_encoding("cl100k_base")


def token_count(text):
    return len(encoder.encode(text or ""))


def load_gold():
    result = {}
    with GOLD_PATH.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                result[item["question_id"]] = item
    return result


def load_chunks():
    chunks = {}
    by_document = defaultdict(dict)

    with (SNAPSHOT / "chunks.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)
            metadata = item.get("metadata", {})
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue

            chunks[chunk_id] = item

            document_key = (
                metadata.get("document_id"),
                metadata.get("document_version"),
            )
            chunk_index = metadata.get("chunk_index")

            if chunk_index is not None:
                by_document[document_key][int(chunk_index)] = chunk_id

    return chunks, by_document


def load_rankings():
    result = defaultdict(list)

    with RANKINGS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == EXPERIMENT:
                result[(row["question_id"], row["stage"])].append(row)

    for rows in result.values():
        rows.sort(key=lambda row: int(row["rank"]))

    return result


def get_neighbors(chunk, chunks, by_document):
    metadata = chunk["metadata"]
    key = (
        metadata.get("document_id"),
        metadata.get("document_version"),
    )
    index = metadata.get("chunk_index")

    if index is None or key not in by_document:
        return []

    neighbors = []
    document_chunks = by_document[key]

    for offset in NEIGHBOR_OFFSETS:
        neighbor_id = document_chunks.get(int(index) + offset)
        if neighbor_id and neighbor_id in chunks:
            neighbors.append(neighbor_id)

    return neighbors


def build_context(seed_ids, chunks, by_document):
    selected = []
    selected_set = set()
    token_total = 0

    def add(chunk_id):
        nonlocal token_total

        if chunk_id in selected_set or chunk_id not in chunks:
            return False

        content = chunks[chunk_id].get("page_content", "")
        cost = token_count(content)

        if token_total + cost > TOKEN_BUDGET:
            return False

        selected.append(chunk_id)
        selected_set.add(chunk_id)
        token_total += cost
        return True

    # Preserve the existing selected context first.
    for chunk_id in seed_ids:
        add(chunk_id)

    # Then add adjacent chunks around those selected seeds.
    neighbor_candidates = []
    for seed_id in seed_ids:
        chunk = chunks.get(seed_id)
        if not chunk:
            continue

        for neighbor_id in get_neighbors(chunk, chunks, by_document):
            if neighbor_id not in selected_set:
                neighbor_candidates.append(neighbor_id)

    for neighbor_id in neighbor_candidates:
        add(neighbor_id)

    return selected, token_total


def main():
    gold = load_gold()
    chunks, by_document = load_chunks()
    rankings = load_rankings()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_rows = []
    baseline_covered = 0
    expanded_covered = 0
    expanded_counts = []

    for qid in sorted(gold):
        gold_ids = set(gold[qid]["gold_chunk_ids"])

        baseline_rows = rankings.get((qid, "context_top5"), [])
        seed_ids = [row["chunk_id"] for row in baseline_rows]

        expanded_ids, tokens = build_context(
            seed_ids, chunks, by_document
        )

        baseline_match = bool(set(seed_ids) & gold_ids)
        expanded_match = bool(set(expanded_ids) & gold_ids)

        baseline_covered += int(baseline_match)
        expanded_covered += int(expanded_match)
        expanded_counts.append(len(expanded_ids))

        matched_gold = sorted(set(expanded_ids) & gold_ids)

        report_rows.append({
            "question_id": qid,
            "baseline_context_chunks": len(seed_ids),
            "baseline_gold_coverage": baseline_match,
            "expanded_context_chunks": len(expanded_ids),
            "expanded_gold_coverage": expanded_match,
            "new_gold_recovered": expanded_match and not baseline_match,
            "gold_chunk_ids_found": ";".join(matched_gold),
            "context_tokens": tokens,
            "added_neighbor_chunks": len(set(expanded_ids) - set(seed_ids)),
        })

    output_file = OUTPUT_DIR / "neighbor_experiment_per_question.csv"

    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(report_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(report_rows)

    total = len(report_rows)

    print("\nPHASE 3B — NEIGHBOR EXPANSION COMPLETE")
    print("Snapshot:", SNAPSHOT.name)
    print("Questions:", total)
    print("Token budget:", TOKEN_BUDGET)
    print()
    print(
        f"Baseline exact gold coverage: "
        f"{baseline_covered}/{total} = {baseline_covered / total:.2%}"
    )
    print(
        f"Expanded exact gold coverage: "
        f"{expanded_covered}/{total} = {expanded_covered / total:.2%}"
    )
    print(
        "New questions recovered:",
        sum(row["new_gold_recovered"] for row in report_rows)
    )
    print(
        "Average expanded context chunks:",
        round(sum(expanded_counts) / total, 2)
    )
    print(
        "Average context tokens:",
        round(sum(row["context_tokens"] for row in report_rows) / total, 2)
    )
    print("\nPer-question coverage:")
    for row in report_rows:
        print(
            row["question_id"],
            "| baseline:", row["baseline_gold_coverage"],
            "| expanded:", row["expanded_gold_coverage"],
            "| added:", row["added_neighbor_chunks"],
            "| tokens:", row["context_tokens"],
        )

    print("\nResults:", output_file)


if __name__ == "__main__":
    main()
