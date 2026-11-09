import csv
import json
from pathlib import Path

import tiktoken


BASE_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_ID = "snapshot_20261101T075233_ddc94958"

SNAPSHOT_DIR = BASE_DIR / "index_store" / "snapshots" / SNAPSHOT_ID
CHUNKS_FILE = SNAPSHOT_DIR / "chunks.jsonl"

RANKINGS_FILE = (
    BASE_DIR
    / "evaluation"
    / "results"
    / "phase3_candidates_20261103_171559"
    / "candidate_rankings.csv"
)

OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "phase3_neighbor_audit"
OUTPUT_FILE = OUTPUT_DIR / "neighbor_context_audit.csv"

TARGET_IDS = {
    "Q02", "Q03", "Q06", "Q07", "Q08",
    "Q23", "Q24", "Q25", "Q27", "Q36",
}

TOKEN_BUDGET = 5000
ENCODER = tiktoken.get_encoding("cl100k_base")


def load_chunks():
    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            item = json.loads(line)
            metadata = item.get("metadata", {})

            chunk_id = metadata.get("chunk_id") or item.get("id")
            if not chunk_id:
                continue

            item["_chunk_id"] = chunk_id
            item["_metadata"] = metadata
            item["_text"] = (
                item.get("page_content")
                or item.get("text")
                or item.get("content")
                or ""
            )

            chunks.append(item)

    return chunks


def load_top5_rankings():
    selected = {}

    with RANKINGS_FILE.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row.get("question_id") not in TARGET_IDS:
                continue

            if row.get("experiment") != "expanded_100_100_100":
                continue

            if row.get("stage") != "context_top5":
                continue

            selected.setdefault(row["question_id"], []).append(row)

    for question_id in selected:
        selected[question_id].sort(
            key=lambda row: int(row["rank"])
        )

    return selected


def token_count(text):
    return len(ENCODER.encode(text or ""))


def get_document_identity(metadata):
    return (
        metadata.get("document_id"),
        metadata.get("document_version"),
    )


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_FILE}")

    if not RANKINGS_FILE.exists():
        raise FileNotFoundError(f"Missing rankings file: {RANKINGS_FILE}")

    chunks = load_chunks()
    chunks_by_id = {chunk["_chunk_id"]: chunk for chunk in chunks}

    # Build ordered chunk groups for finding adjacent chunks.
    document_groups = {}

    for chunk in chunks:
        metadata = chunk["_metadata"]
        identity = get_document_identity(metadata)

        if identity == (None, None):
            continue

        chunk_index = metadata.get("chunk_index")
        if chunk_index is None:
            continue

        document_groups.setdefault(identity, []).append(chunk)

    for identity in document_groups:
        document_groups[identity].sort(
            key=lambda chunk: int(chunk["_metadata"]["chunk_index"])
        )

    rankings = load_top5_rankings()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_rows = []
    total_questions = 0
    total_baseline_chunks = 0
    total_expanded_chunks = 0

    for question_id in sorted(TARGET_IDS):
        seed_rows = rankings.get(question_id, [])

        if not seed_rows:
            print(f"WARNING: No top-five context found for {question_id}")
            continue

        total_questions += 1

        selected_chunks = []
        selected_ids = set()

        def add_chunk(chunk, selection_type, seed_rank=""):
            chunk_id = chunk["_chunk_id"]

            if chunk_id in selected_ids:
                return False

            selected_chunks.append({
                "chunk": chunk,
                "selection_type": selection_type,
                "seed_rank": seed_rank,
            })
            selected_ids.add(chunk_id)
            return True

        # Preserve original production-style top-five context first.
        for seed in seed_rows:
            chunk_id = seed["chunk_id"]
            chunk = chunks_by_id.get(chunk_id)

            if chunk:
                add_chunk(
                    chunk,
                    selection_type="original_top5",
                    seed_rank=seed["rank"],
                )

        baseline_count = len(selected_chunks)
        total_baseline_chunks += baseline_count

        # Add previous and next chunks around each original seed.
        for seed in seed_rows:
            seed_chunk = chunks_by_id.get(seed["chunk_id"])
            if not seed_chunk:
                continue

            metadata = seed_chunk["_metadata"]
            identity = get_document_identity(metadata)
            group = document_groups.get(identity, [])

            seed_index = metadata.get("chunk_index")
            if seed_index is None:
                continue

            position = next(
                (
                    i for i, candidate in enumerate(group)
                    if candidate["_chunk_id"] == seed_chunk["_chunk_id"]
                ),
                None,
            )

            if position is None:
                continue

            for neighbor_position in (position - 1, position + 1):
                if 0 <= neighbor_position < len(group):
                    neighbor = group[neighbor_position]
                    add_chunk(
                        neighbor,
                        selection_type="neighbor",
                        seed_rank=seed["rank"],
                    )

        # Apply the same 5,000-token budget used in the earlier experiment.
        final_chunks = []
        used_tokens = 0

        for item in selected_chunks:
            text = item["chunk"]["_text"]
            count = token_count(text)

            if used_tokens + count > TOKEN_BUDGET:
                continue

            final_chunks.append(item)
            used_tokens += count

        total_expanded_chunks += len(final_chunks)

        for final_rank, item in enumerate(final_chunks, start=1):
            chunk = item["chunk"]
            metadata = chunk["_metadata"]

            output_rows.append({
                "question_id": question_id,
                "final_context_rank": final_rank,
                "selection_type": item["selection_type"],
                "seed_rank": item["seed_rank"],
                "chunk_id": chunk["_chunk_id"],
                "document_id": metadata.get("document_id", ""),
                "document_version": metadata.get("document_version", ""),
                "document": (
                    metadata.get("source")
                    or metadata.get("document")
                    or metadata.get("filename")
                    or ""
                ),
                "page": metadata.get("page", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "token_count": token_count(chunk["_text"]),
                "text": chunk["_text"],
            })

    fieldnames = [
        "question_id",
        "final_context_rank",
        "selection_type",
        "seed_rank",
        "chunk_id",
        "document_id",
        "document_version",
        "document",
        "page",
        "chunk_index",
        "token_count",
        "text",
    ]

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\nPHASE 3 NEIGHBOR CONTEXT AUDIT COMPLETE")
    print(f"Questions audited: {total_questions}")
    print(f"Original top-five chunks: {total_baseline_chunks}")
    print(f"Expanded context chunks: {total_expanded_chunks}")
    print(f"Audit rows written: {len(output_rows)}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()