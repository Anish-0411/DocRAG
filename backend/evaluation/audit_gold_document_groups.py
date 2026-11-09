import json
from collections import defaultdict
from pathlib import Path

DATASET = Path("evaluation/gold_questions.jsonl")
SNAPSHOT_POINTER = Path("index_store/current_snapshot")
SNAPSHOT_ROOT = Path("index_store/snapshots")

snapshot_id = SNAPSHOT_POINTER.read_text(
    encoding="utf-8"
).strip()

chunks_path = SNAPSHOT_ROOT / snapshot_id / "chunks.jsonl"

if not chunks_path.exists():
    raise FileNotFoundError(chunks_path)

with DATASET.open(encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

chunk_lookup = {}

with chunks_path.open(encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        metadata = record["metadata"]
        chunk_id = metadata.get("chunk_id")

        if chunk_id:
            chunk_lookup[chunk_id] = metadata

groups = defaultdict(list)
missing = []

for question in questions:
    for chunk_id in question["gold_chunk_ids"]:
        metadata = chunk_lookup.get(chunk_id)

        if metadata is None:
            missing.append((question["question_id"], chunk_id))
            continue

        document = metadata.get("file_name", "UNKNOWN")

        groups[document].append({
            "question_id": question["question_id"],
            "chunk_index": metadata.get("chunk_index"),
            "page": metadata.get("page"),
            "document_id": metadata.get("document_id"),
        })

print("\nACTIVE SNAPSHOT:", snapshot_id)
print("TOTAL QUESTIONS:", len(questions))
print("UNIQUE GOLD DOCUMENTS:", len(groups))
print("MISSING GOLD CHUNKS:", len(missing))

print("\n" + "=" * 90)
print("QUESTIONS GROUPED BY GOLD DOCUMENT")
print("=" * 90)

for document, entries in sorted(
    groups.items(),
    key=lambda item: len(item[1]),
    reverse=True
):
    question_ids = sorted({
        entry["question_id"] for entry in entries
    })

    print(f"\nDOCUMENT: {document}")
    print("QUESTION COUNT:", len(question_ids))
    print("QUESTIONS:", ", ".join(question_ids))

    for entry in entries:
        print(
            f"  {entry['question_id']} | "
            f"chunk={entry['chunk_index']} | "
            f"page={entry['page']}"
        )

if missing:
    print("\nMISSING GOLD CHUNKS:")
    for question_id, chunk_id in missing:
        print(question_id, chunk_id)
