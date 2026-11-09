import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]

GOLD_FILE = BACKEND / "evaluation" / "phase7_regression_questions.jsonl"
PHASE5_FILE = BACKEND / "evaluation" / "phase5c_answers.jsonl"

snapshot_id = (
    BACKEND / "index_store" / "current_snapshot"
).read_text(encoding="utf-8").strip()

CHUNKS_FILE = (
    BACKEND / "index_store" / "snapshots"
    / snapshot_id / "chunks.jsonl"
)

question_ids = {"Q02", "Q03", "Q04", "Q06", "Q07"}

gold_questions = {}
for line in GOLD_FILE.read_text(encoding="utf-8").splitlines():
    if line.strip():
        item = json.loads(line)
        if item["question_id"] in question_ids:
            gold_questions[item["question_id"]] = item

phase5_records = [
    json.loads(line)
    for line in PHASE5_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

chunk_lookup = {}

with CHUNKS_FILE.open(encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        metadata = item.get("metadata", {})
        chunk_id = item.get("chunk_id") or metadata.get("chunk_id")

        if chunk_id:
            chunk_lookup[chunk_id] = item

for qid in sorted(question_ids):
    question = gold_questions[qid]

    matches = [
        r for r in phase5_records
        if r["question_id"] == qid
        and r["strategy"] == "phase5_experimental"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Phase 5C record for {qid}; found {len(matches)}"
        )

    record = matches[0]
    retrieved = record.get("retrieved_evidence", [])

    print("\n" + "=" * 90)
    print("QUESTION:", qid)
    print("QUESTION TEXT:", question["question"])
    print("EXPECTED:", question["expected_answer"])

    retrieved_ids = {
        item.get("chunk_id") for item in retrieved
    }

    print("\n--- GOLD PASSAGES ---")

    for gold in question.get("gold_evidence", []):
        chunk_id = gold["chunk_id"]
        item = chunk_lookup.get(chunk_id)

        print("\nGold chunk:", chunk_id)
        print("Document:", gold.get("document"))
        print("Page:", gold.get("page"))
        print("Retrieved in Phase 5C:", chunk_id in retrieved_ids)

        if item is None:
            print("ERROR: Gold chunk not found in active snapshot.")
            continue

        metadata = item.get("metadata", {})
        content = item.get(
            "content",
            item.get("page_content", "")
        )

        print("Metadata:", metadata)
        print("CONTENT:\n", content)

    print("\n--- PHASE 5C RETRIEVED PASSAGES ---")

    for rank, evidence in enumerate(retrieved, start=1):
        print(
            f"\nRank {rank} | "
            f"{evidence.get('file_name')} | "
            f"Page {evidence.get('page')} | "
            f"Chunk {evidence.get('chunk_index')}"
        )
        print("Chunk ID:", evidence.get("chunk_id"))
        print("Cited:", evidence.get("cited"))
        print("CONTENT:\n", evidence.get("content", ""))

print("\nGold passage inspection completed.")
print("Snapshot:", snapshot_id)
