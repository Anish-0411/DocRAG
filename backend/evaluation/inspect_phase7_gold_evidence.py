import json
from pathlib import Path

GOLD_FILE = Path("evaluation/gold_questions.jsonl")
PHASE5_FILE = Path("evaluation/phase5c_answers.jsonl")
POINTER = Path("index_store/current_snapshot")

snapshot_id = POINTER.read_text().strip()
CHUNKS_FILE = Path("index_store/snapshots") / snapshot_id / "chunks.jsonl"

gold_questions = {
    json.loads(line)["question_id"]: json.loads(line)
    for line in GOLD_FILE.read_text().splitlines()
    if line.strip()
}

records = [
    json.loads(line)
    for line in PHASE5_FILE.read_text().splitlines()
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

for qid in ["Q36", "Q37"]:
    question = gold_questions[qid]
    record = next(
        r for r in records
        if r["question_id"] == qid
        and r["strategy"] == "phase5_experimental"
    )

    retrieved_ids = {
        e.get("chunk_id")
        for e in record.get("retrieved_evidence", [])
    }

    print("\n" + "=" * 85)
    print("QUESTION:", qid)
    print(question["question"])
    print("Expected:", question["expected_answer"])

    print("\nGOLD CHUNKS:")
    for gold in question["gold_chunk_ids"]:
        print("\nGold chunk ID:", gold)
        print("Retrieved in Phase 5C:", gold in retrieved_ids)

        item = chunk_lookup.get(gold)
        if item is None:
            print("Chunk not found in active snapshot.")
            continue

        metadata = item.get("metadata", {})
        content = item.get("content", item.get("page_content", ""))

        print("Document:", metadata.get("file_name", metadata.get("source")))
        print("Page:", metadata.get("page"))
        print("Chunk index:", metadata.get("chunk_index"))
        print("Content:\n", content)

    print("\nRETRIEVED EVIDENCE CHUNK IDS:")
    for e in record.get("retrieved_evidence", []):
        print(
            e.get("chunk_id"),
            "|",
            e.get("file_name"),
            "| page",
            e.get("page")
        )
