
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

BASE_DIR = Path(__file__).resolve().parent

GOLD_DATASET = BASE_DIR / "gold_questions.jsonl"
DATASET = BASE_DIR / "phase2_answers.jsonl"

BACKUP = BASE_DIR / "phase2_answers.pre_40q_expansion.jsonl"

API_URL = "http://127.0.0.1:8000/ask"
TIMEOUT = 180
DELAY_SECONDS = 1


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl_atomic(path, records):
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    temporary_path.replace(path)


if not GOLD_DATASET.exists():
    raise FileNotFoundError(GOLD_DATASET)

if not DATASET.exists():
    raise FileNotFoundError(DATASET)


# Preserve the existing manually evaluated pilot.
if not BACKUP.exists():
    BACKUP.write_text(
        DATASET.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

existing_records = load_jsonl(DATASET)
gold_records = load_jsonl(GOLD_DATASET)

existing_by_id = {
    record["question_id"]: record
    for record in existing_records
}

if len(existing_by_id) != len(existing_records):
    raise ValueError("Duplicate question IDs in existing answer dataset.")

if len(gold_records) != 40:
    raise ValueError(
        f"Expected 40 gold questions, found {len(gold_records)}."
    )

records = []
added = 0

# Preserve existing records and append missing gold questions.
for question in gold_records:
    question_id = question["question_id"]

    if question_id in existing_by_id:
        records.append(existing_by_id[question_id])
        continue

    record = {
        "question_id": question_id,
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "category": question.get("category"),
        "gold_evidence": question.get("gold_evidence", []),
        "answer": "",
        "citations": [],
        "retrieved_evidence": [],
        "retrieval_metadata": {},
        "human_evaluation": {
            "correctness": None,
            "faithfulness": None,
            "answer_relevance": None,
            "citation_support": None,
            "abstention_appropriate": None
        },
        "evaluation_notes": ""
    }

    records.append(record)
    added += 1


# Reject records not present in the gold dataset.
gold_ids = {q["question_id"] for q in gold_records}
unexpected_ids = set(existing_by_id) - gold_ids

if unexpected_ids:
    raise ValueError(
        f"Unexpected question IDs in answer dataset: {sorted(unexpected_ids)}"
    )

if len(records) != 40:
    raise ValueError(f"Expected 40 records, found {len(records)}.")

# Save the expanded dataset before making API requests.
save_jsonl_atomic(DATASET, records)

print("Phase 2 dataset prepared.")
print("Existing records preserved:", len(existing_records))
print("New records appended:", added)
print("Total questions:", len(records))
print("Backup:", BACKUP)

# Capture only unanswered questions.
completed = 0
failed = []

for record in records:
    question_id = record["question_id"]

    if record.get("answer"):
        print(f"Skipping {question_id}: answer already captured")
        continue

    # Never recapture the already manually evaluated pilot.
    if question_id in {f"Q{i:02d}" for i in range(1, 11)}:
        print(f"Skipping {question_id}: pilot record requires preservation")
        continue

    payload = {
        "question": record["question"],
        "session_id": f"phase2-full-{question_id}"
    }

    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"\nCapturing {question_id}: {record['question']}")

    try:
        with urlopen(request, timeout=TIMEOUT) as response:
            result = json.loads(response.read().decode("utf-8"))

        answer = result.get("answer")
        citations = result.get("citations", [])

        if not answer:
            raise ValueError("API returned an empty answer")

        record["answer"] = answer
        record["citations"] = citations
        record["retrieved_evidence"] = citations
        record["retrieval_metadata"] = result.get("retrieval", {})

        # Save immediately after every successful answer.
        save_jsonl_atomic(DATASET, records)

        completed += 1

        print("Captured successfully")
        print("Citation entries:", len(citations))
        print(
            "Retrieved evidence:",
            result.get("retrieval", {}).get(
                "retrieved_evidence_count", "unknown"
            )
        )

        time.sleep(DELAY_SECONDS)

    except (
        HTTPError,
        URLError,
        TimeoutError,
        ValueError,
        json.JSONDecodeError
    ) as error:
        failed.append((question_id, str(error)))
        print(f"FAILED {question_id}: {error}")

print("\n" + "=" * 55)
print("PHASE 2 CAPTURE SUMMARY")
print("New answers captured:", completed)
print("Total records:", len(records))
print(
    "Remaining unanswered:",
    sum(not record.get("answer") for record in records)
)
print("Failed:", len(failed))

for question_id, error in failed:
    print(f"{question_id}: {error}")

print("\nDataset:", DATASET)
print("Backup:", BACKUP)
