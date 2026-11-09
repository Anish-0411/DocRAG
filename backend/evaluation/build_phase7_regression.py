import json
from pathlib import Path

source = Path("evaluation/gold_questions.jsonl")
output = Path("evaluation/phase7_regression_questions.jsonl")

target_focus = {
    "Q02": "wrong_document_retrieval",
    "Q03": "wrong_document_retrieval",
    "Q04": "cross_document_evidence_mixing",
    "Q06": "evidence_retrieval_and_numerical_correctness",
    "Q07": "insufficient_evidence_and_abstention",
    "Q36": "multi_evidence_completeness",
    "Q37": "evidence_utilization_and_citation_selection",
    "Q08": "control_previously_successful",
    "Q27": "control_previously_successful",
    "Q31": "control_previously_successful",
}

questions = [
    json.loads(line)
    for line in source.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

by_id = {q["question_id"]: q for q in questions}

missing = set(target_focus) - set(by_id)
if missing:
    raise SystemExit(f"Missing question IDs in gold dataset: {sorted(missing)}")

records = []

for qid, focus in target_focus.items():
    record = dict(by_id[qid])
    record["phase7_focus"] = focus
    records.append(record)

with output.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Phase 7A regression dataset created.")
print("Questions:", len(records))
print("Output:", output)

print("\nQuestion coverage:")
for record in records:
    print(
        f'{record["question_id"]}: '
        f'{record["phase7_focus"]} | '
        f'{record["document"]}, page {record["page"]} | '
        f'{len(record["gold_chunk_ids"])} gold chunk(s)'
    )

print("\nValidation:")
print("Unique question IDs:", len({r["question_id"] for r in records}))
print("All expected answers present:", all(r.get("expected_answer") for r in records))
print("All gold evidence present:", all(r.get("gold_evidence") for r in records))
print("Q25 excluded:", "Q25" not in {r["question_id"] for r in records})
