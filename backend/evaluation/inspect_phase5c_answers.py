import json
from pathlib import Path

path = Path("evaluation/phase5c_answers.jsonl")
records = [
    json.loads(line)
    for line in path.read_text().splitlines()
    if line.strip()
]

review_ids = [
    "Q02", "Q03", "Q04", "Q06", "Q07",
    "Q08", "Q27", "Q31", "Q36", "Q37"
]

strategies = ["production_baseline", "phase5_experimental"]

by_question = {}
for r in records:
    by_question.setdefault(r["question_id"], {})[r["strategy"]] = r

for qid in review_ids:
    pair = by_question.get(qid, {})
    if len(pair) != 2:
        print(f"\nMissing pair: {qid}")
        continue

    baseline = pair["production_baseline"]
    experimental = pair["phase5_experimental"]

    print("\n" + "=" * 90)
    print("QUESTION:", qid)
    print("QUESTION TEXT:", baseline["question"])
    print("EXPECTED:", baseline.get("expected_answer"))

    for label, r in [
        ("PRODUCTION BASELINE", baseline),
        ("PHASE 5 EXPERIMENTAL", experimental)
    ]:
        print("\n---", label, "---")
        print("ANSWER:", r.get("answer"))

        evidence = r.get("retrieved_evidence") or []
        print("EVIDENCE:")
        for i, e in enumerate(evidence, 1):
            print(
                f"  {i}. {e.get('file_name')} | "
                f"page={e.get('page')} | "
                f"cited={e.get('cited')} | "
                f"chunk={e.get('chunk_id')}"
            )

print("\nReview complete.")
