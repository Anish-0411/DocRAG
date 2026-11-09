import json
import re
from pathlib import Path
from collections import Counter

path = Path(
    "evaluation/results/phase6b_prompt_experiment_20261104_213505.jsonl"
)

records = [
    json.loads(line)
    for line in path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

pattern = re.compile(r"EVD-[A-F0-9]{12}", re.IGNORECASE)

for record in records:
    answer = record["answer"]
    evidence_ids = {
        item["evidence_id"].upper()
        for item in record["source_evidence"]
    }

    used = [
        match.upper()
        for match in pattern.findall(answer)
    ]

    counts = Counter(used)
    unique_used = set(used)

    invalid = unique_used - evidence_ids
    unused = evidence_ids - unique_used

    print("\n" + "=" * 65)
    print(record["question_id"], record["variant"])

    print("Evidence records:", len(evidence_ids))
    print("Citation occurrences:", len(used))
    print("Unique citations:", len(unique_used))
    print("Invalid citation IDs:", sorted(invalid))
    print("Unused evidence IDs:", sorted(unused))
    print("Repeated citation IDs:", {
        key: count for key, count in counts.items() if count > 1
    })
    print("Has citation:", bool(used))
    print("Answer length:", len(answer))

print("\nCitation audit completed. No source files modified.")
