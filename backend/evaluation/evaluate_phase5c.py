import json
import re
import csv
from pathlib import Path
from collections import defaultdict

INPUT = Path("evaluation/phase5c_answers.jsonl")
OUTPUT = Path("evaluation/phase5c_comparison.csv")

records = [
    json.loads(line)
    for line in INPUT.read_text().splitlines()
    if line.strip()
]

groups = defaultdict(dict)
for r in records:
    groups[r["question_id"]][r["strategy"]] = r

strategies = ["production_baseline", "phase5_experimental"]

def get_ids(items, key):
    return {
        str(x.get(key))
        for x in items
        if isinstance(x, dict) and x.get(key) is not None
    }

def evaluate(r):
    answer = str(r.get("answer") or "")
    evidence = r.get("retrieved_evidence") or []
    citations = r.get("citations") or []
    gold_chunks = set(map(str, r.get("gold_chunk_ids") or []))

    evidence_ids = get_ids(evidence, "id")
    cited_ids = {
        str(c.get("id") or c.get("evidence_id"))
        for c in citations if isinstance(c, dict)
    }
    retrieved_chunks = get_ids(evidence, "chunk_id")
    cited_chunks = {
        str(e.get("chunk_id"))
        for e in evidence
        if isinstance(e, dict) and e.get("cited")
        and e.get("chunk_id") is not None
    }

    valid_citations = cited_ids.issubset(evidence_ids)
    gold_retrieved = bool(gold_chunks & retrieved_chunks)
    gold_cited = bool(gold_chunks & cited_chunks)

    return {
        "question_id": r["question_id"],
        "strategy": r["strategy"],
        "answer_present": bool(answer.strip()),
        "answer_chars": len(answer),
        "citation_markers": len(re.findall(r"EVD-[A-F0-9]{12}", answer, re.I)),
        "citation_records": len(citations),
        "valid_citation_ids": valid_citations,
        "retrieved_evidence_count": len(evidence),
        "cited_evidence_count": sum(
            1 for e in evidence
            if isinstance(e, dict) and e.get("cited")
        ),
        "gold_chunk_retrieved": gold_retrieved,
        "gold_chunk_cited": gold_cited,
        "gold_chunk_recall": (
            len(gold_chunks & retrieved_chunks) / len(gold_chunks)
            if gold_chunks else None
        ),
    }

rows = [evaluate(r) for r in records]

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("\n========== PHASE 5C OBJECTIVE EVALUATION ==========")

for strategy in strategies:
    subset = [x for x in rows if x["strategy"] == strategy]
    n = len(subset)

    def rate(key):
        return sum(bool(x[key]) for x in subset) / n if n else 0

    recalls = [
        x["gold_chunk_recall"] for x in subset
        if x["gold_chunk_recall"] is not None
    ]

    print(f"\n{strategy} ({n} questions)")
    print(f"Answer presence:          {rate('answer_present'):.1%}")
    print(f"Answers with EVD markers: {sum(x['citation_markers'] > 0 for x in subset)}/{n}")
    print(f"Valid citation IDs:       {rate('valid_citation_ids'):.1%}")
    print(f"Mean citation records:    {sum(x['citation_records'] for x in subset)/n:.2f}")
    print(f"Mean cited evidence:      {sum(x['cited_evidence_count'] for x in subset)/n:.2f}")
    print(f"Gold chunk retrieved:     {rate('gold_chunk_retrieved'):.1%}")
    print(f"Gold chunk cited:         {rate('gold_chunk_cited'):.1%}")
    print(f"Mean gold chunk recall:   {sum(recalls)/len(recalls):.3f}" if recalls else "Mean gold chunk recall: N/A")

print("\n========== PAIRED COMPARISON ==========")
for qid in sorted(groups):
    pair = groups[qid]
    if not all(s in pair for s in strategies):
        continue

    b = next(x for x in rows if x["question_id"] == qid and x["strategy"] == strategies[0])
    e = next(x for x in rows if x["question_id"] == qid and x["strategy"] == strategies[1])

    if b["gold_chunk_recall"] != e["gold_chunk_recall"]:
        print(
            f"{qid}: gold recall "
            f"{b['gold_chunk_recall']} -> {e['gold_chunk_recall']}"
        )

print(f"\nDetailed results saved to: {OUTPUT}")
