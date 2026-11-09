import json
import re
import csv
from pathlib import Path
from collections import defaultdict

INPUT = Path("evaluation/phase7m_full_answers.jsonl")
OUTPUT_DIR = Path("evaluation/phase7m_answer_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with INPUT.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

records = [r for r in records if not r.get("error")]

paired = defaultdict(dict)

for r in records:
    paired[r["question_id"]][r["strategy"]] = r

strategies = [
    "production_baseline",
    "metadata_experimental"
]

summary = {}

for strategy in strategies:
    total = 0
    gold_retrieved = 0
    gold_cited = 0
    gold_total = 0
    citation_ids_valid = 0
    citation_ids_total = 0
    answers_present = 0
    selected_total = 0
    cited_total = 0

    for question_id, pair in paired.items():
        if strategy not in pair:
            continue

        r = pair[strategy]
        total += 1

        gold_ids = set(r.get("gold_chunk_ids") or [])
        evidence = r.get("retrieved_evidence") or []

        retrieved_ids = {
            e.get("chunk_id")
            for e in evidence
            if e.get("chunk_id")
        }

        cited_gold_ids = {
            e.get("chunk_id")
            for e in evidence
            if e.get("chunk_id") and e.get("cited") is True
        }

        gold_total += len(gold_ids)

        if gold_ids & retrieved_ids:
            gold_retrieved += 1

        if gold_ids & cited_gold_ids:
            gold_cited += 1

        answer = r.get("answer") or ""

        if answer.strip():
            answers_present += 1

        citations = r.get("citations") or []
        supplied_ids = {
            c.get("evidence_id") or c.get("id")
            for c in citations
            if c.get("evidence_id") or c.get("id")
        }

        answer_ids = set(re.findall(r"EVD-[A-Fa-f0-9]+", answer))

        citation_ids_total += len(answer_ids)
        citation_ids_valid += len(answer_ids & supplied_ids)

        selected_total += len(evidence)
        cited_total += sum(
            1 for e in evidence if e.get("cited") is True
        )

    summary[strategy] = {
        "questions": total,
        "answers_present": answers_present,
        "gold_questions_with_any_retrieved_chunk": gold_retrieved,
        "gold_questions_with_any_cited_chunk": gold_cited,
        "total_gold_chunks": gold_total,
        "citation_ids_in_answers": citation_ids_total,
        "valid_citation_ids": citation_ids_valid,
        "selected_evidence_total": selected_total,
        "cited_evidence_total": cited_total,
        "gold_retrieval_question_rate": (
            gold_retrieved / total if total else 0
        ),
        "gold_citation_question_rate": (
            gold_cited / total if total else 0
        ),
        "citation_id_validity": (
            citation_ids_valid / citation_ids_total
            if citation_ids_total else 0
        ),
        "mean_selected_evidence": (
            selected_total / total if total else 0
        ),
        "mean_cited_evidence": (
            cited_total / total if total else 0
        )
    }

# Save aggregate metrics
summary_path = OUTPUT_DIR / "aggregate_metrics.json"
summary_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# Save paired question-level comparison
csv_path = OUTPUT_DIR / "paired_answer_comparison.csv"

columns = [
    "question_id",
    "category",
    "question",
    "expected_answer",
    "baseline_answer",
    "metadata_answer",
    "baseline_gold_retrieved",
    "metadata_gold_retrieved",
    "baseline_gold_cited",
    "metadata_gold_cited",
    "baseline_selected_evidence",
    "metadata_selected_evidence",
    "baseline_cited_evidence",
    "metadata_cited_evidence"
]

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()

    for question_id in sorted(paired):
        pair = paired[question_id]

        if not all(s in pair for s in strategies):
            continue

        b = pair["production_baseline"]
        m = pair["metadata_experimental"]

        def evidence_stats(r):
            evidence = r.get("retrieved_evidence") or []
            gold = set(r.get("gold_chunk_ids") or [])

            retrieved = {
                e.get("chunk_id") for e in evidence
                if e.get("chunk_id")
            }

            cited = {
                e.get("chunk_id") for e in evidence
                if e.get("chunk_id") and e.get("cited") is True
            }

            return {
                "retrieved": len(gold & retrieved),
                "cited": len(gold & cited),
                "selected": len(evidence),
                "cited_count": sum(
                    1 for e in evidence if e.get("cited") is True
                )
            }

        bs = evidence_stats(b)
        ms = evidence_stats(m)

        writer.writerow({
            "question_id": question_id,
            "category": b.get("category", ""),
            "question": b.get("question", ""),
            "expected_answer": b.get("expected_answer", ""),
            "baseline_answer": b.get("answer", ""),
            "metadata_answer": m.get("answer", ""),
            "baseline_gold_retrieved": bs["retrieved"],
            "metadata_gold_retrieved": ms["retrieved"],
            "baseline_gold_cited": bs["cited"],
            "metadata_gold_cited": ms["cited"],
            "baseline_selected_evidence": bs["selected"],
            "metadata_selected_evidence": ms["selected"],
            "baseline_cited_evidence": bs["cited_count"],
            "metadata_cited_evidence": ms["cited_count"]
        })

print("\n========== PHASE 7M ANSWER ANALYSIS ==========")

for strategy, metrics in summary.items():
    print(f"\n{strategy.upper()}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

print("\nFiles generated:")
print(summary_path)
print(csv_path)

print("\nPaired questions:", len(paired))
print("Successful answer records:", len(records))
print("==============================================")
