
import csv
import json
import re
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).resolve().parent

INPUT = BASE_DIR / "phase2_answers.jsonl"
OUTPUT_CSV = BASE_DIR / "phase2_question_report.csv"
OUTPUT_MD = BASE_DIR / "phase2_report.md"

CITATION_PATTERN = re.compile(r"EVD-[A-Fa-f0-9]{12}")

records = [
    json.loads(line)
    for line in INPUT.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not records:
    raise ValueError("No evaluation records found.")

question_ids = [r["question_id"] for r in records]

if len(question_ids) != len(set(question_ids)):
    raise ValueError("Duplicate question IDs found.")

rows = []

for record in records:
    answer = record.get("answer", "")
    citations = record.get("citations", [])
    gold_evidence = record.get("gold_evidence", [])

    answer_ids = set(CITATION_PATTERN.findall(answer))

    retrieved_ids = {
        item.get("id") or item.get("evidence_id")
        for item in citations
        if item.get("id") or item.get("evidence_id")
    }

    cited_items = [
        item for item in citations
        if item.get("cited") is True
    ]

    cited_ids = {
        item.get("id") or item.get("evidence_id")
        for item in cited_items
        if item.get("id") or item.get("evidence_id")
    }

    # Structural citation checks, not factual validation.
    citation_ids_valid = answer_ids.issubset(retrieved_ids)
    answer_has_citation = bool(answer_ids)
    cited_flag_consistent = answer_ids.issubset(cited_ids)

    gold_chunk_ids = {
        item.get("chunk_id")
        for item in gold_evidence
        if item.get("chunk_id")
    }

    cited_chunk_ids = {
        item.get("chunk_id")
        for item in cited_items
        if item.get("chunk_id")
    }

    exact_gold_match = bool(gold_chunk_ids & cited_chunk_ids)

    gold_documents = {
        item.get("document")
        for item in gold_evidence
        if item.get("document")
    }

    gold_pages = {
        item.get("page")
        for item in gold_evidence
        if item.get("page") is not None
    }

    cited_documents = {
        item.get("file_name") or item.get("source")
        for item in cited_items
        if item.get("file_name") or item.get("source")
    }

    # Gold page numbers are zero-based; citation pages are one-based.
    cited_pages = {
        int(item["page"]) - 1
        for item in cited_items
        if item.get("page") is not None
    }

    document_match = bool(gold_documents & cited_documents)
    page_match = bool(gold_pages & cited_pages)

    human = record.get("human_evaluation", {})

    rows.append({
        "question_id": record["question_id"],
        "category": record.get("category", ""),
        "answer_present": bool(answer),
        "answer_citation_count": len(answer_ids),
        "retrieved_evidence_count": len(citations),
        "marked_cited_count": len(cited_items),
        "citation_ids_valid": citation_ids_valid,
        "cited_flag_consistent": cited_flag_consistent,
        "exact_gold_chunk_match": exact_gold_match,
        "gold_document_match": document_match,
        "gold_page_match": page_match,
        "correctness": human.get("correctness"),
        "faithfulness": human.get("faithfulness"),
        "answer_relevance": human.get("answer_relevance"),
        "citation_support": human.get("citation_support"),
        "abstention_appropriate": human.get("abstention_appropriate"),
        "evaluation_notes": record.get("evaluation_notes", "")
    })


# Validate human scores.
manual_fields = [
    "correctness",
    "faithfulness",
    "answer_relevance",
    "citation_support"
]

for row in rows:
    for field in manual_fields:
        value = row[field]

        if value is not None and value not in (0, 1, 2):
            raise ValueError(
                f"Invalid {field} score for {row['question_id']}: {value}"
            )


# Save question-level CSV.
with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


total = len(rows)


def rate(predicate):
    return (
        sum(bool(predicate(row)) for row in rows) / total
        if total else 0
    )


answers_present = rate(lambda r: r["answer_present"])
answers_with_citations = rate(
    lambda r: r["answer_citation_count"] > 0
)
valid_citation_answers = rate(lambda r: r["citation_ids_valid"])
consistent_citation_flags = rate(lambda r: r["cited_flag_consistent"])
exact_gold_matches = rate(lambda r: r["exact_gold_chunk_match"])
document_matches = rate(lambda r: r["gold_document_match"])
page_matches = rate(lambda r: r["gold_page_match"])

total_retrieved = sum(r["retrieved_evidence_count"] for r in rows)
total_marked_cited = sum(r["marked_cited_count"] for r in rows)

cited_evidence_ratio = (
    total_marked_cited / total_retrieved
    if total_retrieved else 0
)


# Human assessment metrics: only explicitly scored questions count.
manual_scores = {}

for field in manual_fields:
    values = [
        row[field]
        for row in rows
        if isinstance(row[field], (int, float))
        and not isinstance(row[field], bool)
    ]

    evaluated = len(values)

    manual_scores[field] = {
        "evaluated": evaluated,
        "pending": total - evaluated,
        "mean": round(mean(values), 3) if values else None,
        "score_2_count": sum(v == 2 for v in values),
        "score_2_rate": (
            sum(v == 2 for v in values) / evaluated
            if evaluated else None
        ),
        "score_0_count": sum(v == 0 for v in values),
        "score_1_count": sum(v == 1 for v in values)
    }


# Abstention evaluation.
abstention_values = [
    row["abstention_appropriate"]
    for row in rows
    if row["abstention_appropriate"] in (0, 1, 2)
]

appropriate_abstentions = sum(
    value == 2 for value in abstention_values
)

abstention_rate = (
    appropriate_abstentions / len(abstention_values)
    if abstention_values else None
)


# Diagnostic counts. These are not confirmed root causes.
diagnostics = {
    "no_answer": sum(not r["answer_present"] for r in rows),
    "no_answer_citation": sum(
        r["answer_present"] and r["answer_citation_count"] == 0
        for r in rows
    ),
    "invalid_citation_reference": sum(
        not r["citation_ids_valid"] for r in rows
    ),
    "citation_flag_inconsistency": sum(
        not r["cited_flag_consistent"] for r in rows
    ),
    "exact_gold_chunk_not_cited": sum(
        not r["exact_gold_chunk_match"] for r in rows
    ),
    "gold_document_not_cited": sum(
        not r["gold_document_match"] for r in rows
    ),
    "gold_page_not_cited": sum(
        not r["gold_page_match"] for r in rows
    ),
    "manually_scored_correctness_below_2": sum(
        isinstance(r["correctness"], (int, float))
        and r["correctness"] < 2
        for r in rows
    ),
    "inappropriate_abstention": sum(
        r["abstention_appropriate"] == 0 for r in rows
    )
}


# Build report.
report = f"""# CortexRAG Phase 2 — 40-Question Evaluation

## Dataset status

- Total questions: {total}
- Answers present: {sum(r['answer_present'] for r in rows)}/{total}
- Questions with at least one answer citation: {sum(r['answer_citation_count'] > 0 for r in rows)}/{total}
- Retrieved evidence entries: {total_retrieved}
- Evidence entries marked cited: {total_marked_cited}

## Automatic structural metrics

| Metric | Result |
|---|---:|
| Answer presence | {answers_present:.1%} |
| Answers containing at least one citation | {answers_with_citations:.1%} |
| Citation IDs found in retrieved evidence | {valid_citation_answers:.1%} |
| Citation IDs consistent with cited flags | {consistent_citation_flags:.1%} |
| Retrieved evidence marked cited / total retrieved | {cited_evidence_ratio:.1%} |
| Exact gold chunk match | {exact_gold_matches:.1%} |
| Gold document match | {document_matches:.1%} |
| Gold page match | {page_matches:.1%} |

These structural checks do not establish factual correctness or semantic citation support.

## Manual answer-quality evaluation

Scores range from 0 to 2. Pending questions are excluded from averages.

| Dimension | Evaluated | Pending | Mean | Score 2 rate | Score 0 | Score 1 | Score 2 |
|---|---:|---:|---:|---:|---:|---:|---:|
"""

for field, values in manual_scores.items():
    mean_value = (
        str(values["mean"])
        if values["mean"] is not None else "Pending"
    )

    score_rate = (
        f"{values['score_2_rate']:.1%}"
        if values["score_2_rate"] is not None else "Pending"
    )

    report += (
        f"| {field} | {values['evaluated']} | "
        f"{values['pending']} | {mean_value} | {score_rate} | "
        f"{values['score_0_count']} | {values['score_1_count']} | "
        f"{values['score_2_count']} |\n"
    )


report += f"""

## Abstention evaluation

- Evaluated abstention cases: {len(abstention_values)}
- Appropriate abstentions: {appropriate_abstentions}
- Appropriate abstention rate: {
    f"{abstention_rate:.1%}" if abstention_rate is not None else "Pending"
}

Questions where the system attempted an answer are excluded.

## Structural and manual diagnostic counts

These categories are diagnostic signals, not confirmed root causes.
They may overlap and must not be added together.

| Diagnostic | Questions |
|---|---:|
"""

for category, count in diagnostics.items():
    report += f"| {category} | {count} |\n"


report += """

## Question-level evaluation

| Question | Correctness | Faithfulness | Relevance | Citation support | Notes |
|---|---:|---:|---:|---:|---|
"""

for row in rows:
    def display_score(value):
        return str(value) if value is not None else "Pending"

    notes = row["evaluation_notes"].replace("|", "/").replace("\n", " ")

    report += (
        f"| {row['question_id']} "
        f"| {display_score(row['correctness'])} "
        f"| {display_score(row['faithfulness'])} "
        f"| {display_score(row['answer_relevance'])} "
        f"| {display_score(row['citation_support'])} "
        f"| {notes} |\n"
    )


report += """

## Interpretation notes

- Citation ID validity checks structure, not factual support.
- Exact gold chunk matching is a strict evidence-alignment proxy.
- A different chunk may still contain semantically valid evidence.
- Document and page matches do not independently prove faithfulness.
- Human scores are manual judgments, not model-generated metrics.
- Pending manual scores are excluded from averages.
- Diagnostic categories can overlap.
- This evaluator performs no model or external API calls.
"""

OUTPUT_MD.write_text(report, encoding="utf-8")


# Console summary.
print("PHASE 2 OFFLINE EVALUATION COMPLETE")
print("Questions:", total)
print("Answers present:", f"{answers_present:.1%}")
print("Answer citation coverage:", f"{answers_with_citations:.1%}")
print("Citation ID validity:", f"{valid_citation_answers:.1%}")
print("Exact gold chunk match:", f"{exact_gold_matches:.1%}")
print("Gold document match:", f"{document_matches:.1%}")
print("Gold page match:", f"{page_matches:.1%}")

print("\nManual quality scores:")
for field, values in manual_scores.items():
    print(
        field,
        "| evaluated:", values["evaluated"],
        "| pending:", values["pending"],
        "| mean:", values["mean"],
        "| score 2 rate:",
        f"{values['score_2_rate']:.1%}"
        if values["score_2_rate"] is not None else "Pending"
    )

print(
    "\nAppropriate abstention rate:",
    f"{abstention_rate:.1%}"
    if abstention_rate is not None else "Pending"
)

print("\nDiagnostics:")
for category, count in diagnostics.items():
    print(f"{category}: {count}")

print("\nMarkdown report:", OUTPUT_MD)
print("CSV report:", OUTPUT_CSV)
print("No API calls made.")
