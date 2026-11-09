#!/usr/bin/env python3
"""Validate CortexRAG gold questions against the active snapshot's chunks.jsonl."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
POINTER_FILE = BACKEND_DIR / "index_store" / "current_snapshot"
SEED_FILE = BACKEND_DIR / "evaluation" / "gold_questions_seed.jsonl"
OUTPUT_FILE = BACKEND_DIR / "evaluation" / "gold_questions.jsonl"
REPORT_FILE = BACKEND_DIR / "evaluation" / "gold_validation_report.txt"


def load_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {exc}") from exc
    return records


def normalize_question(value):
    return re.sub(r"\s+", " ", value.strip().casefold())


def main():
    if not POINTER_FILE.exists():
        raise FileNotFoundError(f"Snapshot pointer not found: {POINTER_FILE}")
    if not SEED_FILE.exists():
        raise FileNotFoundError(
            f"Seed dataset not found: {SEED_FILE}\n"
            "Place gold_questions_seed.jsonl inside backend/evaluation/ first."
        )

    snapshot_id = POINTER_FILE.read_text(encoding="utf-8").strip()
    chunks_file = BACKEND_DIR / "index_store" / "snapshots" / snapshot_id / "chunks.jsonl"
    if not chunks_file.exists():
        raise FileNotFoundError(f"Active snapshot chunks file not found: {chunks_file}")

    print(f"Active snapshot: {snapshot_id}")
    print(f"Loading chunks from: {chunks_file}")

    chunk_by_id = {}
    duplicate_chunk_ids = []
    with chunks_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata") or {}
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                continue
            if chunk_id in chunk_by_id:
                duplicate_chunk_ids.append((chunk_id, line_number))
            else:
                chunk_by_id[chunk_id] = record

    questions = load_jsonl(SEED_FILE)
    errors = []
    warnings = []
    enriched = []

    question_ids = [q.get("question_id") for q in questions]
    duplicate_qids = [qid for qid, count in Counter(question_ids).items() if count > 1]
    if duplicate_qids:
        errors.append(f"Duplicate question IDs: {duplicate_qids}")

    normalized_questions = [normalize_question(q.get("question", "")) for q in questions]
    duplicate_questions = [
        question for question, count in Counter(normalized_questions).items()
        if question and count > 1
    ]
    if duplicate_questions:
        errors.append(f"Duplicate question text detected: {duplicate_questions}")

    if duplicate_chunk_ids:
        errors.append(f"Duplicate chunk IDs in chunks.jsonl: {duplicate_chunk_ids[:10]}")

    missing_chunk_refs = []
    for q in questions:
        qid = q.get("question_id", "<missing-id>")
        question = q.get("question")
        expected_answer = q.get("expected_answer")
        gold_ids = q.get("gold_chunk_ids")

        if not question or not question.strip():
            errors.append(f"{qid}: question is empty")
        if not expected_answer or not expected_answer.strip():
            errors.append(f"{qid}: expected_answer is empty")
        if not isinstance(gold_ids, list) or not gold_ids:
            errors.append(f"{qid}: gold_chunk_ids must be a non-empty list")
            gold_ids = []

        if len(gold_ids) != len(set(gold_ids)):
            errors.append(f"{qid}: duplicate gold chunk IDs within the question")

        evidence = []
        for gold_id in gold_ids:
            record = chunk_by_id.get(gold_id)
            if record is None:
                missing_chunk_refs.append((qid, gold_id))
                continue

            metadata = record.get("metadata") or {}
            evidence.append({
                "chunk_id": gold_id,
                "document": metadata.get("file_name") or metadata.get("source_path"),
                "page": metadata.get("page"),
                "document_id": metadata.get("document_id"),
                "document_version": metadata.get("document_version"),
            })

        if len(evidence) != len(gold_ids):
            errors.append(f"{qid}: {len(gold_ids) - len(evidence)} gold chunk reference(s) not found")

        documents = sorted({item["document"] for item in evidence if item["document"]})
        pages = sorted({item["page"] for item in evidence if item["page"] is not None})

        enriched_record = dict(q)
        enriched_record["gold_evidence"] = evidence
        enriched_record["document"] = documents[0] if len(documents) == 1 else documents
        enriched_record["page"] = pages[0] if len(pages) == 1 else pages
        enriched.append(enriched_record)

    total_gold_refs = sum(len(q.get("gold_chunk_ids", [])) for q in questions)
    found_gold_refs = sum(len(q.get("gold_evidence", [])) for q in enriched)
    report_lines = [
        "CORTEXRAG GOLD DATASET VALIDATION",
        "=" * 40,
        f"Active snapshot: {snapshot_id}",
        f"Questions loaded: {len(questions)}",
        f"Unique question IDs: {len(set(question_ids))}",
        f"Gold chunk references: {total_gold_refs}",
        f"Gold chunk references found: {found_gold_refs}",
        f"Indexed chunks loaded: {len(chunk_by_id)}",
        f"Validation errors: {len(errors)}",
        "",
    ]

    if missing_chunk_refs:
        report_lines.append("MISSING GOLD CHUNK REFERENCES:")
        report_lines.extend(f"  {qid}: {chunk_id}" for qid, chunk_id in missing_chunk_refs)
        report_lines.append("")

    if warnings:
        report_lines.append("WARNINGS:")
        report_lines.extend(f"  {warning}" for warning in warnings)
        report_lines.append("")

    if errors:
        report_lines.append("ERRORS:")
        report_lines.extend(f"  - {error}" for error in errors)
        report_lines.append("")
        report_lines.append("RESULT: FAILED — output dataset was not written.")
    else:
        report_lines.append("RESULT: PASSED — all gold chunk references were found.")
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
            for record in enriched:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))

    if errors:
        sys.exit(1)

    print(f"\nValidated dataset written to: {OUTPUT_FILE}")
    print(f"Validation report written to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
