import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
load_dotenv(BACKEND / ".env")

from config import LLM_MODEL
from app import app, ANSWER_PROMPT

INPUT_FILE = BACKEND / "evaluation" / "phase5c_answers.jsonl"
OUTPUT_DIR = BACKEND / "evaluation" / "results"
QUESTION_IDS = ["Q04", "Q36", "Q37"]
STRATEGY = "phase5_experimental"

IMPROVED_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer the question using ONLY the supplied evidence.

Evidence-handling rules:
1. Treat each evidence passage as belonging to its stated source document.
2. Do not combine facts from different reports, studies, or documents unless
   the question explicitly asks for a comparison.
3. Prefer evidence that directly answers the question over loosely related
   passages.
4. If multiple documents contain similar facts, do not assume they describe
   the same event, subject, or situation.
5. Do not introduce facts, values, dates, locations, or interpretations that
   are not explicitly supported by the selected evidence.

Completeness rules:
6. Identify every distinct component requested by the question.
7. Address each component separately and concisely.
8. Do not replace a multi-part answer with a general summary.
9. If evidence supports only some components, answer those components and
   clearly state which requested information is unavailable.
10. If none of the requested information is supported, respond exactly:
"I don't have enough information in the provided documents."

Citation rules:
11. Cite every factual statement using the exact evidence ID.
12. Use only IDs supplied in the evidence.
13. Use this exact format: [EVD-XXXXXXXXXXXX].
14. Place citations immediately after the statement they support.
15. Never cite a passage that does not support the associated statement.

Evidence:
{context}

Question:
{question}

Answer:
""")

def read_records():
    return [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def build_context(evidence):
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item['evidence_id']}]\n"
            f"Source: {item.get('file_name', 'Unknown')}\n"
            f"Page: {item.get('page', 'Unknown')}\n"
            f"Content:\n{item.get('content', '')}"
        )
    return "\n\n---\n\n".join(blocks)

def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY was not found in the existing backend .env. "
            "No key value is requested or printed."
        )

    records = read_records()
    selected = {}

    for qid in QUESTION_IDS:
        matches = [
            r for r in records
            if r["question_id"] == qid and r["strategy"] == STRATEGY
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {STRATEGY} record for {qid}; "
                f"found {len(matches)}"
            )
        selected[qid] = matches[0]

    # Initialize the same model configuration as Phase 5C.
    app.state.llm = __import__(
        "langchain.chat_models",
        fromlist=["init_chat_model"],
    ).init_chat_model(LLM_MODEL)

    current_chain = ANSWER_PROMPT | app.state.llm | StrOutputParser()
    improved_chain = IMPROVED_PROMPT | app.state.llm | StrOutputParser()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"phase6b_prompt_experiment_{timestamp}.jsonl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for qid in QUESTION_IDS:
        record = selected[qid]
        evidence = record.get("retrieved_evidence", [])

        if not evidence:
            raise RuntimeError(f"No stored evidence found for {qid}")

        context = build_context(evidence)
        inputs = {
            "context": context,
            "question": record["question"],
        }

        print(f"\nRunning {qid}: {record['question']}", flush=True)

        for variant, chain in [
            ("current_prompt", current_chain),
            ("improved_prompt", improved_chain),
        ]:
            answer = chain.invoke(inputs)

            result = {
                "question_id": qid,
                "variant": variant,
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "source_strategy": STRATEGY,
                "source_evidence": evidence,
                "answer": answer,
            }
            results.append(result)

            print(f"\n--- {variant} ---\n{answer}\n", flush=True)

    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("=" * 65)
    print("Phase 6B completed.")
    print(f"Questions: {len(QUESTION_IDS)}")
    print(f"Answers generated: {len(results)}")
    print(f"Evidence held fixed: Yes")
    print(f"Output: {output_file.relative_to(BACKEND)}")

if __name__ == "__main__":
    main()
