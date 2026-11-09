import json
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

PHASE5_FILE = BACKEND / "evaluation" / "phase5c_answers.jsonl"
PHASE7C_FILE = (
    BACKEND / "evaluation" / "results"
    / "phase7c_isolated_20261104_220134.jsonl"
)
REGRESSION_FILE = (
    BACKEND / "evaluation" / "phase7_regression_questions.jsonl"
)
OUTPUT_DIR = BACKEND / "evaluation" / "results"

QUESTION_IDS = [
    "Q02", "Q03", "Q04", "Q06", "Q07",
    "Q08", "Q27", "Q31", "Q36", "Q37"
]

EXPERIMENTAL_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer the question using ONLY the supplied evidence.

EVIDENCE ANALYSIS:
1. Identify the distinct facts, actions, entities, conditions,
   quantities, or categories requested by the question.
2. Find the passage that directly supports each requested component.
3. Do not treat related information as proof of a different component.
4. Do not combine facts from unrelated documents or events.

ANSWER CONSTRUCTION:
1. Address every requested component that has direct evidence.
2. Use concise bullets when multiple components are requested.
3. Preserve important names, values, dates, units, and distinctions.
4. If only some components are supported, state which information
   is unavailable.
5. Do not add background or unrelated details.
6. Do not invent information to make an answer complete.
7. If none of the requested information is supported, respond exactly:
"I don't have enough information in the provided documents."

CITATION VALIDATION:
1. Cite every factual claim.
2. Use only the exact evidence IDs supplied.
3. Place citations immediately after the claims they support.
4. Do not cite a passage for a claim it does not establish.
5. Do not omit a supported component simply because it appears
   in a different passage.

Before responding, verify that every requested component has
been considered and every factual claim has supporting evidence.

Evidence:
{context}

Question:
{question}

Answer:
""")

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
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
            "Existing development model credentials not found."
        )

    regression = read_jsonl(REGRESSION_FILE)
    expected_ids = {
        item["question_id"] for item in regression
    }

    if expected_ids != set(QUESTION_IDS):
        raise RuntimeError(
            f"Regression dataset mismatch: {expected_ids}"
        )

    phase5 = read_jsonl(PHASE5_FILE)
    phase7c = read_jsonl(PHASE7C_FILE)

    selected = {}

    for qid in QUESTION_IDS:
        matches = [
            r for r in phase5
            if r["question_id"] == qid
            and r["strategy"] == "phase5_experimental"
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one Phase 5C experimental record for "
                f"{qid}; found {len(matches)}"
            )

        selected[qid] = matches[0]

    # Use the validated adjacent-chunk evidence for Q36.
    q36_extra = [
        r for r in phase7c
        if r["question_id"] == "Q36"
        and r["variant"] == "q36_adjacent_chunk_added"
    ]

    if len(q36_extra) != 1:
        raise RuntimeError("Q36 adjacent evidence not found.")

    selected["Q36"]["retrieved_evidence"] = (
        q36_extra[0]["source_evidence"]
    )

    app.state.llm = __import__(
        "langchain.chat_models",
        fromlist=["init_chat_model"],
    ).init_chat_model(LLM_MODEL)

    baseline_chain = ANSWER_PROMPT | app.state.llm | StrOutputParser()
    experimental_chain = (
        EXPERIMENTAL_PROMPT | app.state.llm | StrOutputParser()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR / f"phase7e_generalization_{timestamp}.jsonl"
    )

    results = []

    for qid in QUESTION_IDS:
        record = selected[qid]
        evidence = record["retrieved_evidence"]

        if not evidence:
            raise RuntimeError(f"No evidence for {qid}")

        inputs = {
            "context": build_context(evidence),
            "question": record["question"],
        }

        print("\n" + "=" * 75)
        print(qid, "|", record["question"], flush=True)

        for variant, chain in [
            ("production_baseline", baseline_chain),
            ("generic_completeness", experimental_chain),
        ]:
            answer = chain.invoke(inputs)

            results.append({
                "question_id": qid,
                "variant": variant,
                "question": record["question"],
                "expected_answer": record["expected_answer"],
                "source_evidence": evidence,
                "answer": answer,
            })

            print("\n---", variant, "---")
            print(answer, flush=True)

    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nPhase 7E completed.")
    print("Questions:", len(QUESTION_IDS))
    print("Answers generated:", len(results))
    print("Same evidence per question: Yes")
    print("Q36 adjacent chunk retained: Yes")
    print("Production files modified: No")
    print("Output:", output_file.relative_to(BACKEND))

if __name__ == "__main__":
    main()
