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
from app import app

INPUT_FILE = BACKEND / "evaluation" / "phase5c_answers.jsonl"
OUTPUT_DIR = BACKEND / "evaluation" / "results"

QUESTION_IDS = ["Q36", "Q37"]
STRATEGY = "phase5_experimental"

BASELINE_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer the question using ONLY the supplied evidence.

Evidence-handling rules:
1. Treat each evidence passage as belonging to its stated source document.
2. Do not combine facts from different documents unless requested.
3. Prefer evidence that directly answers the question.
4. Do not introduce unsupported facts.

Completeness rules:
5. Identify every distinct component requested by the question.
6. Address each supported component separately and concisely.
7. If evidence supports only some components, identify what is unavailable.
8. If none is supported, respond exactly:
"I don't have enough information in the provided documents."

Citation rules:
9. Cite every factual statement using the exact evidence ID.
10. Use only supplied IDs in [EVD-XXXXXXXXXXXX] format.
11. Place citations immediately after the supported statement.

Evidence:
{context}

Question:
{question}

Answer:
""")

COMPLETENESS_PROMPT = ChatPromptTemplate.from_template("""
You are a precise, evidence-grounded document question-answering system.

Answer the question using ONLY the supplied evidence.

STEP 1 — DECOMPOSE THE QUESTION
Identify every distinct fact, measure, action, entity, condition, or sector
that the question requests. Do not overlook sub-components.

STEP 2 — MAP EVIDENCE
For each requested component:
- Identify the passage that directly supports it.
- Record the exact evidence ID.
- Do not infer that a related statement proves the component.
- Do not combine unrelated documents or events.

STEP 3 — CONSTRUCT THE ANSWER
1. Address every requested component that has direct evidence.
2. Use concise bullets when the question has multiple components.
3. Put citations immediately after the statement they support.
4. If a component is not supported, explicitly say that the provided
   documents do not establish that component.
5. Never invent missing details to make the answer appear complete.
6. Do not include unrelated programmes, examples, statistics or background.
7. Do not repeat the same information in multiple bullets.

STEP 4 — CHECK COMPLETENESS
Before responding, verify:
- Every requested component has been considered.
- Every factual statement has supporting evidence.
- No citation is attached to an unrelated claim.
- No requested component has been silently omitted.

ABSTENTION:
If none of the requested information is supported, respond exactly:
"I don't have enough information in the provided documents."

CITATION FORMAT:
Use only supplied evidence IDs in this exact format:
[EVD-XXXXXXXXXXXX]

EVIDENCE:
{context}

QUESTION:
{question}

ANSWER:
""")

def read_records():
    return [
        json.loads(line)
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Existing development model credentials were not found. "
            "No credentials are printed or requested."
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
                f"Expected one experimental record for {qid}; "
                f"found {len(matches)}"
            )
        selected[qid] = matches[0]

    app.state.llm = __import__(
        "langchain.chat_models",
        fromlist=["init_chat_model"],
    ).init_chat_model(LLM_MODEL)

    baseline_chain = BASELINE_PROMPT | app.state.llm | StrOutputParser()
    completeness_chain = COMPLETENESS_PROMPT | app.state.llm | StrOutputParser()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        OUTPUT_DIR / f"phase7b_completeness_{timestamp}.jsonl"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for qid in QUESTION_IDS:
        record = selected[qid]
        evidence = record.get("retrieved_evidence", [])

        if not evidence:
            raise RuntimeError(f"No evidence available for {qid}")

        # Same evidence construction as Phase 6B.
        blocks = []
        for item in evidence:
            blocks.append(
                f"[{item['evidence_id']}]\n"
                f"Source: {item.get('file_name', 'Unknown')}\n"
                f"Page: {item.get('page', 'Unknown')}\n"
                f"Content:\n{item.get('content', '')}"
            )
        context = "\n\n---\n\n".join(blocks)

        inputs = {
            "context": context,
            "question": record["question"],
        }

        print("\n" + "=" * 70)
        print(f"{qid}: {record['question']}", flush=True)

        for variant, chain in [
            ("phase6b_baseline", baseline_chain),
            ("phase7b_completeness", completeness_chain),
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

            print(f"\n--- {variant} ---")
            print(answer, flush=True)

    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("Phase 7B experiment completed.")
    print("Questions:", len(QUESTION_IDS))
    print("Answers generated:", len(results))
    print("Evidence held fixed: Yes")
    print("Production files modified: No")
    print("Output:", output_file.relative_to(BACKEND))

if __name__ == "__main__":
    main()
