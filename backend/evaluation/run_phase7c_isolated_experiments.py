import json
import os
import sys
import hashlib
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
CHUNKS_FILE = None
OUTPUT_DIR = BACKEND / "evaluation" / "results"

Q36_MISSING_CHUNK = (
    "63d474c175f9d38214cbfbe9daa129a0091ea64b07185d9eacce45b88f69ec90"
)

BASELINE_PROMPT = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer using ONLY the supplied evidence.

Rules:
1. Identify every distinct component requested by the question.
2. Use only directly supported information.
3. Do not combine unrelated facts.
4. Do not include irrelevant information.
5. Cite every factual statement using the exact evidence ID.
6. If a requested component is unsupported, say so explicitly.
7. If none of the requested information is supported, respond exactly:
"I don't have enough information in the provided documents."

Evidence:
{context}

Question:
{question}

Answer:
""")

CHECKLIST_PROMPT = ChatPromptTemplate.from_template("""
You are a precise, evidence-grounded document question-answering assistant.

Answer using ONLY the supplied evidence.

Before writing:
1. Break the question into its distinct requested components.
2. For each component, locate the exact supporting evidence.
3. Record which evidence ID supports that component.
4. Do not treat related information as proof of a different component.

When writing:
1. Address every supported component.
2. Use one concise bullet per distinct component when appropriate.
3. Place the supporting citation immediately after each claim.
4. If a component is not supported, explicitly identify that limitation.
5. Do not add background, unrelated programmes, or unsupported details.
6. Do not omit a supported component merely because another component
   already appears in the same passage.

Final verification:
- Every requested component has been considered.
- Every factual claim has a supporting evidence ID.
- Every citation supports the statement immediately before it.
- No unrelated information has been introduced.

Use only supplied IDs in [EVD-XXXXXXXXXXXX] format.

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

def make_context(evidence):
    blocks = []
    for item in evidence:
        blocks.append(
            f"[{item['evidence_id']}]\n"
            f"Source: {item.get('file_name', 'Unknown')}\n"
            f"Page: {item.get('page', 'Unknown')}\n"
            f"Content:\n{item.get('content', '')}"
        )
    return "\n\n---\n\n".join(blocks)

def load_chunk(chunk_id):
    snapshot_id = (
        BACKEND / "index_store" / "current_snapshot"
    ).read_text(encoding="utf-8").strip()

    chunks_file = (
        BACKEND / "index_store" / "snapshots"
        / snapshot_id / "chunks.jsonl"
    )

    with chunks_file.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get("metadata", {})
            current_id = (
                item.get("chunk_id")
                or metadata.get("chunk_id")
            )

            if current_id == chunk_id:
                content = item.get(
                    "content",
                    item.get("page_content", "")
                )

                if not content:
                    raise RuntimeError(
                        "Matching chunk has no readable content."
                    )

                evidence_id = "EVD-" + hashlib.sha256(
                    chunk_id.encode()
                ).hexdigest()[:12].upper()

                return {
                    "evidence_id": evidence_id,
                    "chunk_id": chunk_id,
                    "file_name": metadata.get(
                        "file_name",
                        metadata.get("source", "Unknown")
                    ),
                    "page": metadata.get("page", "Unknown"),
                    "content": content,
                }

    raise RuntimeError(
        f"Chunk not found in active snapshot: {chunk_id}"
    )

def select_record(records, qid):
    matches = [
        r for r in records
        if r["question_id"] == qid
        and r["strategy"] == "phase5_experimental"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one record for {qid}; found {len(matches)}"
        )

    return matches[0]

def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Existing development model credentials not found."
        )

    records = read_jsonl(INPUT_FILE)
    q36 = select_record(records, "Q36")
    q37 = select_record(records, "Q37")

    # Verify the adjacent chunk before making any model calls.
    extra_chunk = load_chunk(Q36_MISSING_CHUNK)

    q36_evidence = q36.get("retrieved_evidence", [])
    q37_evidence = q37.get("retrieved_evidence", [])

    if any(
        e.get("chunk_id") == Q36_MISSING_CHUNK
        for e in q36_evidence
    ):
        raise RuntimeError(
            "Q36 adjacent chunk is already in the original evidence."
        )

    app.state.llm = __import__(
        "langchain.chat_models",
        fromlist=["init_chat_model"],
    ).init_chat_model(LLM_MODEL)

    baseline_chain = BASELINE_PROMPT | app.state.llm | StrOutputParser()
    checklist_chain = CHECKLIST_PROMPT | app.state.llm | StrOutputParser()

    experiments = [
        {
            "question_id": "Q36",
            "variant": "q36_original_evidence",
            "chain": BASELINE_PROMPT | app.state.llm | StrOutputParser(),
            "evidence": q36_evidence,
            "record": q36,
        },
        {
            "question_id": "Q36",
            "variant": "q36_adjacent_chunk_added",
            "chain": baseline_chain,
            "evidence": q36_evidence + [extra_chunk],
            "record": q36,
        },
        {
            "question_id": "Q37",
            "variant": "q37_existing_prompt",
            "chain": baseline_chain,
            "evidence": q37_evidence,
            "record": q37,
        },
        {
            "question_id": "Q37",
            "variant": "q37_component_checklist",
            "chain": checklist_chain,
            "evidence": q37_evidence,
            "record": q37,
        },
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR /
        f"phase7c_isolated_{timestamp}.jsonl"
    )

    results = []

    for experiment in experiments:
        record = experiment["record"]
        evidence = experiment["evidence"]

        print("\n" + "=" * 75)
        print(
            experiment["question_id"],
            "|",
            experiment["variant"],
            flush=True,
        )

        answer = experiment["chain"].invoke({
            "context": make_context(evidence),
            "question": record["question"],
        })

        result = {
            "question_id": experiment["question_id"],
            "variant": experiment["variant"],
            "question": record["question"],
            "expected_answer": record["expected_answer"],
            "source_strategy": "phase5_experimental",
            "source_evidence": evidence,
            "answer": answer,
        }

        results.append(result)
        print(answer, flush=True)

    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nPhase 7C completed.")
    print("Answers generated:", len(results))
    print("Q36 adjacent chunk added:", extra_chunk["evidence_id"])
    print("Original evidence preserved: Yes")
    print("Production files modified: No")
    print("Output:", output_file.relative_to(BACKEND))

if __name__ == "__main__":
    main()
