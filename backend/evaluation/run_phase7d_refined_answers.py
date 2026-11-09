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

INPUT_FILE = (
    BACKEND / "evaluation" / "results"
    / "phase7c_isolated_20261104_220134.jsonl"
)
OUTPUT_DIR = BACKEND / "evaluation" / "results"

Q36_BASELINE = ChatPromptTemplate.from_template("""
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

Q36_REFINED = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer the question using ONLY the supplied evidence.

The question asks for measures to reduce greenhouse gas emissions
across sectors. Check each of these requested components separately:

1. Energy efficiency.
2. Resource efficiency.
3. Renewable energy use.
4. Decarbonization of buildings.
5. Decarbonization of transportation.
6. Decarbonization of power.
7. Decarbonization of industry.

For each component:
- Include it only if directly supported by the evidence.
- Preserve the meaning of the source passage.
- Cite the exact evidence ID immediately after the claim.

Important:
- Do not substitute a related measure for a requested component.
- For example, vehicle emissions rules alone do not establish every
  transportation decarbonization measure.
- Grid decarbonization does not replace the explicit power-sector claim.
- Do not omit a supported sector merely because it appears in a
  different evidence passage.
- Do not add regulatory programmes, statistics, or background that
  the question did not request.
- Keep the answer concise, preferably one sentence or a short list.
- Do not claim that a component is unsupported if the evidence
  directly establishes it.

If none of the requested measures is supported, respond exactly:
"I don't have enough information in the provided documents."

Use only supplied evidence IDs in [EVD-XXXXXXXXXXXX] format.

Evidence:
{context}

Question:
{question}

Answer:
""")

Q37_BASELINE = ChatPromptTemplate.from_template("""
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

Q37_REFINED = ChatPromptTemplate.from_template("""
You are a precise document question-answering assistant.

Answer the question using ONLY the supplied evidence.

The question asks for exactly three aspects of EPA's international role:

1. Sharing expertise internationally.
2. Learning from the expertise of others.
3. Supporting science-based climate mitigation and adaptation goals.

Write exactly three concise bullets, one for each aspect.

For every bullet:
- State the aspect directly.
- Use only what the evidence establishes.
- Place its exact supporting evidence ID immediately after the claim.
- Do not add examples, background, programmes, or other international
  activities unless needed to explain one of the three aspects.

Do not omit an aspect when it is directly supported.
Do not use a citation merely because it comes from the same document.
Each citation must support its associated bullet.

If an aspect is genuinely unsupported, state that limitation in its
bullet rather than inventing information.

If none of the three aspects is supported, respond exactly:
"I don't have enough information in the provided documents."

Use only supplied IDs in [EVD-XXXXXXXXXXXX] format.

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

def get_variant(records, qid, variant):
    matches = [
        r for r in records
        if r["question_id"] == qid and r["variant"] == variant
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {qid}/{variant} record; found {len(matches)}"
        )
    return matches[0]

def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "Existing development model credentials not found."
        )

    records = read_records()

    q36 = get_variant(
        records, "Q36", "q36_adjacent_chunk_added"
    )
    q37 = get_variant(
        records, "Q37", "q37_component_checklist"
    )

    if not any(
        e.get("chunk_id", "").startswith("63d474c1")
        for e in q36["source_evidence"]
    ):
        raise RuntimeError(
            "Q36 adjacent gold chunk is missing from the captured evidence."
        )

    if not any(
        e.get("chunk_id", "").startswith("69666065")
        for e in q37["source_evidence"]
    ):
        raise RuntimeError(
            "Q37 gold evidence is missing from the captured evidence."
        )

    app.state.llm = __import__(
        "langchain.chat_models",
        fromlist=["init_chat_model"],
    ).init_chat_model(LLM_MODEL)

    experiments = [
        (
            "Q36",
            "q36_adjacent_existing_prompt",
            q36,
            Q36_BASELINE,
        ),
        (
            "Q36",
            "q36_adjacent_refined_prompt",
            q36,
            Q36_REFINED,
        ),
        (
            "Q37",
            "q37_checklist_existing_prompt",
            q37,
            Q37_BASELINE,
        ),
        (
            "Q37",
            "q37_checklist_refined_prompt",
            q37,
            Q37_REFINED,
        ),
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = (
        OUTPUT_DIR / f"phase7d_refined_{timestamp}.jsonl"
    )

    results = []

    for qid, variant, record, prompt in experiments:
        evidence = record["source_evidence"]
        chain = prompt | app.state.llm | StrOutputParser()

        print("\n" + "=" * 75)
        print(qid, "|", variant, flush=True)

        answer = chain.invoke({
            "context": build_context(evidence),
            "question": record["question"],
        })

        result = {
            "question_id": qid,
            "variant": variant,
            "question": record["question"],
            "expected_answer": record["expected_answer"],
            "source_evidence": evidence,
            "answer": answer,
        }
        results.append(result)
        print(answer, flush=True)

    with output_file.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nPhase 7D completed.")
    print("Answers generated:", len(results))
    print("Evidence held fixed within each question: Yes")
    print("Production files modified: No")
    print("Output:", output_file.relative_to(BACKEND))

if __name__ == "__main__":
    main()
