import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from langchain_core.output_parsers import StrOutputParser

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

load_dotenv(BACKEND / ".env")

from config import (
    LLM_MODEL,
    RERANKER_MODEL,
    RERANK_TOP_K,
    MAX_CONTEXT_TOKENS,
)

from app import app, ANSWER_PROMPT
from snapshot_manager import load_active_snapshot

from retrieval import (
    build_retrievers,
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)

from reranker import select_context

from citation_service import (
    prepare_citations,
    finalize_citations,
)

EVALUATION_DIR = BACKEND / "evaluation"

QUESTIONS_FILE = EVALUATION_DIR / "gold_questions.jsonl"
OUTPUT_FILE = EVALUATION_DIR / "phase7m_full_answers.jsonl"

EXPERIMENTAL_DEPTH = 100
RRF_K = 60
PAUSE_BETWEEN_CALLS = 1.0

STRATEGIES = (
    "production_baseline",
    "metadata_experimental",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger("phase7m_answers")


def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def atomic_write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp:
            for record in records:
                temp.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

            temp_path = Path(temp.name)

        os.replace(temp_path, path)

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def load_completed():
    if not OUTPUT_FILE.exists():
        return []

    return [
        record
        for record in read_jsonl(OUTPUT_FILE)
        if not record.get("error")
    ]


def metadata_rerank(query, candidates, model):
    pairs = []

    for candidate in candidates:
        document = candidate["document"]
        metadata = document.metadata

        filename = metadata.get("file_name") or "Unknown source"
        page = metadata.get("page")

        page_label = (
            str(int(page) + 1)
            if page is not None
            else "Unknown page"
        )

        enriched_content = (
            f"Source: {filename}\n"
            f"Page: {page_label}\n"
            f"Content:\n{document.page_content}"
        )

        pairs.append((query, enriched_content))

    scores = model.predict(
        pairs,
        show_progress_bar=False,
        batch_size=16,
    )

    ranked = []

    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)

    ranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    return ranked


def generate_metadata_answer(question, retrievers, model):
    bm25 = retrievers["bm25_retriever"]
    original_bm25_k = bm25.k

    try:
        dense_results = dense_search(
            query=question,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=EXPERIMENTAL_DEPTH,
        )

        bm25.k = EXPERIMENTAL_DEPTH

        sparse_results = sparse_search(
            query=question,
            bm25_retriever=bm25,
            top_k=EXPERIMENTAL_DEPTH,
        )

        candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            rrf_k=RRF_K,
            top_k=EXPERIMENTAL_DEPTH,
        )

        ranked = metadata_rerank(
            query=question,
            candidates=candidates,
            model=model,
        )

        selected = select_context(
            reranked_candidates=ranked,
            top_k=RERANK_TOP_K,
            max_tokens=MAX_CONTEXT_TOKENS,
        )

    finally:
        bm25.k = original_bm25_k

    if not selected:
        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "citations": [],
            "retrieval": {
                "candidate_count": 0,
                "selected_count": 0,
                "retrieved_evidence_count": 0,
                "cited_evidence_count": 0,
            },
        }

    citation_data = prepare_citations(selected)

    chain = ANSWER_PROMPT | app.state.llm | StrOutputParser()

    raw_answer = chain.invoke({
        "context": citation_data["evidence_context"],
        "question": question,
    })

    citation_result = finalize_citations(
        answer=raw_answer,
        evidence_records=citation_data["evidence_records"],
    )

    return {
        "answer": citation_result["answer"],
        "citations": citation_result["citations"],
        "retrieval": {
            "candidate_count": len(candidates),
            "selected_count": len(selected),
            "estimated_context_tokens": sum(
                max(
                    1,
                    (len(item["document"].page_content) + 2) // 3,
                )
                for item in selected
            ),
            "retrieved_evidence_count": (
                citation_result["retrieved_evidence_count"]
            ),
            "cited_evidence_count": (
                citation_result["cited_evidence_count"]
            ),
        },
    }


def make_record(question_data, strategy, result, snapshot_id):
    return {
        "question_id": question_data["question_id"],
        "strategy": strategy,
        "question": question_data["question"],
        "expected_answer": question_data.get("expected_answer"),
        "category": question_data.get("category"),
        "document": question_data.get("document"),
        "page": question_data.get("page"),
        "gold_chunk_ids": question_data.get("gold_chunk_ids", []),
        "gold_evidence": question_data.get("gold_evidence", []),
        "answer": result["answer"],
        "citations": result.get("citations", []),
        "retrieved_evidence": result.get("citations", []),
        "retrieval_metadata": {
            **result.get("retrieval", {}),
            "snapshot_id": snapshot_id,
            "strategy": strategy,
        },
        "human_evaluation": {},
        "evaluation_notes": "",
    }


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY was not found in the existing backend .env. "
            "No key value is requested or printed."
        )

    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(QUESTIONS_FILE)

    questions = read_jsonl(QUESTIONS_FILE)

    if len(questions) != 40:
        raise ValueError(
            f"Expected 40 questions, found {len(questions)}"
        )

    question_ids = [q["question_id"] for q in questions]

    if len(set(question_ids)) != 40:
        raise ValueError("Question IDs are not unique.")

    completed_records = load_completed()

    completed_keys = {
        (record["question_id"], record["strategy"])
        for record in completed_records
    }

    results = list(completed_records)

    print("Loading active snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    snapshot_id = snapshot.get("snapshot_id", "unknown")

    print("Active snapshot:", snapshot_id)

    print("Initializing development LLM...")
    from langchain.chat_models import init_chat_model

    app.state.llm = init_chat_model(LLM_MODEL)
    app.state.retrievers = retrievers
    app.state.snapshot = snapshot

    print("Loading CrossEncoder:", RERANKER_MODEL)
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    from app import run_rag

    total_tasks = len(questions) * len(STRATEGIES)
    completed_count = len(completed_keys)

    print("\n========== PHASE 7M ANSWER BENCHMARK ==========")
    print("Questions:", len(questions))
    print("Strategies:", len(STRATEGIES))
    print("Total answer tasks:", total_tasks)
    print("Already completed:", completed_count)
    print("Remaining:", total_tasks - completed_count)
    print("Output:", OUTPUT_FILE)
    print("===============================================\n")

    for question_data in questions:
        question_id = question_data["question_id"]
        question = question_data["question"]

        for strategy in STRATEGIES:
            key = (question_id, strategy)

            if key in completed_keys:
                print(f"SKIPPING {question_id} | {strategy}")
                continue

            print(f"\nRUNNING {question_id} | {strategy}")
            started = time.time()

            try:
                if strategy == "production_baseline":
                    result = run_rag(question)

                elif strategy == "metadata_experimental":
                    result = generate_metadata_answer(
                        question=question,
                        retrievers=retrievers,
                        model=cross_encoder,
                    )

                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                record = make_record(
                    question_data=question_data,
                    strategy=strategy,
                    result=result,
                    snapshot_id=snapshot_id,
                )

                results.append(record)
                completed_keys.add(key)

                atomic_write_jsonl(results, OUTPUT_FILE)

                elapsed = time.time() - started
                completed_count += 1

                print(
                    f"SUCCESS {question_id} | {strategy} "
                    f"| {elapsed:.2f}s "
                    f"| {completed_count}/{total_tasks}"
                )

            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                print("Completed results have been saved.")
                raise

            except Exception as exc:
                logger.exception(
                    "Failed: %s | %s",
                    question_id,
                    strategy,
                )

                error_record = {
                    "question_id": question_id,
                    "strategy": strategy,
                    "question": question,
                    "error": str(exc),
                    "snapshot_id": snapshot_id,
                }

                results.append(error_record)
                atomic_write_jsonl(results, OUTPUT_FILE)

                print(
                    f"FAILED {question_id} | {strategy} "
                    f"| {type(exc).__name__}: {exc}"
                )

            if PAUSE_BETWEEN_CALLS:
                time.sleep(PAUSE_BETWEEN_CALLS)

    successful = [
        r for r in results if not r.get("error")
    ]

    failed = [
        r for r in results if r.get("error")
    ]

    print("\n========== PHASE 7M ANSWER BENCHMARK COMPLETE ==========")
    print("Successful answers:", len(successful), "/", total_tasks)
    print("Failed attempts:", len(failed))
    print("Output:", OUTPUT_FILE)

    for strategy in STRATEGIES:
        count = sum(
            1 for r in successful
            if r["strategy"] == strategy
        )
        print(f"{strategy}: {count}/40")

    print("=========================================================")


if __name__ == "__main__":
    main()
