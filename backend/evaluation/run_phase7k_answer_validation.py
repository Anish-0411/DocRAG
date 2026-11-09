import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
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
QUESTIONS_FILE = EVALUATION_DIR / "phase7_regression_questions.jsonl"
OUTPUT_FILE = EVALUATION_DIR / "phase7k_answers.jsonl"

EXPERIMENTAL_DEPTH = 100

STRATEGIES = (
    "production_baseline",
    "metadata_experimental",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("phase7k")


def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def atomic_write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp:
        for record in records:
            temp.write(json.dumps(record, ensure_ascii=False) + "\n")

        temp_path = Path(temp.name)

    os.replace(temp_path, path)


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
            rrf_k=60,
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
        "phase7_focus": question_data.get("phase7_focus"),
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
        raise FileNotFoundError(
            f"Regression dataset not found: {QUESTIONS_FILE}"
        )

    questions = read_jsonl(QUESTIONS_FILE)

    if len(questions) != 10:
        raise ValueError(
            f"Expected 10 regression questions, found {len(questions)}"
        )

    completed_records = load_completed()

    completed_keys = {
        (record["question_id"], record["strategy"])
        for record in completed_records
    }

    results = list(completed_records)

    print("Loading active snapshot...")
    snapshot = load_active_snapshot()
    retrievers = build_retrievers(snapshot)

    print("Initializing the same development LLM...")
    from langchain.chat_models import init_chat_model

    app.state.llm = init_chat_model(LLM_MODEL)
    app.state.retrievers = retrievers
    app.state.snapshot = snapshot

    print(f"Loading CrossEncoder: {RERANKER_MODEL}")
    cross_encoder = CrossEncoder(RERANKER_MODEL)

    original_bm25_k = retrievers["bm25_retriever"].k

    total = len(questions) * len(STRATEGIES)
    completed_count = len(completed_keys)

    print(f"\nQuestions: {len(questions)}")
    print(f"Snapshot: {snapshot['snapshot_id']}")
    print(f"Configurations: {len(STRATEGIES)}")
    print(f"Expected answers: {total}")
    print(f"Already completed: {completed_count}")
    print("Starting Phase 7K...\n")

    try:
        for question_data in questions:
            qid = question_data["question_id"]
            question = question_data["question"]

            for strategy in STRATEGIES:
                key = (qid, strategy)

                if key in completed_keys:
                    logger.info(
                        "Skipping completed %s / %s",
                        qid,
                        strategy,
                    )
                    continue

                logger.info(
                    "Generating %s / %s (%d/%d)",
                    qid,
                    strategy,
                    completed_count + 1,
                    total,
                )

                try:
                    if strategy == "production_baseline":
                        retrievers["bm25_retriever"].k = original_bm25_k

                        result = __import__(
                            "app",
                            fromlist=["run_rag"],
                        ).run_rag(question)

                    else:
                        result = generate_metadata_answer(
                            question=question,
                            retrievers=retrievers,
                            model=cross_encoder,
                        )

                    record = make_record(
                        question_data,
                        strategy,
                        result,
                        snapshot["snapshot_id"],
                    )

                    results.append(record)
                    completed_keys.add(key)
                    completed_count += 1

                    atomic_write_jsonl(results, OUTPUT_FILE)

                    logger.info(
                        "Saved %s / %s | cited=%d | selected=%d",
                        qid,
                        strategy,
                        result.get("retrieval", {}).get(
                            "cited_evidence_count", 0
                        ),
                        result.get("retrieval", {}).get(
                            "selected_count", 0
                        ),
                    )

                except Exception as error:
                    logger.exception(
                        "Failed %s / %s",
                        qid,
                        strategy,
                    )

                    results.append({
                        "question_id": qid,
                        "strategy": strategy,
                        "question": question,
                        "error": repr(error),
                    })

                    atomic_write_jsonl(results, OUTPUT_FILE)

                finally:
                    results = [
                        record
                        for record in results
                        if not record.get("error")
                    ]

    finally:
        retrievers["bm25_retriever"].k = original_bm25_k

    print("\n========== PHASE 7K COMPLETE ==========")
    print(f"Successful answers: {len(completed_keys)}/{total}")
    print(f"Output: {OUTPUT_FILE}")

    if len(completed_keys) != total:
        print(
            "Some configurations are incomplete. "
            "Run this script again to retry failed configurations."
        )


if __name__ == "__main__":
    main()
