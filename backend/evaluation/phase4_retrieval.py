import logging
from collections import Counter, defaultdict
from typing import Any

from langchain_core.documents import Document

from retrieval import (
    dense_search,
    sparse_search,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


def document_identity(document: Document) -> tuple[str, str]:
    """
    Return a stable document identity using its ID and version.
    """
    metadata = document.metadata

    return (
        str(metadata.get("document_id", "unknown")),
        str(metadata.get("document_version", "unknown")),
    )


def retrieve_phase4_candidates(
    query: str,
    retrievers: dict[str, Any],
    dense_top_k: int = 100,
    sparse_top_k: int = 100,
    rrf_k: int = 60,
    candidate_count: int = 100,
) -> dict[str, Any]:
    """
    Experimental hybrid retrieval with expanded candidate depth.

    This function does not modify production retrieval settings.
    It temporarily adjusts BM25 k for this isolated experiment.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if min(
        dense_top_k,
        sparse_top_k,
        candidate_count,
        rrf_k,
    ) <= 0:
        raise ValueError("Retrieval parameters must be positive")

    bm25 = retrievers["bm25_retriever"]

    # Preserve the original BM25 configuration.
    original_bm25_k = bm25.k

    try:
        # BM25Retriever otherwise returns only its configured k,
        # even when sparse_search requests more results.
        bm25.k = sparse_top_k

        dense_results = dense_search(
            query=query,
            vectors=retrievers["vectors"],
            embeddings=retrievers["embeddings"],
            chunks=retrievers["chunks"],
            top_k=dense_top_k,
        )

        sparse_results = sparse_search(
            query=query,
            bm25_retriever=bm25,
            top_k=sparse_top_k,
        )

    finally:
        bm25.k = original_bm25_k

    fused_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=rrf_k,
        top_k=candidate_count,
    )

    return {
        "query": query,
        "dense_results": dense_results,
        "sparse_results": sparse_results,
        "fused_results": fused_results,
        "statistics": {
            "dense_count": len(dense_results),
            "sparse_count": len(sparse_results),
            "fused_count": len(fused_results),
            "unique_documents": len({
                document_identity(item["document"])
                for item in fused_results
            }),
        },
    }


def analyze_document_distribution(
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Analyze how retrieved candidates are distributed across documents.

    This is diagnostic only. It does not alter candidate rankings.
    """

    document_counts = Counter()
    document_names = {}

    for rank, candidate in enumerate(candidates, start=1):
        document = candidate["document"]
        identity = document_identity(document)

        document_counts[identity] += 1

        document_names[identity] = (
            document.metadata.get("file_name", "Unknown")
        )

    total = len(candidates)

    distribution = []

    for identity, count in document_counts.most_common():
        distribution.append({
            "document_id": identity[0],
            "document_version": identity[1],
            "file_name": document_names[identity],
            "candidate_count": count,
            "candidate_percentage": (
                round(count / total * 100, 2)
                if total else 0.0
            ),
        })

    return {
        "total_candidates": total,
        "unique_documents": len(document_counts),
        "distribution": distribution,
    }


def inspect_candidate_metadata(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Export candidate metadata for manual inspection.
    """

    rows = []

    for rank, candidate in enumerate(candidates, start=1):
        document = candidate["document"]
        metadata = document.metadata

        rows.append({
            "rank": rank,
            "chunk_id": metadata.get("chunk_id"),
            "document_id": metadata.get("document_id"),
            "document_version": metadata.get("document_version"),
            "file_name": metadata.get("file_name"),
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "rrf_score": candidate.get("rrf_score"),
            "retrieved_by": ",".join(
                candidate.get("retrieved_by", [])
            ),
            "text_preview": document.page_content[:350],
        })

    return rows
