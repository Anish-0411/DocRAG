import logging
from collections import defaultdict
from typing import Any

import numpy as np

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from config import (
    DENSE_TOP_K,
    SPARSE_TOP_K,
    RRF_K,
    RERANK_CANDIDATES,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# BUILD RETRIEVERS
# --------------------------------------------------

def build_bm25_retriever(
    chunks: list[Document],
) -> BM25Retriever:
    """Build a sparse BM25 retriever from snapshot chunks."""
    if not chunks:
        raise ValueError("Cannot build BM25 without chunks")

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = SPARSE_TOP_K

    logger.info(
        "BM25 retriever initialized with %d chunks",
        len(chunks),
    )
    return retriever


def build_retrievers(
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Build NumPy dense and BM25 sparse retrievers."""
    vectors = snapshot.get("vectors")
    chunks = snapshot.get("chunks")
    embeddings = snapshot.get("embeddings")

    if vectors is None:
        raise ValueError("Snapshot has no NumPy vector matrix")
    if not chunks:
        raise ValueError("Snapshot has no chunks")
    if embeddings is None:
        raise ValueError("Snapshot has no embedding model")

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Snapshot vectors must be a 2D matrix")
    if vectors.shape[0] != len(chunks):
        raise ValueError("Vector and chunk counts do not match")
    if not np.isfinite(vectors).all():
        raise ValueError("Snapshot vectors contain invalid values")

    bm25_retriever = build_bm25_retriever(chunks)

    return {
        "snapshot_id": snapshot["snapshot_id"],
        "vectors": vectors,
        "embeddings": embeddings,
        "bm25_retriever": bm25_retriever,
        "chunks": chunks,
    }


# --------------------------------------------------
# DOCUMENT IDENTITY
# --------------------------------------------------

def get_chunk_identity(document: Document) -> str:
    """Obtain a stable identity for a retrieved chunk."""
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)

    document_id = document.metadata.get(
        "document_id",
        document.metadata.get("file_name", "unknown"),
    )
    page = document.metadata.get("page", "none")
    start_index = document.metadata.get("start_index", "none")

    return (
        f"{document_id}:"
        f"{page}:"
        f"{start_index}:"
        f"{hash(document.page_content)}"
    )


# --------------------------------------------------
# DENSE RETRIEVAL - EXACT NUMPY SQUARED L2
# --------------------------------------------------

def dense_search(
    query: str,
    vectors: np.ndarray,
    embeddings,
    chunks: list[Document],
    top_k: int = DENSE_TOP_K,
) -> list[Document]:
    """
    Retrieve the nearest chunks using exact squared L2 distance.

    The existing FAISS index was IndexFlatL2, so ascending squared
    Euclidean distance preserves its ranking semantics.
    """
    if not query or not query.strip():
        raise ValueError("Dense search query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    vectors = np.asarray(vectors, dtype=np.float32)

    if vectors.ndim != 2:
        raise ValueError("Vector matrix must be two-dimensional")
    if vectors.shape[0] != len(chunks):
        raise ValueError("Vector and chunk counts do not match")
    if vectors.shape[0] == 0:
        return []

    query_vector = np.asarray(
        embeddings.embed_query(query),
        dtype=np.float32,
    )

    if query_vector.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional")
    if query_vector.shape[0] != vectors.shape[1]:
        raise ValueError(
            "Query embedding dimension does not match stored vectors"
        )
    if not np.isfinite(query_vector).all():
        raise ValueError("Query embedding contains invalid values")

    # ||x - q||^2 = ||x||^2 + ||q||^2 - 2(x.q)
    # This avoids allocating a 12,124 x 384 difference matrix.
    vector_norms = np.einsum("ij,ij->i", vectors, vectors)
    query_norm = float(np.dot(query_vector, query_vector))
    dot_products = vectors @ query_vector

    distances = vector_norms + query_norm - (2.0 * dot_products)
    np.maximum(distances, 0.0, out=distances)

    result_count = min(top_k, len(chunks))
    top_indices = np.argsort(distances, kind="stable")[:result_count]

    results = [chunks[int(index)] for index in top_indices]

    logger.info(
        "NumPy dense retrieval returned %d chunks",
        len(results),
    )
    return results


# --------------------------------------------------
# SPARSE RETRIEVAL
# --------------------------------------------------

def sparse_search(
    query: str,
    bm25_retriever: BM25Retriever,
    top_k: int = SPARSE_TOP_K,
) -> list[Document]:
    """Retrieve keyword-relevant chunks using BM25."""
    results = bm25_retriever.invoke(query)
    results = results[:top_k]

    logger.info(
        "Sparse retrieval returned %d chunks",
        len(results),
    )
    return results


# --------------------------------------------------
# RECIPROCAL RANK FUSION
# --------------------------------------------------

def reciprocal_rank_fusion(
    dense_results: list[Document],
    sparse_results: list[Document],
    rrf_k: int = RRF_K,
    top_k: int = RERANK_CANDIDATES,
) -> list[dict[str, Any]]:
    """Combine dense and sparse rankings using RRF."""
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero")

    scores = defaultdict(float)
    documents = {}
    retrieval_sources = defaultdict(set)

    retrieval_lists = [
        ("dense", dense_results),
        ("sparse", sparse_results),
    ]

    for source, results in retrieval_lists:
        for rank, document in enumerate(results, start=1):
            chunk_id = get_chunk_identity(document)

            scores[chunk_id] += 1.0 / (rrf_k + rank)
            retrieval_sources[chunk_id].add(source)

            if chunk_id not in documents:
                documents[chunk_id] = document

    fused_results = []
    for chunk_id, score in scores.items():
        fused_results.append({
            "chunk_id": chunk_id,
            "document": documents[chunk_id],
            "rrf_score": score,
            "retrieved_by": sorted(retrieval_sources[chunk_id]),
        })

    fused_results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )
    fused_results = fused_results[:top_k]

    logger.info(
        "RRF produced %d unique candidates",
        len(fused_results),
    )
    return fused_results


# --------------------------------------------------
# HYBRID RETRIEVAL
# --------------------------------------------------

def retrieve_candidates(
    query: str,
    retrievers: dict[str, Any],
    dense_top_k: int = DENSE_TOP_K,
    sparse_top_k: int = SPARSE_TOP_K,
    rrf_k: int = RRF_K,
    candidate_count: int = RERANK_CANDIDATES,
) -> list[dict[str, Any]]:
    """Execute dense retrieval, sparse retrieval and RRF."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    dense_results = dense_search(
        query=query,
        vectors=retrievers["vectors"],
        embeddings=retrievers["embeddings"],
        chunks=retrievers["chunks"],
        top_k=dense_top_k,
    )

    sparse_results = sparse_search(
        query=query,
        bm25_retriever=retrievers["bm25_retriever"],
        top_k=sparse_top_k,
    )

    return reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=rrf_k,
        top_k=candidate_count,
    )
