
import logging
import re
from typing import Any

from sentence_transformers import CrossEncoder

from config import (
    RERANKER_MODEL,
    RERANK_TOP_K,
    MAX_CONTEXT_TOKENS
)

logger = logging.getLogger(__name__)

# Shared model instance within the application process.
_cross_encoder = None


# --------------------------------------------------
# MODEL INITIALIZATION
# --------------------------------------------------

def get_reranker() -> CrossEncoder:
    """
    Lazy-load the CrossEncoder.

    The model is loaded only when reranking is first requested.
    """

    global _cross_encoder

    if _cross_encoder is None:

        logger.info(
            "Loading CrossEncoder: %s",
            RERANKER_MODEL
        )

        _cross_encoder = CrossEncoder(
            RERANKER_MODEL
        )

    return _cross_encoder


# --------------------------------------------------
# TOKEN ESTIMATION
# --------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Approximate token count.

    This is a conservative character-based estimate,
    not an exact tokenizer count.
    """

    return max(1, (len(text) + 2) // 3)


# --------------------------------------------------
# TEXT NORMALIZATION
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize whitespace for duplicate detection."""

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower()
    )


# --------------------------------------------------
# CROSS ENCODER RERANKING
# --------------------------------------------------

def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    model: CrossEncoder | None = None
) -> list[dict[str, Any]]:
    """
    Score RRF candidates using a CrossEncoder.

    Each candidate contains:
        chunk_id
        document
        rrf_score
        retrieved_by
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if not candidates:
        return []

    model = model or get_reranker()

    pairs = []

    for candidate in candidates:

        document = candidate["document"]

        pairs.append((
            query,
            document.page_content
        ))

    scores = model.predict(
        pairs,
        show_progress_bar=False
    )

    reranked = []

    for candidate, score in zip(candidates, scores):

        item = dict(candidate)

        item["rerank_score"] = float(score)

        reranked.append(item)

    reranked.sort(
        key=lambda item: item["rerank_score"],
        reverse=True
    )

    logger.info(
        "CrossEncoder reranked %d candidates",
        len(reranked)
    )

    return reranked


# --------------------------------------------------
# CONTEXT SELECTION
# --------------------------------------------------

def select_context(
    reranked_candidates: list[dict[str, Any]],
    top_k: int = RERANK_TOP_K,
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> list[dict[str, Any]]:
    """
    Select the most relevant chunks within the context budget.

    Removes exact normalized duplicates and avoids exceeding
    the configured approximate token budget.
    """

    if top_k <= 0:
        return []

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    selected = []
    seen_content = set()

    total_tokens = 0

    for candidate in reranked_candidates:

        document = candidate["document"]

        content = document.page_content.strip()

        if not content:
            continue

        normalized = normalize_text(content)

        if normalized in seen_content:
            continue

        chunk_tokens = estimate_tokens(content)

        # Never exceed the context budget.
        if total_tokens + chunk_tokens > max_tokens:
            continue

        selected.append(candidate)

        seen_content.add(normalized)

        total_tokens += chunk_tokens

        if len(selected) >= top_k:
            break

    logger.info(
        "Selected %d chunks using approximately %d/%d tokens",
        len(selected),
        total_tokens,
        max_tokens
    )

    return selected


# --------------------------------------------------
# COMPLETE RERANKING PIPELINE
# --------------------------------------------------

def rerank_and_select(
    query: str,
    candidates: list[dict[str, Any]],
    model: CrossEncoder | None = None,
    top_k: int = RERANK_TOP_K,
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> dict[str, Any]:
    """
    Rerank candidates and select final context chunks.
    """

    reranked = rerank_candidates(
        query=query,
        candidates=candidates,
        model=model
    )

    selected = select_context(
        reranked_candidates=reranked,
        top_k=top_k,
        max_tokens=max_tokens
    )

    return {
        "reranked_candidates": reranked,
        "selected_context": selected,
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "estimated_context_tokens": sum(
            estimate_tokens(
                item["document"].page_content
            )
            for item in selected
        )
    }


# --------------------------------------------------
# FORMAT CONTEXT FOR LLM
# --------------------------------------------------

def format_context(
    selected_context: list[dict[str, Any]]
) -> str:
    """
    Format selected documents for the LLM.

    Source metadata is retained for citation generation.
    """

    formatted = []

    for index, candidate in enumerate(
        selected_context,
        start=1
    ):

        document = candidate["document"]

        metadata = document.metadata

        file_name = metadata.get(
            "file_name",
            "Unknown source"
        )

        page = metadata.get("page")

        source_label = file_name

        if page is not None:
            source_label += f", page {int(page) + 1}"

        formatted.append(
            f"[Evidence {index}]\n"
            f"Source: {source_label}\n"
            f"Content:\n{document.page_content}\n"
        )

    return "\n".join(formatted)
