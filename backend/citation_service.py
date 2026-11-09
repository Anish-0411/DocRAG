import hashlib
import logging
import re
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Matches [EVD-XXXXXXXXXXXX], [ EVD-XXXXXXXXXXXX ],
# and the equivalent full-width brackets.
CITATION_PATTERN = re.compile(
    r"[\[【]\s*(EVD-[A-F0-9]{12})\s*[\]】]",
    flags=re.IGNORECASE
)


def generate_evidence_id(chunk_id: str) -> str:
    """Generate a stable evidence identifier from chunk_id."""
    digest = hashlib.sha256(
        chunk_id.encode("utf-8")
    ).hexdigest()[:12]

    return f"EVD-{digest.upper()}"


def get_page_number(metadata: dict[str, Any]) -> int | None:
    """Convert zero-based PDF page index to one-based page number."""
    page = metadata.get("page")

    if page is None:
        return None

    try:
        return int(page) + 1
    except (TypeError, ValueError):
        return None


def get_chunk_id(document: Document) -> str:
    """Obtain the stable chunk identifier."""
    chunk_id = document.metadata.get("chunk_id")

    if not chunk_id:
        raise ValueError("Document is missing chunk_id metadata")

    return str(chunk_id)


def build_evidence_records(
    selected_context: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert selected chunks into provenance-rich evidence records."""
    evidence_records = []

    for candidate in selected_context:
        document = candidate["document"]
        metadata = document.metadata

        chunk_id = get_chunk_id(document)
        evidence_id = generate_evidence_id(chunk_id)

        evidence_records.append({
            "evidence_id": evidence_id,
            "chunk_id": chunk_id,
            "document_id": metadata.get("document_id"),
            "document_version": metadata.get("document_version"),
            "file_name": metadata.get("file_name"),
            "file_type": metadata.get("file_type"),
            "source_path": metadata.get("source_path"),
            "page": get_page_number(metadata),
            "chunk_index": metadata.get("chunk_index"),
            "start_index": metadata.get("start_index"),
            "content_hash": metadata.get("content_hash"),
            "retrieved_by": candidate.get("retrieved_by", []),
            "rrf_score": candidate.get("rrf_score"),
            "rerank_score": candidate.get("rerank_score"),
            "content": document.page_content,
        })

    return evidence_records


def format_evidence_for_prompt(
    evidence_records: list[dict[str, Any]]
) -> str:
    """Format evidence with explicit identifiers for the LLM."""
    formatted = []

    for evidence in evidence_records:
        source = evidence.get("file_name") or "Unknown source"
        page = evidence.get("page")

        if page is not None:
            source += f", page {page}"

        formatted.append(
            f"[{evidence['evidence_id']}]\n"
            f"Source: {source}\n"
            f"Content:\n{evidence['content']}\n"
        )

    return "\n".join(formatted)


def extract_citation_ids(
    answer: str,
    evidence_records: list[dict[str, Any]]
) -> list[str]:
    """Extract only citation IDs belonging to the current evidence set."""
    available_ids = {
        item["evidence_id"].upper()
        for item in evidence_records
    }

    mentioned_ids = CITATION_PATTERN.findall(answer or "")

    result = []

    for evidence_id in mentioned_ids:
        evidence_id = evidence_id.upper()

        if evidence_id in available_ids and evidence_id not in result:
            result.append(evidence_id)

    return result


def normalize_answer_citations(
    answer: str,
    evidence_records: list[dict[str, Any]]
) -> str:
    """
    Normalize valid citations to [EVD-XXXXXXXXXXXX].
    Remove citation-shaped references that are not in the current
    evidence set, so unsupported evidence IDs are not presented.
    """
    available_ids = {
        item["evidence_id"].upper()
        for item in evidence_records
    }

    def replace_reference(match: re.Match) -> str:
        evidence_id = match.group(1).upper()

        if evidence_id not in available_ids:
            logger.warning(
                "Removing citation not present in current evidence: %s",
                evidence_id
            )
            return ""

        return f"[{evidence_id}]"

    return CITATION_PATTERN.sub(replace_reference, answer or "")


def build_citations(
    evidence_records: list[dict[str, Any]],
    cited_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Build frontend-friendly evidence records."""
    cited_set = (
        {item.upper() for item in cited_ids}
        if cited_ids is not None
        else None
    )

    citations = []
    seen = set()

    for evidence in evidence_records:
        evidence_id = evidence["evidence_id"].upper()

        if cited_set is not None and evidence_id not in cited_set:
            continue

        identity = (
            evidence.get("document_id"),
            evidence.get("document_version"),
            evidence.get("page"),
            evidence.get("chunk_id"),
        )

        # If provenance metadata is missing, use evidence_id
        # so distinct evidence records are never collapsed.
        if all(value is None for value in identity):
            identity = (evidence_id,)

        if identity in seen:
            continue

        seen.add(identity)

        citations.append({
            "id": evidence_id,
            "evidence_id": evidence_id,
            "source": evidence.get("file_name"),
            "file_name": evidence.get("file_name"),
            "file_type": evidence.get("file_type"),
            "source_path": evidence.get("source_path"),
            "document_id": evidence.get("document_id"),
            "document_version": evidence.get("document_version"),
            "chunk_id": evidence.get("chunk_id"),
            "page": evidence.get("page"),
            "chunk_index": evidence.get("chunk_index"),
            "start_index": evidence.get("start_index"),
            "retrieved_by": evidence.get("retrieved_by", []),
            "rrf_score": evidence.get("rrf_score"),
            "rerank_score": evidence.get("rerank_score"),
            "content": evidence.get("content"),
            "cited": False,
        })

    return citations


def prepare_citations(
    selected_context: list[dict[str, Any]]
) -> dict[str, Any]:
    """Prepare evidence records and prompt context."""
    evidence_records = build_evidence_records(selected_context)

    evidence_context = format_evidence_for_prompt(evidence_records)

    return {
        "evidence_records": evidence_records,
        "evidence_context": evidence_context,
        "retrieved_evidence_count": len(evidence_records),
    }


def finalize_citations(
    answer: str,
    evidence_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Validate, normalize and associate answer citations with evidence."""
    normalized_answer = normalize_answer_citations(
        answer,
        evidence_records
    )

    cited_ids = extract_citation_ids(
        normalized_answer,
        evidence_records
    )

    citations = build_citations(evidence_records)
    cited_set = set(cited_ids)

    for citation in citations:
        citation["cited"] = citation["evidence_id"] in cited_set

    cited_count = sum(
        1 for citation in citations if citation["cited"]
    )

    return {
        "answer": normalized_answer,
        "citations": citations,
        "retrieved_evidence_count": len(evidence_records),
        "cited_evidence_count": cited_count,
        "uncited_evidence_count": len(evidence_records) - cited_count,
    }