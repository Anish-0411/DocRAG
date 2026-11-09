import hashlib
import logging
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    SUPPORTED_EXTENSIONS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.0"
CHUNKER_VERSION = "1.0"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_id(value: str) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def discover_files() -> list[Path]:
    """
    Discover supported files directly inside data/.
    Nested directories are deliberately ignored.
    """

    files = []

    for path in DATA_DIR.iterdir():

        if not path.is_file():
            continue

        if path.name.startswith("."):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files, key=lambda p: p.name.lower())


def validate_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "File does not exist"

    if not path.is_file():
        return False, "Not a regular file"

    if path.stat().st_size == 0:
        return False, "File is empty"

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, "Unsupported file type"

    return True, "Valid"


def clean_text(text: str) -> str:
    """
    Conservative cleaning.

    Preserve paragraph boundaries and meaningful punctuation.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        " ".join(line.split())
        for line in text.split("\n")
    ]

    cleaned = "\n".join(lines)

    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")

    return cleaned.strip()


def parse_document(path: Path) -> list[Document]:

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyMuPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8"
        )

    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    if not documents:
        raise ValueError("Parser returned no documents")

    for document in documents:
        document.page_content = clean_text(
            document.page_content
        )

    valid_documents = [
        document
        for document in documents
        if document.page_content.strip()
    ]

    if not valid_documents:
        if extension == ".pdf":
            raise ValueError(
                "PDF extraction produced no text. "
                "It may be scanned or image-only; OCR is not "
                "implemented yet."
            )

        raise ValueError(
            "Document extraction produced no usable text"
        )

    return valid_documents


def attach_document_metadata(
    documents: list[Document],
    path: Path,
    file_hash: str
) -> list[Document]:

    document_id = stable_id(
        path.name.lower()
    )

    file_type = path.suffix.lower().lstrip(".")

    for paragraph_index, document in enumerate(documents):

        document.metadata.update({
            "document_id": document_id,
            "document_version": file_hash,
            "file_name": path.name,
            "file_type": file_type,
            "source_path": path.name,
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "paragraph_index": paragraph_index
        })

        if file_type == "pdf":
            document.metadata["page"] = document.metadata.get(
                "page",
                None
            )

    return documents


def create_chunks(
    documents: list[Document]
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):

        document_id = chunk.metadata["document_id"]
        document_version = chunk.metadata["document_version"]

        chunk_content_hash = sha256_bytes(
            chunk.page_content.encode("utf-8")
        )

        identity = (
            f"{document_id}:"
            f"{document_version}:"
            f"{chunk.metadata.get('page', 'none')}:"
            f"{chunk.metadata.get('start_index', index)}:"
            f"{chunk_content_hash}"
        )

        chunk.metadata.update({
            "chunk_id": stable_id(identity),
            "chunk_index": index,
            "content_hash": chunk_content_hash
        })

    return chunks


def ingest_documents() -> dict[str, Any]:

    files = discover_files()

    all_chunks = []
    report = {
        "discovered": len(files),
        "successful": [],
        "failed": [],
        "total_chunks": 0
    }

    for path in files:

        try:
            valid, reason = validate_file(path)

            if not valid:
                raise ValueError(reason)

            file_bytes = path.read_bytes()
            file_hash = sha256_bytes(file_bytes)

            documents = parse_document(path)

            documents = attach_document_metadata(
                documents,
                path,
                file_hash
            )

            chunks = create_chunks(documents)

            if not chunks:
                raise ValueError(
                    "Chunking produced no chunks"
                )

            all_chunks.extend(chunks)

            report["successful"].append({
                "file_name": path.name,
                "document_id": documents[0].metadata[
                    "document_id"
                ],
                "document_version": file_hash,
                "pages_or_sections": len(documents),
                "chunks": len(chunks)
            })

            logger.info(
                "Ingested %s: %d chunks",
                path.name,
                len(chunks)
            )

        except Exception as error:

            report["failed"].append({
                "file_name": path.name,
                "error": str(error)
            })

            logger.exception(
                "Failed to ingest %s",
                path.name
            )

    if not all_chunks:
        raise ValueError(
            "No usable document chunks were produced. "
            "No index should be published."
        )

    report["total_chunks"] = len(all_chunks)

    return {
        "chunks": all_chunks,
        "report": report
    }