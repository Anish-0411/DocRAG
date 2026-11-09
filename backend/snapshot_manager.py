
# import hashlib
# import json
# import logging
# import os
# import shutil
# import tempfile
# import uuid

# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any

# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# from config import (
#     INDEX_DIR,
#     SNAPSHOT_DIR,
#     CURRENT_SNAPSHOT_FILE,
#     EMBEDDING_MODEL
# )

# logger = logging.getLogger(__name__)

# SNAPSHOT_SCHEMA_VERSION = "1.0"


# # --------------------------------------------------
# # UTILITIES
# # --------------------------------------------------

# def calculate_file_hash(path: Path) -> str:
#     """Calculate SHA-256 checksum of a file."""

#     sha256 = hashlib.sha256()

#     with path.open("rb") as file:
#         for block in iter(lambda: file.read(1024 * 1024), b""):
#             sha256.update(block)

#     return sha256.hexdigest()


# def get_embeddings():
#     """Create the configured embedding model."""

#     return HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL
#     )


# def snapshot_path(snapshot_id: str) -> Path:
#     """Resolve a snapshot ID safely."""

#     if not snapshot_id or Path(snapshot_id).name != snapshot_id:
#         raise ValueError("Invalid snapshot ID")

#     return SNAPSHOT_DIR / snapshot_id


# def serialize_chunks(chunks: list[Document], output_path: Path):
#     """Persist document chunks as JSONL."""

#     with output_path.open("w", encoding="utf-8") as file:

#         for chunk in chunks:

#             record = {
#                 "page_content": chunk.page_content,
#                 "metadata": chunk.metadata
#             }

#             file.write(
#                 json.dumps(record, ensure_ascii=False, default=str)
#                 + "\n"
#             )


# def load_chunks(input_path: Path) -> list[Document]:
#     """Reconstruct LangChain Documents from JSONL."""

#     chunks = []

#     with input_path.open("r", encoding="utf-8") as file:

#         for line in file:

#             if not line.strip():
#                 continue

#             record = json.loads(line)

#             chunks.append(
#                 Document(
#                     page_content=record["page_content"],
#                     metadata=record["metadata"]
#                 )
#             )

#     return chunks


# # --------------------------------------------------
# # SNAPSHOT MANIFEST
# # --------------------------------------------------

# def create_manifest(
#     snapshot_id: str,
#     chunks: list[Document],
#     ingestion_report: dict[str, Any],
#     snapshot_dir: Path
# ) -> dict[str, Any]:

#     files = [
#         "index.faiss",
#         "index.pkl",
#         "chunks.jsonl"
#     ]

#     checksums = {}

#     for filename in files:

#         file_path = snapshot_dir / filename

#         if not file_path.exists():
#             raise FileNotFoundError(
#                 f"Snapshot artifact missing: {filename}"
#             )

#         checksums[filename] = calculate_file_hash(file_path)

#     return {
#         "schema_version": SNAPSHOT_SCHEMA_VERSION,
#         "snapshot_id": snapshot_id,
#         "created_at": datetime.now(timezone.utc).isoformat(),
#         "embedding_model": EMBEDDING_MODEL,
#         "chunk_count": len(chunks),
#         "document_count": len({
#             chunk.metadata.get("document_id")
#             for chunk in chunks
#         }),
#         "ingestion_report": ingestion_report,
#         "checksums": checksums
#     }


# # --------------------------------------------------
# # BUILD SNAPSHOT
# # --------------------------------------------------

# def build_snapshot(
#     chunks: list[Document],
#     ingestion_report: dict[str, Any],
#     embeddings=None
# ) -> str:
#     """
#     Build a new snapshot without changing the active snapshot.

#     Returns the generated snapshot ID.
#     """

#     if not chunks:
#         raise ValueError(
#             "Cannot build a snapshot without document chunks"
#         )

#     embeddings = embeddings or get_embeddings()

#     snapshot_id = (
#         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
#         + "_"
#         + uuid.uuid4().hex[:8]
#     )

#     temporary_dir = Path(
#         tempfile.mkdtemp(
#             prefix=".building_",
#             dir=SNAPSHOT_DIR
#         )
#     )

#     final_dir = snapshot_path(snapshot_id)

#     try:

#         logger.info(
#             "Building snapshot %s with %d chunks",
#             snapshot_id,
#             len(chunks)
#         )

#         # 1. Build FAISS index
#         vectorstore = FAISS.from_documents(
#             documents=chunks,
#             embedding=embeddings
#         )

#         # 2. Persist FAISS artifacts
#         vectorstore.save_local(
#             str(temporary_dir)
#         )

#         # 3. Persist original chunks and metadata
#         serialize_chunks(
#             chunks,
#             temporary_dir / "chunks.jsonl"
#         )

#         # 4. Generate manifest and checksums
#         manifest = create_manifest(
#             snapshot_id=snapshot_id,
#             chunks=chunks,
#             ingestion_report=ingestion_report,
#             snapshot_dir=temporary_dir
#         )

#         with (temporary_dir / "manifest.json").open(
#             "w",
#             encoding="utf-8"
#         ) as file:

#             json.dump(
#                 manifest,
#                 file,
#                 indent=4,
#                 ensure_ascii=False
#             )

#         # 5. Validate before finalizing
#         validation = validate_snapshot(
#             temporary_dir,
#             expected_snapshot_id=snapshot_id
#         )

#         if not validation["valid"]:
#             raise ValueError(
#                 f"Snapshot validation failed: {validation['errors']}"
#             )

#         # 6. Publish directory under its permanent snapshot ID.
#         # The active snapshot pointer is NOT changed here.
#         if final_dir.exists():
#             raise FileExistsError(final_dir)

#         os.replace(
#             str(temporary_dir),
#             str(final_dir)
#         )

#         logger.info(
#             "Snapshot built successfully: %s",
#             snapshot_id
#         )

#         return snapshot_id

#     except Exception:

#         logger.exception(
#             "Snapshot build failed: %s",
#             snapshot_id
#         )

#         raise

#     finally:

#         if temporary_dir.exists():
#             shutil.rmtree(
#                 temporary_dir,
#                 ignore_errors=True
#             )


# # --------------------------------------------------
# # VALIDATE SNAPSHOT
# # --------------------------------------------------

# def validate_snapshot(
#     snapshot: str | Path,
#     expected_snapshot_id: str | None = None
# ) -> dict[str, Any]:
#     """
#     Validate manifest, artifacts and file checksums.

#     Accepts either a snapshot ID or a snapshot directory.
#     """

#     snapshot_dir = (
#         snapshot_path(snapshot)
#         if isinstance(snapshot, str)
#         else Path(snapshot)
#     )

#     errors = []

#     manifest_path = snapshot_dir / "manifest.json"

#     if not snapshot_dir.is_dir():
#         return {
#             "valid": False,
#             "errors": ["Snapshot directory does not exist"]
#         }

#     if not manifest_path.exists():
#         return {
#             "valid": False,
#             "errors": ["Manifest is missing"]
#         }

#     try:

#         with manifest_path.open(
#             "r",
#             encoding="utf-8"
#         ) as file:

#             manifest = json.load(file)

#     except Exception as error:

#         return {
#             "valid": False,
#             "errors": [f"Invalid manifest: {error}"]
#         }

#     if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
#         errors.append("Unsupported snapshot schema version")

#     if (
#         expected_snapshot_id is not None
#         and manifest.get("snapshot_id") != expected_snapshot_id
#     ):
#         errors.append("Snapshot ID mismatch")

#     checksums = manifest.get("checksums", {})

#     required_files = [
#         "index.faiss",
#         "index.pkl",
#         "chunks.jsonl"
#     ]

#     for filename in required_files:

#         file_path = snapshot_dir / filename

#         if not file_path.is_file():
#             errors.append(f"Missing artifact: {filename}")
#             continue

#         expected_hash = checksums.get(filename)

#         if not expected_hash:
#             errors.append(f"Missing checksum: {filename}")
#             continue

#         actual_hash = calculate_file_hash(file_path)

#         if actual_hash != expected_hash:
#             errors.append(f"Checksum mismatch: {filename}")

#     # Verify chunk count against persisted records.
#     chunks_path = snapshot_dir / "chunks.jsonl"

#     if chunks_path.exists():

#         try:

#             chunks = load_chunks(chunks_path)

#             if len(chunks) != manifest.get("chunk_count"):
#                 errors.append("Chunk count mismatch")

#             chunk_ids = [
#                 chunk.metadata.get("chunk_id")
#                 for chunk in chunks
#             ]

#             if any(chunk_id is None for chunk_id in chunk_ids):
#                 errors.append("One or more chunks have no chunk_id")

#             if len(chunk_ids) != len(set(chunk_ids)):
#                 errors.append("Duplicate chunk IDs detected")

#         except Exception as error:

#             errors.append(
#                 f"Could not read persisted chunks: {error}"
#             )

#     return {
#         "valid": len(errors) == 0,
#         "snapshot_id": manifest.get("snapshot_id"),
#         "chunk_count": manifest.get("chunk_count"),
#         "errors": errors
#     }


# # --------------------------------------------------
# # PUBLISH SNAPSHOT
# # --------------------------------------------------

# def publish_snapshot(snapshot_id: str) -> dict[str, Any]:
#     """
#     Atomically update the active snapshot pointer.

#     The existing active snapshot remains untouched
#     if validation or pointer publication fails.
#     """

#     snapshot_dir = snapshot_path(snapshot_id)

#     validation = validate_snapshot(
#         snapshot_dir,
#         expected_snapshot_id=snapshot_id
#     )

#     if not validation["valid"]:
#         raise ValueError(
#             f"Cannot publish invalid snapshot: {validation['errors']}"
#         )

#     temporary_pointer = (
#         INDEX_DIR / f".current_snapshot_{uuid.uuid4().hex}.tmp"
#     )

#     try:

#         temporary_pointer.write_text(
#             snapshot_id,
#             encoding="utf-8"
#         )

#         # Atomic pointer replacement on the same filesystem.
#         os.replace(
#             str(temporary_pointer),
#             str(CURRENT_SNAPSHOT_FILE)
#         )

#         logger.info(
#             "Published active snapshot: %s",
#             snapshot_id
#         )

#     finally:

#         if temporary_pointer.exists():
#             temporary_pointer.unlink()

#     return {
#         "published": True,
#         "snapshot_id": snapshot_id,
#         "chunk_count": validation["chunk_count"]
#     }


# # --------------------------------------------------
# # LOAD ACTIVE SNAPSHOT
# # --------------------------------------------------

# def load_active_snapshot(embeddings=None) -> dict[str, Any]:
#     """
#     Load the currently published FAISS index and chunks.
#     """

#     if not CURRENT_SNAPSHOT_FILE.exists():
#         raise FileNotFoundError(
#             "No active snapshot found. Run ingestion first."
#         )

#     snapshot_id = CURRENT_SNAPSHOT_FILE.read_text(
#         encoding="utf-8"
#     ).strip()

#     if not snapshot_id:
#         raise ValueError("Active snapshot pointer is empty")

#     snapshot_dir = snapshot_path(snapshot_id)

#     validation = validate_snapshot(
#         snapshot_dir,
#         expected_snapshot_id=snapshot_id
#     )

#     if not validation["valid"]:
#         raise ValueError(
#             f"Active snapshot is invalid: {validation['errors']}"
#         )

#     embeddings = embeddings or get_embeddings()

#     # FAISS's local persistence uses a pickle file.
#     # Enable deserialization ONLY for snapshots created
#     # and controlled by this application.
#     vectorstore = FAISS.load_local(
#         str(snapshot_dir),
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     chunks = load_chunks(
#         snapshot_dir / "chunks.jsonl"
#     )

#     if vectorstore.index.ntotal != len(chunks):
#         raise ValueError(
#             "FAISS vector count does not match stored chunk count"
#         )

#     with (snapshot_dir / "manifest.json").open(
#         "r",
#         encoding="utf-8"
#     ) as file:

#         manifest = json.load(file)

#     logger.info(
#         "Loaded snapshot %s with %d chunks",
#         snapshot_id,
#         len(chunks)
#     )

#     return {
#         "snapshot_id": snapshot_id,
#         "vectorstore": vectorstore,
#         "chunks": chunks,
#         "manifest": manifest
#     }


# # --------------------------------------------------
# # SNAPSHOT INFORMATION
# # --------------------------------------------------

# def get_active_snapshot_id() -> str | None:
#     """Return the currently published snapshot ID."""

#     if not CURRENT_SNAPSHOT_FILE.exists():
#         return None

#     snapshot_id = CURRENT_SNAPSHOT_FILE.read_text(
#         encoding="utf-8"
#     ).strip()

#     return snapshot_id or None



# import hashlib
# import json
# import logging
# import os
# import shutil
# import tempfile
# import uuid

# from datetime import datetime, timezone
# from pathlib import Path
# from typing import Any

# from langchain_core.documents import Document
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings

# from config import (
#     INDEX_DIR,
#     SNAPSHOT_DIR,
#     CURRENT_SNAPSHOT_FILE,
#     EMBEDDING_MODEL,
# )


# logger = logging.getLogger(__name__)

# SNAPSHOT_SCHEMA_VERSION = 1

# FAISS_INDEX_FILE = "index.faiss"
# FAISS_METADATA_FILE = "index.pkl"
# CHUNKS_FILE = "chunks.jsonl"
# MANIFEST_FILE = "manifest.json"

# REQUIRED_SNAPSHOT_FILES = [
#     FAISS_INDEX_FILE,
#     FAISS_METADATA_FILE,
#     CHUNKS_FILE,
# ]

# _embeddings_instance = None


# # --------------------------------------------------
# # DIRECTORY INITIALIZATION
# # --------------------------------------------------

# def initialize_directories() -> None:
#     """Create required storage directories."""

#     INDEX_DIR.mkdir(parents=True, exist_ok=True)
#     SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

#     CURRENT_SNAPSHOT_FILE.parent.mkdir(
#         parents=True,
#         exist_ok=True
#     )


# # --------------------------------------------------
# # HASHING
# # --------------------------------------------------

# def sha256_bytes(data: bytes) -> str:
#     return hashlib.sha256(data).hexdigest()


# def calculate_file_hash(file_path: Path) -> str:
#     """Calculate SHA-256 checksum of a file."""

#     digest = hashlib.sha256()

#     with file_path.open("rb") as file:
#         for block in iter(
#             lambda: file.read(1024 * 1024),
#             b""
#         ):
#             digest.update(block)

#     return digest.hexdigest()


# # --------------------------------------------------
# # EMBEDDINGS
# # --------------------------------------------------

# def get_embeddings():
#     """Return a lazily initialized embedding model."""

#     global _embeddings_instance

#     if _embeddings_instance is None:
#         logger.info(
#             "Loading embedding model: %s",
#             EMBEDDING_MODEL
#         )

#         _embeddings_instance = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL
#         )

#     return _embeddings_instance


# # --------------------------------------------------
# # SNAPSHOT PATH VALIDATION
# # --------------------------------------------------

# def snapshot_path(snapshot_id: str) -> Path:
#     """
#     Resolve a snapshot ID safely.

#     Prevents directory traversal outside SNAPSHOT_DIR.
#     """

#     if not snapshot_id:
#         raise ValueError("Snapshot ID cannot be empty")

#     if (
#         "/" in snapshot_id
#         or "\\" in snapshot_id
#         or snapshot_id in {".", ".."}
#     ):
#         raise ValueError("Invalid snapshot ID")

#     base_dir = SNAPSHOT_DIR.resolve()
#     target = (base_dir / snapshot_id).resolve()

#     if target.parent != base_dir:
#         raise ValueError("Invalid snapshot directory")

#     return target


# # --------------------------------------------------
# # JSON SERIALIZATION
# # --------------------------------------------------

# def make_json_safe(value: Any) -> Any:
#     """Convert common Python metadata values to JSON-safe values."""

#     if value is None or isinstance(
#         value,
#         (str, int, float, bool)
#     ):
#         return value

#     if isinstance(value, Path):
#         return str(value)

#     if isinstance(value, dict):
#         return {
#             str(key): make_json_safe(item)
#             for key, item in value.items()
#         }

#     if isinstance(value, (list, tuple, set)):
#         return [
#             make_json_safe(item)
#             for item in value
#         ]

#     if hasattr(value, "isoformat"):
#         return value.isoformat()

#     return str(value)


# # --------------------------------------------------
# # CHUNK SERIALIZATION
# # --------------------------------------------------

# def serialize_chunks(
#     chunks: list[Document],
#     output_path: Path
# ) -> None:
#     """Save LangChain Documents as JSON Lines."""

#     with output_path.open(
#         "w",
#         encoding="utf-8"
#     ) as file:

#         for chunk in chunks:

#             record = {
#                 "page_content": chunk.page_content,
#                 "metadata": make_json_safe(chunk.metadata),
#                 "id": getattr(chunk, "id", None),
#             }

#             file.write(
#                 json.dumps(
#                     record,
#                     ensure_ascii=False
#                 ) + "\n"
#             )


# def load_chunks(input_path: Path) -> list[Document]:
#     """Load JSONL records into LangChain Documents."""

#     chunks = []

#     if not input_path.exists():
#         raise FileNotFoundError(
#             f"Chunk file not found: {input_path}"
#         )

#     with input_path.open(
#         "r",
#         encoding="utf-8"
#     ) as file:

#         for line_number, line in enumerate(file, start=1):

#             if not line.strip():
#                 continue

#             try:
#                 record = json.loads(line)

#                 chunk = Document(
#                     page_content=record["page_content"],
#                     metadata=record.get("metadata", {}),
#                 )

#                 if record.get("id") is not None:
#                     chunk.id = record["id"]

#                 chunks.append(chunk)

#             except Exception as exc:
#                 raise ValueError(
#                     f"Invalid chunk record at line "
#                     f"{line_number}: {exc}"
#                 ) from exc

#     return chunks


# # --------------------------------------------------
# # MANIFEST CREATION
# # --------------------------------------------------

# def create_manifest(
#     snapshot_id: str,
#     snapshot_dir: Path,
#     chunks: list[Document],
#     ingestion_report: dict | None = None,
# ) -> dict:
#     """
#     Create a manifest containing snapshot metadata
#     and checksums for all required snapshot files.
#     """

#     checksums = {}

#     for filename in REQUIRED_SNAPSHOT_FILES:

#         file_path = snapshot_dir / filename

#         if not file_path.exists():
#             raise FileNotFoundError(
#                 f"Required snapshot file missing: {filename}"
#             )

#         checksums[filename] = calculate_file_hash(
#             file_path
#         )

#     document_ids = {
#         str(chunk.metadata.get("document_id"))
#         for chunk in chunks
#         if chunk.metadata.get("document_id") is not None
#     }

#     manifest = {
#         "schema_version": SNAPSHOT_SCHEMA_VERSION,
#         "snapshot_id": snapshot_id,
#         "created_at": datetime.now(
#             timezone.utc
#         ).isoformat(),

#         "embedding_model": EMBEDDING_MODEL,

#         "chunk_count": len(chunks),
#         "document_count": len(document_ids),

#         "ingestion_report": make_json_safe(
#             ingestion_report or {}
#         ),

#         "checksums": checksums,
#     }

#     return manifest


# # --------------------------------------------------
# # SNAPSHOT VALIDATION
# # --------------------------------------------------

# def validate_snapshot(
#     snapshot_dir: Path,
#     expected_snapshot_id: str | None = None,
#     check_directory_name: bool = True,
# ) -> dict:
#     """
#     Validate snapshot structure, manifest, checksums,
#     and stored chunk records.

#     check_directory_name:
#         True  -> enforce permanent directory name matching.
#         False -> skip only that check for temporary directories.

#     Returns:
#         {
#             "valid": bool,
#             "errors": list[str]
#         }
#     """

#     snapshot_dir = Path(snapshot_dir)

#     errors = []

#     if not snapshot_dir.exists():
#         return {
#             "valid": False,
#             "errors": ["Snapshot directory does not exist"]
#         }

#     if not snapshot_dir.is_dir():
#         return {
#             "valid": False,
#             "errors": ["Snapshot path is not a directory"]
#         }

#     manifest_path = snapshot_dir / MANIFEST_FILE

#     if not manifest_path.exists():
#         return {
#             "valid": False,
#             "errors": ["Manifest file is missing"]
#         }

#     try:
#         with manifest_path.open(
#             "r",
#             encoding="utf-8"
#         ) as file:
#             manifest = json.load(file)

#     except Exception as exc:
#         return {
#             "valid": False,
#             "errors": [f"Unable to read manifest: {exc}"]
#         }

#     if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
#         errors.append("Unsupported snapshot schema version")

#     snapshot_id = manifest.get("snapshot_id")

#     if not snapshot_id:
#         errors.append("Snapshot ID missing from manifest")

#     if (
#         expected_snapshot_id
#         and snapshot_id != expected_snapshot_id
#     ):
#         errors.append("Snapshot ID does not match expected ID")

#     # IMPORTANT:
#     # Temporary directories have names such as .building_xxxxx.
#     # Their names cannot match the permanent snapshot ID.
#     # Enforce this check only for permanent snapshots.
#     if (
#         check_directory_name
#         and snapshot_dir.name != snapshot_id
#     ):
#         errors.append(
#             "Snapshot directory name does not match manifest ID"
#         )

#     checksums = manifest.get("checksums", {})

#     for filename in REQUIRED_SNAPSHOT_FILES:

#         file_path = snapshot_dir / filename

#         if not file_path.exists():
#             errors.append(f"Missing required file: {filename}")
#             continue

#         expected_hash = checksums.get(filename)

#         if not expected_hash:
#             errors.append(f"Checksum missing for {filename}")
#             continue

#         try:
#             actual_hash = calculate_file_hash(file_path)

#             if actual_hash != expected_hash:
#                 errors.append(
#                     f"Checksum mismatch for {filename}"
#                 )

#         except Exception as exc:
#             errors.append(
#                 f"Unable to calculate checksum for "
#                 f"{filename}: {exc}"
#             )

#     chunks_path = snapshot_dir / CHUNKS_FILE

#     if chunks_path.exists():

#         try:
#             chunks = load_chunks(chunks_path)

#             if len(chunks) != manifest.get("chunk_count"):
#                 errors.append(
#                     "Manifest chunk count does not match stored chunks"
#                 )

#             chunk_ids = [
#                 chunk.metadata.get("chunk_id")
#                 for chunk in chunks
#                 if chunk.metadata.get("chunk_id") is not None
#             ]

#             if len(chunk_ids) != len(set(chunk_ids)):
#                 errors.append("Duplicate chunk IDs detected")

#         except Exception as exc:
#             errors.append(
#                 f"Chunk validation failed: {exc}"
#             )

#     return {
#         "valid": len(errors) == 0,
#         "errors": errors,
#     }


# # --------------------------------------------------
# # BUILD SNAPSHOT
# # --------------------------------------------------

# def build_snapshot(
#     chunks: list[Document],
#     ingestion_report: dict | None = None,
#     embeddings=None,
# ) -> str:
#     """
#     Build a new FAISS snapshot without publishing it.

#     Steps:
#         1. Create temporary snapshot directory.
#         2. Build FAISS index.
#         3. Save chunks.
#         4. Generate manifest.
#         5. Validate snapshot.
#         6. Atomically rename temporary directory.

#     Returns:
#         snapshot_id
#     """

#     initialize_directories()

#     if not chunks:
#         raise ValueError(
#             "Cannot build snapshot: no chunks provided"
#         )

#     for index, chunk in enumerate(chunks):

#         if not isinstance(chunk, Document):
#             raise TypeError(
#                 f"Expected LangChain Document at index {index}"
#             )

#         if not chunk.page_content.strip():
#             raise ValueError(
#                 f"Empty chunk detected at index {index}"
#             )

#     embeddings = embeddings or get_embeddings()

#     timestamp = datetime.now(
#         timezone.utc
#     ).strftime("%Y%m%dT%H%M%S")

#     snapshot_id = (
#         f"snapshot_{timestamp}_{uuid.uuid4().hex[:8]}"
#     )

#     final_dir = snapshot_path(snapshot_id)

#     temporary_dir = Path(
#         tempfile.mkdtemp(
#             prefix=".building_",
#             dir=str(SNAPSHOT_DIR)
#         )
#     )

#     try:

#         logger.info(
#             "Building snapshot %s with %d chunks",
#             snapshot_id,
#             len(chunks)
#         )

#         # Build FAISS vector index.
#         vectorstore = FAISS.from_documents(
#             documents=chunks,
#             embedding=embeddings
#         )

#         # Save FAISS files.
#         vectorstore.save_local(
#             str(temporary_dir)
#         )

#         # Save source chunks.
#         serialize_chunks(
#             chunks,
#             temporary_dir / CHUNKS_FILE
#         )

#         # Create manifest.
#         manifest = create_manifest(
#             snapshot_id=snapshot_id,
#             snapshot_dir=temporary_dir,
#             chunks=chunks,
#             ingestion_report=ingestion_report,
#         )

#         manifest_path = temporary_dir / MANIFEST_FILE

#         with manifest_path.open(
#             "w",
#             encoding="utf-8"
#         ) as file:

#             json.dump(
#                 manifest,
#                 file,
#                 indent=2,
#                 ensure_ascii=False
#             )

#             file.flush()
#             os.fsync(file.fileno())

#         # Validate contents and manifest ID.
#         # Skip only the permanent-directory-name check here.
#         validation = validate_snapshot(
#             temporary_dir,
#             expected_snapshot_id=snapshot_id,
#             check_directory_name=False,
#         )

#         if not validation["valid"]:
#             raise ValueError(
#                 f"New snapshot validation failed: "
#                 f"{validation['errors']}"
#             )

#         # Avoid accidentally replacing an existing snapshot.
#         if final_dir.exists():
#             raise FileExistsError(
#                 f"Snapshot destination already exists: {final_dir}"
#             )

#         # Atomic directory rename within SNAPSHOT_DIR.
#         os.replace(
#             temporary_dir,
#             final_dir
#         )

#         # Validate again after the rename, now enforcing
#         # the permanent directory name.
#         final_validation = validate_snapshot(
#             final_dir,
#             expected_snapshot_id=snapshot_id,
#             check_directory_name=True,
#         )

#         if not final_validation["valid"]:
#             raise ValueError(
#                 f"Final snapshot validation failed: "
#                 f"{final_validation['errors']}"
#             )

#         logger.info(
#             "Snapshot successfully built: %s",
#             snapshot_id
#         )

#         return snapshot_id

#     except Exception:

#         if temporary_dir.exists():
#             shutil.rmtree(
#                 temporary_dir,
#                 ignore_errors=True
#             )

#         logger.exception(
#             "Snapshot build failed: %s",
#             snapshot_id
#         )

#         raise


# # --------------------------------------------------
# # LOAD SNAPSHOT BY ID
# # --------------------------------------------------

# def load_snapshot(
#     snapshot_id: str,
#     embeddings=None,
# ) -> dict[str, Any]:
#     """
#     Load and validate a specific snapshot by ID.

#     This allows a newly built snapshot to be tested
#     before changing the active snapshot pointer.
#     """

#     initialize_directories()

#     snapshot_dir = snapshot_path(snapshot_id)

#     validation = validate_snapshot(
#         snapshot_dir,
#         expected_snapshot_id=snapshot_id,
#         check_directory_name=True,
#     )

#     if not validation["valid"]:
#         raise ValueError(
#             f"Snapshot is invalid: {validation['errors']}"
#         )

#     embeddings = embeddings or get_embeddings()

#     # Only deserialize snapshots generated and controlled
#     # by this application.
#     vectorstore = FAISS.load_local(
#         str(snapshot_dir),
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

#     chunks = load_chunks(
#         snapshot_dir / CHUNKS_FILE
#     )

#     if vectorstore.index.ntotal != len(chunks):
#         raise ValueError(
#             "FAISS vector count does not match stored chunk count"
#         )

#     with (snapshot_dir / MANIFEST_FILE).open(
#         "r",
#         encoding="utf-8"
#     ) as file:
#         manifest = json.load(file)

#     logger.info(
#         "Loaded snapshot %s with %d chunks",
#         snapshot_id,
#         len(chunks)
#     )

#     return {
#         "snapshot_id": snapshot_id,
#         "vectorstore": vectorstore,
#         "chunks": chunks,
#         "manifest": manifest,
#     }


# # --------------------------------------------------
# # PUBLISH SNAPSHOT
# # --------------------------------------------------

# def publish_snapshot(snapshot_id: str) -> str:
#     """
#     Atomically publish a validated snapshot.

#     The current active snapshot is not changed
#     if validation fails.
#     """

#     initialize_directories()

#     snapshot_dir = snapshot_path(snapshot_id)

#     validation = validate_snapshot(
#         snapshot_dir,
#         expected_snapshot_id=snapshot_id,
#         check_directory_name=True,
#     )

#     if not validation["valid"]:
#         raise ValueError(
#             f"Cannot publish invalid snapshot: {validation['errors']}"
#         )

#     temporary_pointer = None

#     try:

#         file_descriptor, temporary_pointer = tempfile.mkstemp(
#             prefix=".current_snapshot_",
#             dir=str(CURRENT_SNAPSHOT_FILE.parent),
#             text=True
#         )

#         with os.fdopen(
#             file_descriptor,
#             "w",
#             encoding="utf-8"
#         ) as file:

#             file.write(snapshot_id + "\n")
#             file.flush()
#             os.fsync(file.fileno())

#         # Atomic pointer replacement.
#         os.replace(
#             temporary_pointer,
#             CURRENT_SNAPSHOT_FILE
#         )

#         logger.info(
#             "Published snapshot: %s",
#             snapshot_id
#         )

#         return snapshot_id

#     except Exception:

#         if (
#             temporary_pointer
#             and os.path.exists(temporary_pointer)
#         ):
#             os.remove(temporary_pointer)

#         logger.exception(
#             "Failed to publish snapshot: %s",
#             snapshot_id
#         )

#         raise


# # --------------------------------------------------
# # LOAD ACTIVE SNAPSHOT
# # --------------------------------------------------

# def load_active_snapshot(
#     embeddings=None,
# ) -> dict[str, Any]:
#     """Load the currently published snapshot."""

#     snapshot_id = get_active_snapshot_id()

#     if not snapshot_id:
#         raise FileNotFoundError(
#             "No active snapshot found. Run ingestion first."
#         )

#     return load_snapshot(
#         snapshot_id=snapshot_id,
#         embeddings=embeddings
#     )


# # --------------------------------------------------
# # ACTIVE SNAPSHOT INFORMATION
# # --------------------------------------------------

# def get_active_snapshot_id() -> str | None:
#     """Return the currently published snapshot ID."""

#     initialize_directories()

#     if not CURRENT_SNAPSHOT_FILE.exists():
#         return None

#     snapshot_id = CURRENT_SNAPSHOT_FILE.read_text(
#         encoding="utf-8"
#     ).strip()

#     return snapshot_id or None

import hashlib
import json
import logging
import os
import shutil
import tempfile
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    INDEX_DIR,
    SNAPSHOT_DIR,
    CURRENT_SNAPSHOT_FILE,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)

SNAPSHOT_SCHEMA_VERSION = 1

# Legacy FAISS files are still validated when loading older snapshots.
FAISS_INDEX_FILE = "index.faiss"
FAISS_METADATA_FILE = "index.pkl"
CHUNKS_FILE = "chunks.jsonl"
VECTORS_FILE = "vectors.npy"
MANIFEST_FILE = "manifest.json"

LEGACY_REQUIRED_FILES = [
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    CHUNKS_FILE,
]

_embeddings_instance = None


# --------------------------------------------------
# DIRECTORY INITIALIZATION
# --------------------------------------------------

def initialize_directories() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    CURRENT_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# HASHING
# --------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def calculate_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------
# EMBEDDINGS
# --------------------------------------------------

def get_embeddings():
    """Return a lazily initialized embedding model."""
    global _embeddings_instance

    if _embeddings_instance is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings_instance


# --------------------------------------------------
# SNAPSHOT PATH VALIDATION
# --------------------------------------------------

def snapshot_path(snapshot_id: str) -> Path:
    if not snapshot_id:
        raise ValueError("Snapshot ID cannot be empty")

    if (
        "/" in snapshot_id
        or "\\" in snapshot_id
        or snapshot_id in {".", ".."}
    ):
        raise ValueError("Invalid snapshot ID")

    base_dir = SNAPSHOT_DIR.resolve()
    target = (base_dir / snapshot_id).resolve()

    if target.parent != base_dir:
        raise ValueError("Invalid snapshot directory")

    return target


# --------------------------------------------------
# JSON SERIALIZATION
# --------------------------------------------------

def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)


# --------------------------------------------------
# CHUNK SERIALIZATION
# --------------------------------------------------

def serialize_chunks(
    chunks: list[Document],
    output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": make_json_safe(chunk.metadata),
                "id": getattr(chunk, "id", None),
            }
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def load_chunks(input_path: Path) -> list[Document]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    chunks = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                chunk = Document(
                    page_content=record["page_content"],
                    metadata=record.get("metadata", {}),
                )
                if record.get("id") is not None:
                    chunk.id = record["id"]
                chunks.append(chunk)
            except Exception as exc:
                raise ValueError(
                    f"Invalid chunk record at line {line_number}: {exc}"
                ) from exc

    return chunks


# --------------------------------------------------
# MANIFEST CREATION
# --------------------------------------------------

def create_manifest(
    snapshot_id: str,
    snapshot_dir: Path,
    chunks: list[Document],
    ingestion_report: dict | None = None,
) -> dict:
    """
    Create checksums for the files actually stored in this snapshot.

    New NumPy snapshots contain chunks.jsonl and vectors.npy.
    Legacy snapshots retain their FAISS files and old checksum format.
    """
    candidate_files = [
        CHUNKS_FILE,
        VECTORS_FILE,
        FAISS_INDEX_FILE,
        FAISS_METADATA_FILE,
    ]

    checksums = {}
    for filename in candidate_files:
        file_path = snapshot_dir / filename
        if file_path.is_file():
            checksums[filename] = calculate_file_hash(file_path)

    if CHUNKS_FILE not in checksums:
        raise FileNotFoundError("Required snapshot file missing: chunks.jsonl")

    if VECTORS_FILE not in checksums:
        if not (
            FAISS_INDEX_FILE in checksums
            and FAISS_METADATA_FILE in checksums
        ):
            raise FileNotFoundError(
                "Snapshot must contain vectors.npy or both legacy FAISS files"
            )

    document_ids = {
        str(chunk.metadata.get("document_id"))
        for chunk in chunks
        if chunk.metadata.get("document_id") is not None
    }

    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "document_count": len(document_ids),
        "storage_format": (
            "numpy" if VECTORS_FILE in checksums else "legacy_faiss"
        ),
        "vector_file": (
            VECTORS_FILE if VECTORS_FILE in checksums else None
        ),
        "ingestion_report": make_json_safe(ingestion_report or {}),
        "checksums": checksums,
    }


# --------------------------------------------------
# SNAPSHOT VALIDATION
# --------------------------------------------------

def validate_snapshot(
    snapshot_dir: Path,
    expected_snapshot_id: str | None = None,
    check_directory_name: bool = True,
) -> dict:
    snapshot_dir = Path(snapshot_dir)
    errors = []

    if not snapshot_dir.exists():
        return {
            "valid": False,
            "errors": ["Snapshot directory does not exist"],
        }

    if not snapshot_dir.is_dir():
        return {
            "valid": False,
            "errors": ["Snapshot path is not a directory"],
        }

    manifest_path = snapshot_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        return {
            "valid": False,
            "errors": ["Manifest file is missing"],
        }

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except Exception as exc:
        return {
            "valid": False,
            "errors": [f"Unable to read manifest: {exc}"],
        }

    if manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        errors.append("Unsupported snapshot schema version")

    snapshot_id = manifest.get("snapshot_id")
    if not snapshot_id:
        errors.append("Snapshot ID missing from manifest")

    if expected_snapshot_id and snapshot_id != expected_snapshot_id:
        errors.append("Snapshot ID does not match expected ID")

    if check_directory_name and snapshot_dir.name != snapshot_id:
        errors.append("Snapshot directory name does not match manifest ID")

    checksums = manifest.get("checksums", {})
    if not isinstance(checksums, dict):
        errors.append("Manifest checksums must be an object")
        checksums = {}

    # Required chunk file plus one supported vector storage format.
    if CHUNKS_FILE not in checksums:
        errors.append("Checksum missing for chunks.jsonl")

    has_numpy = VECTORS_FILE in checksums
    has_legacy = (
        FAISS_INDEX_FILE in checksums
        and FAISS_METADATA_FILE in checksums
    )

    if not has_numpy and not has_legacy:
        errors.append(
            "Snapshot must contain vectors.npy or both legacy FAISS files"
        )

    for filename, expected_hash in checksums.items():
        file_path = snapshot_dir / filename

        # Do not allow manifest paths to escape the snapshot directory.
        if Path(filename).name != filename:
            errors.append(f"Invalid manifest filename: {filename}")
            continue

        if not file_path.is_file():
            errors.append(f"Missing required file: {filename}")
            continue

        if not expected_hash:
            errors.append(f"Checksum missing for {filename}")
            continue

        try:
            actual_hash = calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                errors.append(f"Checksum mismatch for {filename}")
        except Exception as exc:
            errors.append(
                f"Unable to calculate checksum for {filename}: {exc}"
            )

    chunks_path = snapshot_dir / CHUNKS_FILE
    if chunks_path.is_file():
        try:
            chunks = load_chunks(chunks_path)

            if len(chunks) != manifest.get("chunk_count"):
                errors.append(
                    "Manifest chunk count does not match stored chunks"
                )

            chunk_ids = [
                chunk.metadata.get("chunk_id")
                for chunk in chunks
                if chunk.metadata.get("chunk_id") is not None
            ]
            if len(chunk_ids) != len(set(chunk_ids)):
                errors.append("Duplicate chunk IDs detected")
        except Exception as exc:
            errors.append(f"Chunk validation failed: {exc}")

    if has_numpy:
        vectors_path = snapshot_dir / VECTORS_FILE
        if vectors_path.is_file():
            try:
                vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
                if vectors.ndim != 2:
                    errors.append("NumPy vector matrix must be two-dimensional")
                if vectors.shape[0] != manifest.get("chunk_count"):
                    errors.append(
                        "Manifest chunk count does not match NumPy vector count"
                    )
                if vectors.dtype != np.float32:
                    errors.append("NumPy vector matrix must use float32")
                if not np.isfinite(vectors).all():
                    errors.append("NumPy vector matrix contains invalid values")
            except Exception as exc:
                errors.append(f"NumPy vector validation failed: {exc}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# --------------------------------------------------
# BUILD SNAPSHOT - NUMPY ONLY
# --------------------------------------------------

def build_snapshot(
    chunks: list[Document],
    ingestion_report: dict | None = None,
    embeddings=None,
) -> str:
    """
    Build a new snapshot using NumPy vectors.

    This function intentionally does not import or invoke FAISS.
    """
    initialize_directories()

    if not chunks:
        raise ValueError("Cannot build snapshot: no chunks provided")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Document):
            raise TypeError(
                f"Expected LangChain Document at index {index}"
            )
        if not chunk.page_content.strip():
            raise ValueError(f"Empty chunk detected at index {index}")

    embeddings = embeddings or get_embeddings()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    snapshot_id = f"snapshot_{timestamp}_{uuid.uuid4().hex[:8]}"

    final_dir = snapshot_path(snapshot_id)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".building_", dir=str(SNAPSHOT_DIR))
    )

    try:
        logger.info(
            "Building NumPy snapshot %s with %d chunks",
            snapshot_id,
            len(chunks),
        )

        texts = [chunk.page_content for chunk in chunks]
        vectors = np.asarray(
            embeddings.embed_documents(texts),
            dtype=np.float32,
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                f"Invalid embedding matrix shape: {vectors.shape}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Generated embeddings contain invalid values")

        np.save(temporary_dir / VECTORS_FILE, vectors, allow_pickle=False)
        serialize_chunks(chunks, temporary_dir / CHUNKS_FILE)

        manifest = create_manifest(
            snapshot_id=snapshot_id,
            snapshot_dir=temporary_dir,
            chunks=chunks,
            ingestion_report=ingestion_report,
        )

        manifest_path = temporary_dir / MANIFEST_FILE
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.flush()
            os.fsync(file.fileno())

        validation = validate_snapshot(
            temporary_dir,
            expected_snapshot_id=snapshot_id,
            check_directory_name=False,
        )
        if not validation["valid"]:
            raise ValueError(
                f"New snapshot validation failed: {validation['errors']}"
            )

        if final_dir.exists():
            raise FileExistsError(
                f"Snapshot destination already exists: {final_dir}"
            )

        os.replace(temporary_dir, final_dir)

        final_validation = validate_snapshot(
            final_dir,
            expected_snapshot_id=snapshot_id,
            check_directory_name=True,
        )
        if not final_validation["valid"]:
            raise ValueError(
                f"Final snapshot validation failed: {final_validation['errors']}"
            )

        logger.info("Snapshot successfully built: %s", snapshot_id)
        return snapshot_id

    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)

        logger.exception("Snapshot build failed: %s", snapshot_id)
        raise


# --------------------------------------------------
# LOAD SNAPSHOT BY ID
# --------------------------------------------------

def _resolve_vectors_path(snapshot_id: str, snapshot_dir: Path) -> Path:
    """
    Prefer vectors stored inside a new-format snapshot.
    Fall back to the extracted sidecar for the existing legacy snapshot.
    """
    in_snapshot = snapshot_dir / VECTORS_FILE
    if in_snapshot.is_file():
        return in_snapshot

    sidecar = (
        INDEX_DIR
        / "numpy_vectors"
        / f"{snapshot_id}.npy"
    )
    if sidecar.is_file():
        return sidecar

    raise FileNotFoundError(
        f"No NumPy vectors found for snapshot {snapshot_id}. "
        f"Expected {in_snapshot} or {sidecar}"
    )


def load_snapshot(
    snapshot_id: str,
    embeddings=None,
) -> dict[str, Any]:
    initialize_directories()
    snapshot_dir = snapshot_path(snapshot_id)

    validation = validate_snapshot(
        snapshot_dir,
        expected_snapshot_id=snapshot_id,
        check_directory_name=True,
    )
    if not validation["valid"]:
        raise ValueError(f"Snapshot is invalid: {validation['errors']}")

    chunks = load_chunks(snapshot_dir / CHUNKS_FILE)
    vectors_path = _resolve_vectors_path(snapshot_id, snapshot_dir)

    vectors = np.load(vectors_path, allow_pickle=False)
    if vectors.ndim != 2:
        raise ValueError(
            f"Expected a 2D vector matrix, got shape {vectors.shape}"
        )

    if vectors.shape[0] != len(chunks):
        raise ValueError(
            "NumPy vector count does not match stored chunk count"
        )

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    if not np.isfinite(vectors).all():
        raise ValueError("NumPy vector matrix contains invalid values")

    embeddings = embeddings or get_embeddings()

    with (snapshot_dir / MANIFEST_FILE).open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    logger.info(
        "Loaded snapshot %s with %d chunks and vector shape %s",
        snapshot_id,
        len(chunks),
        vectors.shape,
    )

    return {
        "snapshot_id": snapshot_id,
        "vectors": vectors,
        "embeddings": embeddings,
        "chunks": chunks,
        "manifest": manifest,
    }


# --------------------------------------------------
# PUBLISH SNAPSHOT
# --------------------------------------------------

def publish_snapshot(snapshot_id: str) -> str:
    initialize_directories()
    snapshot_dir = snapshot_path(snapshot_id)

    validation = validate_snapshot(
        snapshot_dir,
        expected_snapshot_id=snapshot_id,
        check_directory_name=True,
    )
    if not validation["valid"]:
        raise ValueError(
            f"Cannot publish invalid snapshot: {validation['errors']}"
        )

    temporary_pointer = None

    try:
        file_descriptor, temporary_pointer = tempfile.mkstemp(
            prefix=".current_snapshot_",
            dir=str(CURRENT_SNAPSHOT_FILE.parent),
            text=True,
        )

        with os.fdopen(
            file_descriptor,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(snapshot_id + "\n")
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_pointer, CURRENT_SNAPSHOT_FILE)

        logger.info("Published snapshot: %s", snapshot_id)
        return snapshot_id

    except Exception:
        if temporary_pointer and os.path.exists(temporary_pointer):
            os.remove(temporary_pointer)

        logger.exception("Failed to publish snapshot: %s", snapshot_id)
        raise


# --------------------------------------------------
# LOAD ACTIVE SNAPSHOT
# --------------------------------------------------

def load_active_snapshot(embeddings=None) -> dict[str, Any]:
    snapshot_id = get_active_snapshot_id()

    if not snapshot_id:
        raise FileNotFoundError(
            "No active snapshot found. Run ingestion first."
        )

    return load_snapshot(
        snapshot_id=snapshot_id,
        embeddings=embeddings,
    )


# --------------------------------------------------
# ACTIVE SNAPSHOT INFORMATION
# --------------------------------------------------

def get_active_snapshot_id() -> str | None:
    initialize_directories()

    if not CURRENT_SNAPSHOT_FILE.exists():
        return None

    snapshot_id = CURRENT_SNAPSHOT_FILE.read_text(
        encoding="utf-8"
    ).strip()

    return snapshot_id or None
