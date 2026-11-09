import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index_store"
SNAPSHOT_DIR = INDEX_DIR / "snapshots"
CURRENT_SNAPSHOT_FILE = INDEX_DIR / "current_snapshot"

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# Retrieval
DENSE_TOP_K = int(os.getenv("DENSE_TOP_K", "15"))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", "15"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Reranking
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))

# Context
MAX_CONTEXT_TOKENS = int(
    os.getenv("MAX_CONTEXT_TOKENS", "5000")
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)

RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "groq:openai/gpt-oss-120b"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# Olist SQL-RAG service
OLIST_BASE_URL = os.getenv(
    "OLIST_BASE_URL",
    "http://127.0.0.1:8001"
)

OLIST_SALES_TOKEN = os.getenv("OLIST_SALES_TOKEN", "")
OLIST_ANALYST_TOKEN = os.getenv("OLIST_ANALYST_TOKEN", "")

OLIST_TIMEOUT_SECONDS = float(
    os.getenv("OLIST_TIMEOUT_SECONDS", "20")
)