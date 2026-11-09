# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Dict, List
# import os

# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware

# from langchain_community.document_loaders import (
#     PyMuPDFLoader,
#     Docx2txtLoader,
#     TextLoader
# )

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever

# from langchain_core.runnables import RunnableLambda
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# from sentence_transformers import CrossEncoder

# from langchain.chat_models import init_chat_model


# # ============================================================
# # ENVIRONMENT
# # ============================================================

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY is not set.")

# os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# # ============================================================
# # FASTAPI
# # ============================================================

# app = FastAPI(
#     title="RAG API",
#     version="1.0.0"
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ============================================================
# # REQUEST MODEL
# # ============================================================

# class Query(BaseModel):
#     question: str
#     session_id: str


# # ============================================================
# # CONVERSATION MEMORY
# # ============================================================

# conversation_memory: Dict[str, List[dict]] = {}


# def get_memory(session_id: str) -> List[dict]:

#     return conversation_memory.get(
#         session_id,
#         []
#     )


# def save_memory(
#     session_id: str,
#     question: str,
#     answer: str
# ):

#     if session_id not in conversation_memory:
#         conversation_memory[session_id] = []

#     conversation_memory[session_id].append(
#         {
#             "question": question,
#             "answer": answer
#         }
#     )

#     # Keep the latest 10 conversation turns
#     conversation_memory[session_id] = (
#         conversation_memory[session_id][-10:]
#     )


# def format_history(
#     history: List[dict]
# ) -> str:

#     if not history:
#         return "No previous conversation."

#     return "\n\n".join(
#         f"User: {item['question']}\n"
#         f"Assistant: {item['answer']}"
#         for item in history
#     )


# # ============================================================
# # LOAD DOCUMENTS
# # ============================================================

# DATA_PATH = "/Users/saianish/Desktop/3-2/NLP/RAG/Project/backend/data"


# def load_documents():

#     all_docs = []

#     if not os.path.exists(DATA_PATH):

#         raise FileNotFoundError(
#             f"Data directory '{DATA_PATH}' does not exist."
#         )

#     for file in os.listdir(DATA_PATH):

#         file_path = os.path.join(
#             DATA_PATH,
#             file
#         )

#         try:

#             if file.lower().endswith(".pdf"):

#                 loader = PyMuPDFLoader(
#                     file_path
#                 )

#             elif file.lower().endswith(".docx"):

#                 loader = Docx2txtLoader(
#                     file_path
#                 )

#             elif file.lower().endswith(".txt"):

#                 loader = TextLoader(
#                     file_path,
#                     encoding="utf-8"
#                 )

#             else:

#                 continue

#             docs = loader.load()

#             for doc in docs:

#                 doc.metadata["file_name"] = file

#                 doc.metadata["file_type"] = (
#                     file.split(".")[-1].lower()
#                 )

#             all_docs.extend(docs)

#         except Exception as e:

#             print(
#                 f"Error loading {file}: {e}"
#             )

#     if not all_docs:

#         raise ValueError(
#             "No supported documents found in data directory."
#         )

#     return all_docs


# # ============================================================
# # INITIALIZE RAG
# # ============================================================

# def initialize_rag():

#     docs = load_documents()


#     # ========================================================
#     # TEXT SPLITTING
#     # ========================================================

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100
#     )

#     chunks = splitter.split_documents(
#         docs
#     )


#     # ========================================================
#     # EMBEDDINGS
#     # ========================================================

#     embedding_model = HuggingFaceEmbeddings(
#         model_name="all-MiniLM-L6-v2"
#     )


#     # ========================================================
#     # FAISS DENSE RETRIEVER
#     # ========================================================

#     vector_store = FAISS.from_documents(
#         chunks,
#         embedding_model
#     )

#     dense_retriever = vector_store.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 10,
#             "fetch_k": 20
#         }
#     )


#     # ========================================================
#     # BM25 SPARSE RETRIEVER
#     # ========================================================

#     sparse_retriever = BM25Retriever.from_documents(
#         chunks
#     )

#     sparse_retriever.k = 10


#     # ========================================================
#     # HYBRID RETRIEVAL
#     # ========================================================

#     def hybrid_logic(query):

#         dense_docs = dense_retriever.invoke(
#             query
#         )

#         sparse_docs = sparse_retriever.invoke(
#             query
#         )

#         combined = (
#             dense_docs +
#             sparse_docs
#         )

#         unique_docs = list(
#             {
#                 doc.page_content: doc
#                 for doc in combined
#             }.values()
#         )

#         return unique_docs


#     hybrid_retriever = RunnableLambda(
#         hybrid_logic
#     )


#     # ========================================================
#     # CROSS ENCODER
#     # ========================================================

#     reranker = CrossEncoder(
#         "cross-encoder/ms-marco-MiniLM-L-6-v2"
#     )


#     def rerank_docs(
#         query,
#         docs,
#         top_k=3
#     ):

#         if not docs:

#             return []


#         pairs = [
#             (
#                 query,
#                 doc.page_content
#             )
#             for doc in docs
#         ]


#         scores = reranker.predict(
#             pairs
#         )


#         ranked = sorted(
#             zip(docs, scores),
#             key=lambda x: x[1],
#             reverse=True
#         )


#         # ----------------------------------------------------
#         # INSPECT CROSS-ENCODER SCORES
#         # ----------------------------------------------------

#         print("\n" + "=" * 60)

#         print(
#             f"QUERY: {query}"
#         )

#         print(
#             "CROSS-ENCODER SCORES"
#         )

#         print("=" * 60)


#         for doc, score in ranked:

#             file_name = doc.metadata.get(
#                 "file_name",
#                 "Unknown"
#             )

#             page = doc.metadata.get(
#                 "page",
#                 None
#             )

#             if page is not None:

#                 page_display = page + 1

#             else:

#                 page_display = "N/A"


#             print(
#                 f"Score: {score:.4f} | "
#                 f"File: {file_name} | "
#                 f"Page: {page_display}"
#             )


#         print("=" * 60)


#         # ----------------------------------------------------
#         # ONLY TOP 3 GO TO LLM
#         # ----------------------------------------------------

#         return [
#             doc
#             for doc, _ in ranked[:top_k]
#         ]


#     # ========================================================
#     # FORMAT DOCUMENTS
#     # ========================================================

#     def format_docs(docs):

#         formatted = []

#         for doc in docs:

#             file_name = doc.metadata.get(
#                 "file_name",
#                 "Unknown"
#             )

#             page = doc.metadata.get(
#                 "page",
#                 None
#             )


#             if page is not None:

#                 source = (
#                     f"{file_name}, "
#                     f"Page {page + 1}"
#                 )

#             else:

#                 source = file_name


#             formatted.append(
#                 f"[Source: {source}]\n"
#                 f"{doc.page_content}"
#             )


#         return "\n\n".join(
#             formatted
#         )


#     # ========================================================
#     # LLM
#     # ========================================================

#     llm = init_chat_model(
#         "groq:openai/gpt-oss-120b"
#     )


#     # ========================================================
#     # MEMORY QUESTION REWRITING
#     # ========================================================

#     memory_prompt = PromptTemplate.from_template(
#         """
# You are a question contextualization system.

# Your task is to rewrite the user's latest question
# into a standalone question.

# Use the conversation history only when necessary.

# Resolve references such as:
# - it
# - this
# - that
# - they
# - them
# - those
# - previous question
# - previous answer
# - above
# - mentioned earlier

# Rules:

# 1. Do not answer the question.
# 2. Do not add information that is not present
#    in the conversation.
# 3. If the question is already standalone,
#    return it unchanged.
# 4. Return ONLY the standalone question.

# Conversation History:
# {history}

# New Question:
# {question}

# Standalone Question:
# """

#     )


#     memory_chain = (
#         memory_prompt
#         | llm
#         | StrOutputParser()
#     )


#     # ========================================================
#     # ANSWER PROMPT
#     # ========================================================

#     answer_prompt = PromptTemplate.from_template(
#         """
# You are a precise document question-answering assistant.

# Your task is to answer the user's question using
# ONLY the provided context.

# IMPORTANT RULES:

# 1. Answer the exact question asked.
# 2. Give the shortest complete answer possible.
# 3. Do NOT summarize the entire context.
# 4. Do NOT list unrelated facts from the context.
# 5. Do NOT provide background information unless required.
# 6. If the question asks for a specific fact,
#    provide that fact directly.
# 7. If the question asks for multiple things,
#    answer only those things.
# 8. Do not make assumptions.
# 9. Do not add information that is not supported
#    by the context.
# 10. Treat the context as evidence, not as something
#     that must be summarized completely.
# 11. Do not mention source metadata in the answer.
# 12. If the answer is not present in the context,
#     say exactly:

# "I don't have enough information in the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
#     )


#     document_chain = (
#         answer_prompt
#         | llm
#         | StrOutputParser()
#     )


#     # ========================================================
#     # FINAL RAG PIPELINE
#     # ========================================================

#     def rag_function(input_data):

#         question = input_data["input"]


#         # ----------------------------------------------------
#         # ORIGINAL QUERY
#         # ----------------------------------------------------

#         # Query expansion is intentionally disabled.
#         # The original query is used directly for retrieval.
#         retrieved_docs = (
#             hybrid_retriever.invoke(
#                 question
#             )
#         )


#         # ----------------------------------------------------
#         # CROSS-ENCODER RERANKING
#         # ----------------------------------------------------

#         reranked_docs = rerank_docs(
#             question,
#             retrieved_docs,
#             top_k=3
#         )


#         # ----------------------------------------------------
#         # MINIMAL CONTEXT
#         # ----------------------------------------------------

#         context = format_docs(
#             reranked_docs
#         )


#         # ----------------------------------------------------
#         # GENERATE ANSWER
#         # ----------------------------------------------------

#         answer = document_chain.invoke(
#             {
#                 "context": context,
#                 "question": question
#             }
#         )


#         return {
#             "answer": answer,
#             "documents": reranked_docs
#         }


#     rag_pipeline = RunnableLambda(
#         rag_function
#     )


#     return (
#         rag_pipeline,
#         memory_chain
#     )


# # ============================================================
# # INITIALIZE ONCE
# # ============================================================

# rag_pipeline, memory_chain = initialize_rag()


# # ============================================================
# # CITATIONS
# # ============================================================

# def generate_citations(docs):

#     citations = []

#     seen = set()


#     for doc in docs:

#         file_name = doc.metadata.get(
#             "file_name",
#             "Unknown"
#         )

#         file_type = doc.metadata.get(
#             "file_type",
#             "unknown"
#         )

#         page = doc.metadata.get(
#             "page",
#             None
#         )


#         # -----------------------------------------------
#         # Avoid duplicate source/page citations
#         # -----------------------------------------------

#         key = (
#             file_name,
#             page
#         )

#         if key in seen:

#             continue

#         seen.add(key)


#         if page is not None:

#             page_number = page + 1

#             source = (
#                 f"{file_name}, "
#                 f"Page {page_number}"
#             )

#         else:

#             page_number = None

#             source = file_name


#         citations.append(
#             {
#                 "id": len(citations) + 1,
#                 "source": source,
#                 "file_name": file_name,
#                 "file_type": file_type,
#                 "page": page_number
#             }
#         )


#     return citations


# # ============================================================
# # API ENDPOINT
# # ============================================================

# @app.post("/ask")
# def ask(q: Query):

#     # ========================================================
#     # GET MEMORY
#     # ========================================================

#     history = get_memory(
#         q.session_id
#     )


#     # ========================================================
#     # MEMORY-BASED QUESTION REWRITING
#     # ========================================================

#     if history:

#         standalone_question = (
#             memory_chain.invoke(
#                 {
#                     "history": format_history(
#                         history
#                     ),
#                     "question": q.question
#                 }
#             )
#             .strip()
#         )

#     else:

#         standalone_question = q.question


#     # ========================================================
#     # RUN RAG
#     # ========================================================

#     result = rag_pipeline.invoke(
#         {
#             "input": standalone_question
#         }
#     )


#     answer = result["answer"]

#     retrieved_docs = result["documents"]


#     # ========================================================
#     # GENERATE CITATIONS
#     # ========================================================

#     citations = generate_citations(
#         retrieved_docs
#     )


#     # ========================================================
#     # SAVE CONVERSATION
#     # ========================================================

#     save_memory(
#         q.session_id,
#         q.question,
#         answer
#     )


#     # ========================================================
#     # RESPONSE
#     # ========================================================

#     return {
#         "answer": answer,
#         "citations": citations
#     }




import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Literal

from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



from config import (
    SNAPSHOT_DIR,
    OLIST_BASE_URL,
    OLIST_SALES_TOKEN,
    OLIST_ANALYST_TOKEN,
    OLIST_TIMEOUT_SECONDS,
)

from olist_client import OlistClient, OlistClientError
from llm_factory import create_llm

from ingestion import ingest_documents
from snapshot_manager import (
    build_snapshot,
    publish_snapshot,
    load_snapshot,
    load_active_snapshot,
    get_active_snapshot_id
)
from retrieval import (
    build_retrievers,
    retrieve_candidates
)

from reranker import rerank_and_select

from citation_service import (
    prepare_citations,
    finalize_citations
)


# ============================================================
# ENVIRONMENT AND LOGGING
# ============================================================



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ============================================================
# CONVERSATION MEMORY
# ============================================================

conversation_memory: Dict[str, List[dict]] = {}


def get_memory(session_id: str) -> List[dict]:
    return conversation_memory.get(session_id, [])


def save_memory(
    session_id: str,
    question: str,
    answer: str
):
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []

    conversation_memory[session_id].append({
        "question": question,
        "answer": answer
    })

    conversation_memory[session_id] = (
        conversation_memory[session_id][-10:]
    )


def format_history(history: List[dict]) -> str:

    if not history:
        return "No previous conversation."

    return "\n\n".join(
        f"User: {item['question']}\n"
        f"Assistant: {item['answer']}"
        for item in history
    )


# ============================================================
# PROMPTS
# ============================================================

MEMORY_PROMPT = PromptTemplate.from_template(
    """
You are a question contextualization system.

Rewrite the user's latest question into a standalone question.

Use conversation history only when necessary.

Resolve references such as:
- it
- this
- that
- they
- them
- those
- previous question
- previous answer
- above
- mentioned earlier

Rules:

1. Do not answer the question.
2. Do not add information absent from the conversation.
3. If the question is already standalone, return it unchanged.
4. Return ONLY the standalone question.

Conversation History:
{history}

New Question:
{question}

Standalone Question:
"""
)


ANSWER_PROMPT = PromptTemplate.from_template(
    """
You are a precise document question-answering assistant.

Answer the user's question using ONLY the provided evidence.

IMPORTANT RULES:

1. Answer the exact question asked.
2. Give the shortest complete answer possible.
3. Do not summarize the entire context.
4. Do not list unrelated facts.
5. Do not make assumptions.
6. Do not add information unsupported by the evidence.
7. Treat the evidence as the source of truth.
8. Cite every factual claim using the exact evidence ID.
9. Use citations in this format: [EVD-XXXXXXXXXXXX].
10. Only use evidence IDs supplied in the context.
11. Do not invent evidence IDs.
12. If the answer is not supported, say exactly:

"I don't have enough information in the provided documents."

Evidence:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# REQUEST MODELS
# ============================================================

class Query(BaseModel):
    question: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    mode: Literal["document", "database", "both"] = "document"


class IngestRequest(BaseModel):
    confirm: bool = True


# ============================================================
# APPLICATION STATE
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting RAG API")

    app.state.snapshot = None
    app.state.retrievers = None
    app.state.olist_client = None

    app.state.llm = create_llm()

    if get_active_snapshot_id():

        logger.info("Loading active snapshot")

        try:
            snapshot = load_active_snapshot()

            app.state.snapshot = snapshot
            app.state.retrievers = build_retrievers(snapshot)

            logger.info(
                "Loaded snapshot %s",
                snapshot["snapshot_id"]
            )

        except Exception:
            logger.exception(
                "Could not load active snapshot"
            )

    else:
        logger.warning(
            "No active snapshot. Call POST /ingest first."
        )

    yield

    if app.state.olist_client is not None:
        app.state.olist_client.close()
        app.state.olist_client = None

    logger.info("RAG API shutting down")


app = FastAPI(
    title="RAG API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

def get_olist_client() -> OlistClient:
    """
    Create the Olist HTTP client only when a database
    request actually needs it.
    """

    if app.state.olist_client is not None:
        return app.state.olist_client

    if not OLIST_SALES_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Olist sales credentials are not configured."
        )

    app.state.olist_client = OlistClient(
        base_url=OLIST_BASE_URL,
        sales_token=OLIST_SALES_TOKEN,
        analyst_token=OLIST_ANALYST_TOKEN or None,
        timeout_seconds=OLIST_TIMEOUT_SECONDS,
    )

    return app.state.olist_client

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "snapshot_id": (
            app.state.snapshot["snapshot_id"]
            if app.state.snapshot else None
        ),
        "ready": app.state.retrievers is not None
    }


# ============================================================
# INGESTION API
# ============================================================

@app.post("/ingest")
def ingest():

    try:
        logger.info("Starting document ingestion")

        # 1. Parse documents and generate chunks.
        ingestion_result = ingest_documents()

        chunks = ingestion_result["chunks"]
        report = ingestion_result["report"]

        # Do not publish a partial knowledge base.
        if report["failed"]:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "Ingestion failed for one or more files. "
                        "No snapshot was published."
                    ),
                    "report": report
                }
            )

        # 2. Build a new snapshot without changing
        # the currently active snapshot.
        snapshot_id = build_snapshot(
            chunks=chunks,
            ingestion_report=report
        )

        # 3. Load and validate the newly built snapshot
        # directly using its ID, before publication.
        new_snapshot = load_snapshot(snapshot_id)

        # 4. Build dense and sparse retrievers.
        # If this fails, the old snapshot remains active.
        new_retrievers = build_retrievers(new_snapshot)

        # 5. Publish only after loading and retriever
        # construction have both succeeded.
        publish_result = publish_snapshot(snapshot_id)

        # 6. Replace the application's in-memory state.
        app.state.snapshot = new_snapshot
        app.state.retrievers = new_retrievers

        logger.info(
            "Knowledge base updated successfully: %s",
            snapshot_id
        )

        return {
                "message": "Ingestion completed successfully",
                "snapshot_id": snapshot_id,
                "published": publish_result == snapshot_id,
                "report": report
            }

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Ingestion failed")

        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(error)}"
        )



# ============================================================
# MEMORY QUESTION REWRITING
# ============================================================

def contextualize_question(
    question: str,
    history: List[dict]
) -> str:

    if not history:
        return question

    chain = (
        MEMORY_PROMPT
        | app.state.llm
        | StrOutputParser()
    )

    return chain.invoke({
        "history": format_history(history),
        "question": question
    }).strip()


# ============================================================
# RAG QUERY PIPELINE
# ============================================================

def run_rag(question: str) -> dict[str, Any]:

    if app.state.retrievers is None:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is not ready. Run /ingest first."
        )

    # 1. Hybrid retrieval and RRF.
    candidates = retrieve_candidates(
        query=question,
        retrievers=app.state.retrievers
    )

    if not candidates:
        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "citations": [],
            "retrieval": {
                "candidate_count": 0,
                "selected_count": 0
            }
        }

    # 2. CrossEncoder reranking and context selection.
    reranking_result = rerank_and_select(
        query=question,
        candidates=candidates
    )

    selected_context = reranking_result["selected_context"]

    if not selected_context:
        return {
            "answer": (
                "I don't have enough information "
                "in the provided documents."
            ),
            "citations": [],
            "retrieval": reranking_result
        }

    # 3. Prepare evidence and citation IDs.
    citation_data = prepare_citations(
        selected_context
    )

    evidence_context = citation_data["evidence_context"]

    # 4. Generate answer.
    answer_chain = (
        ANSWER_PROMPT
        | app.state.llm
        | StrOutputParser()
    )

    answer = answer_chain.invoke({
        "context": evidence_context,
        "question": question
    })

    # 5. Associate answer references with evidence.
    citation_result = finalize_citations(
        answer=answer,
        evidence_records=citation_data["evidence_records"]
    )

    return {
        "answer": citation_result["answer"],
        "citations": citation_result["citations"],
        "retrieval": {
            "candidate_count": reranking_result["candidate_count"],
            "selected_count": reranking_result["selected_count"],
            "estimated_context_tokens": (
                reranking_result["estimated_context_tokens"]
            ),
            "retrieved_evidence_count": (
                citation_result["retrieved_evidence_count"]
            ),
            "cited_evidence_count": (
                citation_result["cited_evidence_count"]
            )
        }
    }


# ============================================================
# ASK API
# ============================================================


@app.post("/ask")
def ask(q: Query):
    try:

        # ====================================================
        # DATABASE MODE
        # ====================================================

        if q.mode == "database":

            client = get_olist_client()

            try:
                db_result = client.query(
                    question=q.question,
                    max_rows=100,
                    role="sales"
                )

            except OlistClientError as error:
                raise HTTPException(
                    status_code=error.status_code,
                    detail=error.message
                )

            return {
                "mode": "database",
                "answer": db_result["answer"],
                "citations": [],
                "database": {
                    "request_id": db_result["request_id"],
                    "evidence": db_result["evidence"],
                    "metadata": db_result["metadata"]
                }
            }

        # ====================================================
        # DOCUMENT MODE
        # ====================================================

        if q.mode == "document":

            history = get_memory(q.session_id)

            standalone_question = contextualize_question(
                q.question,
                history
            )

            result = run_rag(standalone_question)

            answer = result["answer"]

            save_memory(
                q.session_id,
                q.question,
                answer
            )

            return {
                "answer": answer,
                "citations": result["citations"],
                "retrieval": result["retrieval"]
            }

        # ====================================================
        # BOTH MODE
        # ====================================================

        if q.mode == "both":

            document_result = None
            database_result = None

            document_error = None
            database_error = None

            # -----------------------------------------------
            # 1. DOCUMENT RAG
            # -----------------------------------------------

            try:
                history = get_memory(q.session_id)

                standalone_question = contextualize_question(
                    q.question,
                    history
                )

                result = run_rag(standalone_question)

                document_result = {
                    "status": "success",
                    "answer": result["answer"],
                    "citations": result["citations"],
                    "retrieval": result["retrieval"]
                }

                save_memory(
                    q.session_id,
                    q.question,
                    result["answer"]
                )

            except Exception:
                logger.exception(
                    "Document RAG failed in both mode"
                )

                document_error = {
                    "status": "error",
                    "error": "Document retrieval failed."
                }

            # -----------------------------------------------
            # 2. DATABASE RAG
            # -----------------------------------------------

            try:
                client = get_olist_client()

                db_result = client.query(
                    question=q.question,
                    max_rows=100,
                    role="sales"
                )

                database_result = {
                    "status": "success",
                    "request_id": db_result["request_id"],
                    "answer": db_result["answer"],
                    "evidence": db_result["evidence"],
                    "metadata": db_result["metadata"]
                }

            except OlistClientError as error:
                database_error = {
                    "status": "error",
                    "http_status": error.status_code,
                    "error": error.message
                }

            except HTTPException as error:
                database_error = {
                    "status": "error",
                    "http_status": error.status_code,
                    "error": error.detail
                }

            except Exception:
                logger.exception(
                    "Database RAG failed in both mode"
                )

                database_error = {
                    "status": "error",
                    "error": "Database query failed."
                }

            # -----------------------------------------------
            # 3. CHECK SOURCE RESULTS
            # -----------------------------------------------

            if document_result is None and database_result is None:

                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "Both retrieval sources failed.",
                        "document": document_error,
                        "database": database_error
                    }
                )

            # -----------------------------------------------
            # 4. BUILD SEPARATE ANSWERS
            # -----------------------------------------------

            answer_parts = []

            if document_result is not None:
                answer_parts.append(
                    "DOCUMENT ANSWER:\n"
                    + document_result["answer"]
                )

            if database_result is not None:
                answer_parts.append(
                    "DATABASE ANSWER:\n"
                    + database_result["answer"]
                )

            combined_answer = "\n\n".join(answer_parts)

            return {
                "mode": "both",
                "answer": combined_answer,
                "sources": {
                    "document": (
                        document_result
                        if document_result is not None
                        else document_error
                    ),
                    "database": (
                        database_result
                        if database_result is not None
                        else database_error
                    )
                }
            }

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported mode: {q.mode}"
        )
    except HTTPException:
        raise

    except Exception as error:

        logger.exception("Question answering failed")

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {str(error)}"
        )
