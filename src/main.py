"""
main.py
----------------
Responsibility: Expose the RAG pipeline as a REST API.
One endpoint: POST /query
Returns: answer + citations + metadata

WHY FastAPI over Flask:
  - Automatic request/response validation via Pydantic
  - Async support for concurrent queries
  - Minimal wiring: uvicorn main:app --reload

HOW TO RUN (from src/):
  uvicorn main:app --reload --port 8000

HOW TO USE:
  Open chat.html from the project root (calls http://localhost:8000/query).
  http://localhost:8000/health — quick health check

HOW TO TEST with curl:
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What is a Key Decision Point in the NASA project life cycle?"}'
"""

import os
import sys
import time
import logging

# Add src to path so imports work
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_SRC_DIR)
_PROJECT_ROOT = os.path.normpath(os.path.join(_SRC_DIR, ".."))
_DEFAULT_HANDBOOK_PDF = os.path.join(
    _PROJECT_ROOT, "data", "nasa_systems_engineering_handbook_0.pdf"
)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import urllib.parse

from embedder import Embedder
from reference_graph import ReferenceGraph
from retriever import Retriever

LOG = logging.getLogger("nasa_handbook_rag")


def _configure_app_logging() -> None:
    """Structured app logs; suppress noisy Uvicorn per-request access lines."""
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Complex Technical Manual QA System",
    description="RAG API for question answering over technical manuals (NASA SE Handbook).",
    version="1.0.0",
    docs_url=None,      # no OpenAPI browser UI — use chat.html
    redoc_url=None,     # disable /redoc
    openapi_url=None,   # disable /openapi.json
)

# Allow all origins for local demo
# In production: restrict to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models (request/response validation)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="What is a Key Decision Point (KDP) in the NASA project life cycle?"
    )
    region: Optional[str] = Field(
        default="global",
        example="global"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to use as context"
    )


class Citation(BaseModel):
    chunk_id: str
    citation: str
    chunk_type: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    chunks_used: list[str]
    model: str
    tokens_used: int
    estimated_cost_usd: float
    latency_seconds: float


class HealthResponse(BaseModel):
    status: str
    vectors_loaded: int
    references_loaded: int
    model: str


# ---------------------------------------------------------------------------
# Startup: load models once, reuse across requests
# ---------------------------------------------------------------------------

# Global instances — loaded once at startup, not per-request
# WHY: loading FAISS index takes ~200ms. Loading per-request
# would make every query 200ms slower.
embedder: Optional[Embedder] = None
graph: Optional[ReferenceGraph] = None
retriever: Optional[Retriever] = None


@app.on_event("startup")
async def startup_event():
    """
    Load FAISS index and reference graph when server starts.
    If files don't exist, server starts but queries will fail
    with a clear error message.
    """
    global embedder, graph, retriever

    _configure_app_logging()

    index_path = "db/faiss.index"
    metadata_db = "db/metadata_store.db"
    graph_db = "db/reference_graph.db"

    try:
        print("Loading FAISS index...")
        embedder = Embedder(
            index_path=index_path,
            metadata_db=metadata_db
        )
        embedder.load()
        print(f"  Loaded {embedder.index.ntotal} vectors")

        print("Loading reference graph...")
        graph = ReferenceGraph(db_path=graph_db)
        stats = graph.get_stats()
        print(f"  Loaded {stats['total_references']} references")

        print("Initializing retriever...")
        retriever = Retriever(
            embedder=embedder,
            reference_graph=graph,
            top_k_initial=20,
            top_k_final=5,
            model="gpt-4o-mini"
        )

        print("Ready. API is live.")
        LOG.info(
            "Startup OK | vectors=%s | references=%s",
            embedder.index.ntotal,
            stats["total_references"],
        )

    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Run the ingestion pipeline first (from repo root):")
        print("  python src/pipeline.py")
        print("  OR: python src/embedder.py   (includes diagrams)")
        print("If index exists but is stale: python src/pipeline.py --reindex")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown."""
    if graph:
        graph.close()
    if embedder:
        embedder.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Quick health check — confirm index is loaded and ready.
    Use this to verify the server started correctly.
    """
    if not embedder or not graph:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run ingestion first."
        )

    stats = graph.get_stats()

    return HealthResponse(
        status="ready",
        vectors_loaded=embedder.index.ntotal,
        references_loaded=stats["total_references"],
        model="gpt-4o-mini"
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint.

    Takes a natural language question and returns:
    - A grounded answer (only from document content)
    - Per-claim citations (page, section, table, diagram)
    - Cost and latency metadata
    """
    if not retriever:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run ingestion first."
        )

    start_time = time.time()

    try:
        result = retriever.query(
            user_question=request.question,
            region=request.region,
            verbose=False
        )
    except Exception as e:
        LOG.exception("Query failed | question=%r", request.question[:200])
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

    latency = time.time() - start_time

    # Cost: gpt-4o-mini is $0.15/1M input + $0.60/1M output tokens
    # Approximating at blended $0.15/1M for simplicity
    cost = result["tokens_used"] * 0.00000015

    rstats = result.get("retrieval_stats") or {}
    LOG.info("========== QUERY ==========")
    LOG.info("question: %s", request.question)
    LOG.info(
        "retrieval: FAISS requested_top_k=%s | similar_hits=%s | "
        "after_cross_ref=%s | chunks_to_LLM=%s",
        rstats.get("faiss_top_k"),
        rstats.get("initial_hits"),
        rstats.get("after_cross_ref"),
        rstats.get("final_to_llm"),
    )
    for i, c in enumerate(result.get("context_chunks") or [], 1):
        cite = next(
            (x["citation"] for x in result["citations"] if x["chunk_id"] == c["chunk_id"]),
            "",
        )
        via = c.get("via_reference") or ""
        via_s = f"via_cross_ref={via}" if via else "direct_retrieval"
        snippet = (c.get("text") or "").replace("\n", " ").strip()
        if len(snippet) > 220:
            snippet = snippet[:220] + "…"
        LOG.info(
            "  context[%s] id=%s type=%s score=%.4f %s | %s",
            i,
            c.get("chunk_id"),
            c.get("chunk_type"),
            float(c.get("score", 0.0)),
            via_s,
            cite,
        )
        LOG.info("    snippet: %s", snippet)
    LOG.info("---------- ANSWER ----------")
    LOG.info("%s", result["answer"])
    LOG.info(
        "---------- META ---------- tokens=%s latency_s=%.2f cost_usd~=%s model=%s",
        result["tokens_used"],
        latency,
        round(cost, 6),
        result["model"],
    )

    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        chunks_used=result["chunks_used"],
        model=result["model"],
        tokens_used=result["tokens_used"],
        estimated_cost_usd=round(cost, 6),
        latency_seconds=round(latency, 2)
    )


@app.get("/")
async def root():
    """API root — open chat.html for the UI."""
    return {
        "message": "Complex Technical Manual QA System API is running.",
        "ui": "Open chat.html in your browser",
        "query": "POST /query with {question: string}",
        "health": "GET /health for status",
    }


@app.get("/page/{page_num}")
async def get_page(page_num: int):
    """Serve the PDF file for citation viewing."""
    pdf_path = _DEFAULT_HANDBOOK_PDF
    safe_filename = urllib.parse.quote(f"nasa_handbook_page_{page_num}.pdf")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=safe_filename
    )


# ---------------------------------------------------------------------------
# Run directly (alternative to uvicorn command)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        access_log=False,
    )