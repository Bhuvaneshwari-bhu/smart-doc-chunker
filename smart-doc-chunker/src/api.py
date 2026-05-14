"""
FastAPI service exposing the Smart Document Chunking + RAG pipeline.

Note: pipeline stages are called directly (not via pipeline.run()) because
pipeline.run() calls sys.exit() on failure, which would terminate the server
process. The logic is identical; only the error-handling mechanism differs.
"""

import json
import logging
import math
import os
from datetime import datetime

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from . import loader
from . import cleaner
from . import chunker
from . import tokenizer
from . import embedding_store
from . import rag_chat

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("smart-doc-chunker")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Smart Document Chunker API",
    description="RAG-ready document chunking and retrieval service.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ProcessRequest(BaseModel):
    file_path:  str
    method:     str = "recursive"
    chunk_size: int = 500
    overlap:    int = 50

    @field_validator("file_path")
    @classmethod
    def file_must_exist(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("file_path must not be empty")
        if not os.path.isfile(v):
            raise ValueError(f"file not found: {v}")
        return v

    @field_validator("method")
    @classmethod
    def method_must_be_valid(cls, v: str) -> str:
        allowed = {"recursive", "sliding"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return v

    @field_validator("overlap")
    @classmethod
    def overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("overlap must be >= 0")
        return v


class ProcessTextRequest(BaseModel):
    text:       str
    method:     str = "recursive"
    chunk_size: int = 500
    overlap:    int = 50

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    @field_validator("method")
    @classmethod
    def method_must_be_valid(cls, v: str) -> str:
        allowed = {"recursive", "sliding"}
        if v not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        return v

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be a positive integer")
        return v

    @field_validator("overlap")
    @classmethod
    def overlap_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("overlap must be >= 0")
        return v


class EmbedChunksRequest(BaseModel):
    chunks: list[dict]

    @field_validator("chunks")
    @classmethod
    def chunks_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("chunks must not be empty")
        return v


class AskChunksRequest(BaseModel):
    chunks: list[dict]
    query:  str
    top_k:  int = 3

    @field_validator("chunks")
    @classmethod
    def chunks_not_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("chunks must not be empty")
        return v

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


class AskRequest(BaseModel):
    query:     str
    file_path: str
    top_k:     int = 3

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()

    @field_validator("file_path")
    @classmethod
    def file_must_exist(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("file_path must not be empty")
        if not os.path.isfile(v):
            raise ValueError(f"file not found: {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def top_k_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("top_k must be a positive integer")
        return v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "output")
    os.makedirs(path, exist_ok=True)
    return path


def _run_pipeline_stages(
    file_path: str,
    method: str,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict], dict, str]:
    """
    Execute loader → cleaner → chunker → tokenizer → save JSON.
    Returns (chunks, stats, output_path).
    Raises HTTPException on any stage failure.
    """
    log.info("Pipeline stage started: load — %s", file_path)
    try:
        raw_text = loader.load_document(file_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Load failed: {exc}") from exc

    log.info("Pipeline stage started: clean")
    cleaned = cleaner.clean_text(raw_text)
    if not cleaned.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Text is empty after cleaning.")

    log.info("Pipeline stage started: chunk — method=%s chunk_size=%d overlap=%d",
             method, chunk_size, overlap)
    try:
        chunks = chunker.chunk_text(
            text=cleaned,
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Chunking produced no output.")

    stats = tokenizer.analyze_chunks(chunks)

    dataset = {
        "source_file": file_path,
        "method":      method,
        "chunk_size":  chunk_size,
        "overlap":     overlap,
        "chunks":      chunks,
        "stats":       stats,
    }

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(_output_dir(), f"chunks_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, indent=2, ensure_ascii=False)

    log.info("Pipeline completed — %d chunks saved to %s", len(chunks), output_path)
    return chunks, stats, output_path


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}


@app.post("/process", summary="Ingest and chunk a document")
def process_document(req: ProcessRequest):
    log.info("Request received: POST /process — file=%s method=%s",
             req.file_path, req.method)

    if req.overlap >= req.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"overlap ({req.overlap}) must be less than chunk_size ({req.chunk_size})",
        )

    chunks, _stats, output_path = _run_pipeline_stages(
        file_path=req.file_path,
        method=req.method,
        chunk_size=req.chunk_size,
        overlap=req.overlap,
    )
    return {
        "message":     "processed",
        "chunks":      len(chunks),
        "output_file": output_path,
    }


@app.post("/process-upload", summary="Upload a .txt file, chunk it, return results (no save)")
async def process_upload(
    file:       UploadFile = File(...),
    method:     str        = Form("recursive"),
    chunk_size: int        = Form(500),
    overlap:    int        = Form(50),
):
    log.info("Request received: POST /process-upload — file=%s method=%s", file.filename, method)

    allowed_methods = {"recursive", "sliding"}
    if method not in allowed_methods:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"method must be one of {sorted(allowed_methods)}")
    if chunk_size <= 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="chunk_size must be a positive integer")
    if overlap < 0:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="overlap must be >= 0")
    if overlap >= chunk_size:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    content_type = file.content_type or ""
    filename = file.filename or ""
    if not (filename.endswith(".txt") or "text/plain" in content_type):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Only .txt files are supported for upload.")

    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    cleaned = cleaner.clean_text(raw_text)
    if not cleaned.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="File is empty after cleaning.")

    try:
        chunks = chunker.chunk_text(text=cleaned, method=method,
                                    chunk_size=chunk_size, overlap=overlap)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Chunking produced no output.")

    full_stats = tokenizer.analyze_chunks(chunks)
    stats = {
        "total_chunks":        full_stats["total_chunks"],
        "avg_tokens_per_chunk": full_stats["avg_tokens_per_chunk"],
        "min_tokens":          full_stats["min_tokens"],
        "max_tokens":          full_stats["max_tokens"],
    }

    log.info("process-upload completed — %d chunks from %s", len(chunks), filename)
    return {"chunks": chunks, "stats": stats, "filename": filename}


@app.post("/process-text", summary="Chunk pasted text directly (no file, no save)")
def process_text(req: ProcessTextRequest):
    log.info("Request received: POST /process-text — method=%s chunk_size=%d overlap=%d",
             req.method, req.chunk_size, req.overlap)

    if req.overlap >= req.chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"overlap ({req.overlap}) must be less than chunk_size ({req.chunk_size})",
        )

    cleaned = cleaner.clean_text(req.text)
    if not cleaned.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Text is empty after cleaning.")

    try:
        chunks = chunker.chunk_text(
            text=cleaned,
            method=req.method,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Chunking produced no output.")

    full_stats = tokenizer.analyze_chunks(chunks)
    stats = {
        "total_chunks":        full_stats["total_chunks"],
        "avg_tokens_per_chunk": full_stats["avg_tokens_per_chunk"],
        "min_tokens":          full_stats["min_tokens"],
        "max_tokens":          full_stats["max_tokens"],
    }

    log.info("process-text completed — %d chunks", len(chunks))
    return {"chunks": chunks, "stats": stats}


@app.post("/embed-chunks", summary="Embed a chunk list and return 8-dim previews + vector length")
def embed_chunks_preview(req: EmbedChunksRequest):
    log.info("Request received: POST /embed-chunks — %d chunks", len(req.chunks))
    try:
        embedded = embedding_store.embed_chunks(req.chunks)
    except (KeyError, TypeError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Invalid chunk format: {exc}") from exc

    result = []
    for c in embedded:
        vec  = c["embedding"]
        vlen = math.sqrt(sum(x * x for x in vec))
        result.append({
            "chunk_id":      c["chunk_id"],
            "text":          c["text"],
            "tokens":        c["tokens"],
            "preview":       [round(x, 4) for x in vec[:8]],
            "vector_length": round(vlen, 4),
            "dim":           len(vec),
        })

    log.info("embed-chunks completed — %d embeddings", len(result))
    return {"embedded": result}


@app.post("/ask-chunks", summary="RAG over pre-supplied chunks (no file required)")
def ask_chunks(req: AskChunksRequest):
    log.info("Request received: POST /ask-chunks — query=%r top_k=%d chunks=%d",
             req.query, req.top_k, len(req.chunks))
    top_k    = min(req.top_k, len(req.chunks))
    embedded = embedding_store.embed_chunks(req.chunks)
    response = rag_chat.answer_query(
        query=req.query,
        embedded_chunks=embedded,
        top_k=top_k,
    )
    log.info("ask-chunks completed — answer=%d chars sources=%d",
             len(response["answer"]), len(response["sources"]))
    return response


@app.post("/ask", summary="Ask a question over a document (RAG)")
def ask_question(req: AskRequest):
    log.info("Request received: POST /ask — query=%r file=%s top_k=%d",
             req.query, req.file_path, req.top_k)

    # Inline RAG flow — each request is fully independent (stateless)
    log.info("Pipeline stage started: load — %s", req.file_path)
    try:
        raw_text = loader.load_document(req.file_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Load failed: {exc}") from exc

    log.info("Pipeline stage started: clean + chunk")
    cleaned = cleaner.clean_text(raw_text)
    if not cleaned.strip():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Text is empty after cleaning.")

    chunks = chunker.chunk_text(text=cleaned, method="recursive",
                                chunk_size=500, overlap=50)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Chunking produced no output.")

    log.info("Pipeline stage started: embed %d chunks", len(chunks))
    embedded = embedding_store.embed_chunks(chunks)

    log.info("Pipeline stage started: RAG retrieval — top_k=%d", req.top_k)
    response = rag_chat.answer_query(
        query=req.query,
        embedded_chunks=embedded,
        top_k=req.top_k,
    )

    log.info("Pipeline completed — answer generated, %d sources", len(response["sources"]))
    return response


# ---------------------------------------------------------------------------
# Validation error → clean JSON (no stack traces)
# ---------------------------------------------------------------------------

from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    errors = [
        {"field": ".".join(str(l) for l in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": errors},
    )
