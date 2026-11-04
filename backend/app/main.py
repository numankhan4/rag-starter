"""FastAPI app: ingest documents, then ask questions answered from them with citations."""

from __future__ import annotations

import logging
import os
from http import HTTPStatus
from pathlib import PurePath
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .answer import ExtractiveAnswerer
from .chunking import chunk_text
from .embeddings import HashingEmbedder
from .loaders import LOADERS
from .logging_setup import REQUEST_ID_HEADER, RequestIDMiddleware, configure_logging
from .store import InMemoryVectorStore


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)
    title: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=4, ge=1, le=20)


class Source(BaseModel):
    n: int
    doc_id: str
    title: str | None
    chunk: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[int]
    sources: list[Source]


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody


_ERRORS: dict[int | str, dict[str, object]] = {
    status: {"model": ErrorResponse, "description": desc}
    for status, desc in ((404, "Not found"), (422, "Validation error"), (500, "Unexpected error"))
}


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


def _code_for(status: int) -> str:
    try:
        return HTTPStatus(status).name.lower()
    except ValueError:
        return "error"


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def on_http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, _code_for(exc.status_code), str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def on_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = [
            f"{'.'.join(str(p) for p in e['loc'] if p != 'body')}: {e['msg']}".lstrip(": ")
            for e in exc.errors()
        ]
        return _error(422, "validation_error", "; ".join(parts) or "Invalid request")

    @app.exception_handler(Exception)
    async def on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logging.getLogger("app.error").error("unhandled exception", exc_info=exc)
        return _error(500, "internal_error", "Internal server error")


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="rag-starter", version="0.1.0", responses=_ERRORS)
    _install_error_handlers(app)
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.add_middleware(RequestIDMiddleware)

    embedder = HashingEmbedder()
    store = InMemoryVectorStore()
    answerer = ExtractiveAnswerer()
    titles: dict[str, str | None] = {}

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "chunks": len(store), "documents": len(titles)}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> IngestResponse:
        store.delete_document(req.doc_id)
        chunks = chunk_text(req.doc_id, req.text)
        if not chunks:
            raise HTTPException(status_code=422, detail="Document has no text")
        store.add(chunks, embedder.embed([c.text for c in chunks]))
        titles[req.doc_id] = req.title
        return IngestResponse(doc_id=req.doc_id, chunks=len(chunks))

    @app.post("/ingest/file", response_model=IngestResponse)
    async def ingest_file(
        file: Annotated[UploadFile, File()],
        doc_id: Annotated[str | None, Form(max_length=200)] = None,
        title: Annotated[str | None, Form()] = None,
    ) -> IngestResponse:
        name = PurePath(file.filename or "").name
        loader = LOADERS.get(PurePath(name).suffix.lower())
        if loader is None:
            raise HTTPException(status_code=415, detail="Supported files: .md, .html, .txt")
        try:
            content = (await file.read()).decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(status_code=422, detail="File must be UTF-8 text") from None
        loaded = loader(content)
        return ingest(
            IngestRequest(
                doc_id=doc_id or name or "upload",
                text=loaded.text or " ",
                title=title or loaded.title or name or None,
            )
        )

    @app.delete("/documents/{doc_id}")
    def delete(doc_id: str) -> dict[str, int]:
        removed = store.delete_document(doc_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Unknown document")
        titles.pop(doc_id, None)
        return {"removed_chunks": removed}

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest) -> QueryResponse:
        hits = store.search(embedder.embed([req.question])[0], k=req.k)
        result = answerer.answer(req.question, hits)
        sources = [
            Source(
                n=i,
                doc_id=h.chunk.doc_id,
                title=titles.get(h.chunk.doc_id),
                chunk=h.chunk.index,
                score=round(h.score, 4),
                text=h.chunk.text,
            )
            for i, h in enumerate(hits, start=1)
        ]
        return QueryResponse(answer=result.text, citations=result.citations, sources=sources)

    return app


app = create_app()
