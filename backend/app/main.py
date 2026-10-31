"""FastAPI app: ingest documents, then ask questions answered from them with citations."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .answer import ExtractiveAnswerer
from .chunking import chunk_text
from .embeddings import HashingEmbedder
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


def create_app() -> FastAPI:
    app = FastAPI(title="rag-starter", version="0.1.0")
    origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])

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
