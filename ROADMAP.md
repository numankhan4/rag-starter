# Roadmap

The backlog lives in GitHub Issues labeled `roadmap`. This file sets direction for the weekly planner.

1. **Foundation** — config via env vars, structured logging, error handling, OpenAPI docs polish.
2. **Ingestion** — Markdown/HTML/PDF loaders, smarter chunking, metadata, a WordPress connector.
3. **Retrieval** — pluggable embedders (Voyage, OpenAI, local), persistent stores (SQLite, pgvector, Qdrant),
   hybrid BM25 + vector search, reranking, filters.
4. **Generation** — Claude answerer with streaming (SSE), prompt templates, citation enforcement, refusal when unsupported.
5. **Frontend** — streaming chat UI, source previews, document manager, upload, dark mode, accessibility.
6. **Quality** — retrieval evaluation harness (recall@k, MRR) with a fixture dataset, answer faithfulness checks.
7. **Ops** — Docker Compose with Postgres/pgvector, deploy guides (Fly.io, Render, Vercel), rate limiting, API keys.

Non-goals: a hosted SaaS, user accounts, or features that require paid APIs to run the test suite.
