# CLAUDE.md — conventions for AI contributors

Read by the automated daily-dev, PR-review, PR-maintenance and backlog-planner workflows.

## Project
`rag-starter` is a retrieval-augmented generation starter:
- `backend/` — FastAPI (Python 3.10+): chunking (`app/chunking.py`), embeddings (`app/embeddings.py`),
  vector store (`app/store.py`), answer generation (`app/answer.py`), HTTP API (`app/main.py`).
- `frontend/` — Next.js App Router + TypeScript: ingest + ask UI with cited answers (`app/page.tsx`, `lib/api.ts`).

## Rules
- The project must run and pass tests **without API keys or network**. Real providers (Claude, OpenAI,
  Voyage, pgvector, Qdrant…) are optional adapters behind the existing protocols (`Embedder`, `Answerer`,
  a store interface), selected by environment variables, with the offline defaults kept.
- Tests: pytest in `backend/tests/` (use `fastapi.testclient`, mock all HTTP); `node:test` in `frontend/lib/*.test.ts`.
  Never call real external APIs in tests.
- Python: typed, small functions, `ruff check` and `ruff format` clean. Frontend: strict TypeScript, accessible
  markup (labels, focus styles, `aria-live` for async status), no UI library unless the issue asks for one.
- New Python dependencies go in `backend/requirements.txt` (runtime) or `requirements-dev.txt` (dev); keep them few.
- Run `bash scripts/check.sh` before finishing; it must pass.
- One issue per PR, small diffs. Update README.md / docs when behaviour or the API changes.
- Never modify `.github/`, deployment secrets, or anything that handles credentials.
