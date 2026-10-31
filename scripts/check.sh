#!/usr/bin/env bash
# Full quality gate used by CI and the automation: lint + tests for both apps, plus a production build.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== backend: ruff + pytest"
(cd backend && ruff check . && ruff format --check . && pytest -q)
echo "== frontend: typecheck + tests + build"
(cd frontend && npm run typecheck && npm test && NEXT_TELEMETRY_DISABLED=1 npm run build)
