#!/usr/bin/env bash
# Install dependencies for both apps (Python 3.10+, Node 22+).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -r backend/requirements-dev.txt
(cd frontend && npm ci)
