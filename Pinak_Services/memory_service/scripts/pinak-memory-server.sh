#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

export PINAK_JWT_SECRET="${PINAK_JWT_SECRET:-secret}"
export PINAK_EMBEDDING_BACKEND="${PINAK_EMBEDDING_BACKEND:-dummy}"
export PINAK_EMBEDDING_TIMEOUT_MS="${PINAK_EMBEDDING_TIMEOUT_MS:-50}"
export PINAK_VERIFY_IN_BACKGROUND="${PINAK_VERIFY_IN_BACKGROUND:-1}"

exec "$BASE_DIR/.venv/bin/python" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
