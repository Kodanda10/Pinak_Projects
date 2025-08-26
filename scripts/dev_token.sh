#!/usr/bin/env bash
set -euo pipefail

# Mint a development JWT (HS256) for local use.
# Usage:
#   ./scripts/dev_token.sh [--sub <subject>] [--secret <SECRET_KEY>] [--set]
#   SECRET_KEY defaults to change-me-in-prod

SUB="analyst"
SECRET_KEY="${SECRET_KEY:-change-me-in-prod}"
SET_TOKEN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sub) SUB="$2"; shift 2 ;;
    --secret) SECRET_KEY="$2"; shift 2 ;;
    --set) SET_TOKEN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT_DIR/.venv"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip >/dev/null
"$VENV/bin/pip" -q install "python-jose[cryptography]" >/dev/null

TOKEN=$("$VENV/bin/python" - <<PY
from jose import jwt
import os
sub = os.getenv('SUBJECT','${SUB}')
secret = os.getenv('SECRET_KEY','${SECRET_KEY}')
print(jwt.encode({'sub': sub}, secret, algorithm='HS256'))
PY
)

echo "$TOKEN"

if [[ "$SET_TOKEN" -eq 1 ]]; then
  if ! "$VENV/bin/pinak-bridge" token set --token "$TOKEN" >/dev/null 2>&1; then
    "$VENV/bin/pip" -q install -e "$ROOT_DIR" >/dev/null
    "$VENV/bin/pinak-bridge" token set --token "$TOKEN"
  else
    echo "Stored token via pinak-bridge." >&2
  fi
fi

