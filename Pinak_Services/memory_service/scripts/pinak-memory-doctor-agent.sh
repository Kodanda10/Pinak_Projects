#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="/Users/abhi-macmini/Library/Logs/pinak-memory-doctor.log"
VENV_PY="$BASE_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
  echo "[doctor] Missing venv python at $VENV_PY" >> "$LOG_FILE"
  exit 1
fi

HEAVY_FLAG=""
if [ "${PINAK_DOCTOR_HEAVY:-0}" = "1" ]; then
  HEAVY_FLAG="--heavy"
fi

"$VENV_PY" cli/main.py doctor --fix $HEAVY_FLAG >> "$LOG_FILE" 2>&1

if [ "${PINAK_DOCTOR_LLM:-0}" = "1" ] && [ -x "/opt/homebrew/bin/gemini" ]; then
  tail -n 200 "$LOG_FILE" | /opt/homebrew/bin/gemini -p "Summarize the latest doctor run and highlight any unresolved issues." >> "$LOG_FILE" 2>&1
fi
