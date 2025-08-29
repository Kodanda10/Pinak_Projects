#!/usr/bin/env bash
set -euo pipefail

# Activate the virtual environment
source /Users/abhijita/Pinak_Package/.venv/bin/activate

# 1) Ensure deps
pip install --quiet fastapi uvicorn pytest requests >/dev/null

# 2) Start mock server on :8000
python3 mock_server.py &
MOCK_PID=$!
echo "Mock server PID: $MOCK_PID"
sleep 1

# 3) Pytest (unit/integration via client)
cd /Users/abhijita/Pinak_Package/Pinak_Services/memory_service && pytest -q --disable-warnings -k "memory_layers" || true && cd -

# 4) CLI smoke (only if entry exists)
if command -v pinak-memory >/dev/null 2>&1; then
  bash cli_smoke.sh || true
else
  echo "WARN: pinak-memory CLI not found on PATH. Skip CLI smoke."
fi

# 5) Cleanup
kill "$MOCK_PID" 2>/dev/null || true
wait "$MOCK_PID" 2>/dev/null || true
echo "All done."