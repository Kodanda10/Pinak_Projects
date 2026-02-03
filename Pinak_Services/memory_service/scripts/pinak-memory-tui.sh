#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

if [ -f "$HOME/pinak-memory/pinak.env" ]; then
  set -a
  . "$HOME/pinak-memory/pinak.env"
  set +a
fi

LAUNCH_CMD="$BASE_DIR/pinak-memory"

/usr/bin/osascript <<EOF
tell application "Terminal"
  activate
  do script "cd '$BASE_DIR' && '$LAUNCH_CMD'"
end tell
EOF
