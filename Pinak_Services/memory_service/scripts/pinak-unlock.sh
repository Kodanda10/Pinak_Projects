#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_MARK="/Users/abhi-macmini/pinak-memory/LOCKED"

if [ ! -d "$BASE_DIR" ]; then
  echo "Repo root not found: $BASE_DIR" >&2
  exit 1
fi

/bin/echo "Unlocking $BASE_DIR (requires macOS admin password)" 

# Remove system immutable flag, then restore user write permissions
sudo /usr/bin/chflags -R noschg "$BASE_DIR"
sudo /bin/chmod -R u+rwX "$BASE_DIR"
sudo /bin/chmod -R go-rwx "$BASE_DIR"

/bin/rm -f "$LOCK_MARK"
/bin/echo "Unlocked."
