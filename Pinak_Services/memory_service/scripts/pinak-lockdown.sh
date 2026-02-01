#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_MARK="/Users/abhi-macmini/pinak-memory/LOCKED"

if [ ! -d "$BASE_DIR" ]; then
  echo "Repo root not found: $BASE_DIR" >&2
  exit 1
fi

/bin/echo "Locking $BASE_DIR (requires macOS admin password)" 

# Remove ACLs and strip group/other permissions
sudo /bin/chmod -R -N "$BASE_DIR" || true
sudo /bin/chmod -R go-rwx "$BASE_DIR"
# Apply system immutable flag (requires admin password to remove)
sudo /usr/bin/chflags -R schg "$BASE_DIR"

/bin/mkdir -p "$(dirname "$LOCK_MARK")"
/bin/date > "$LOCK_MARK"
/bin/echo "Locked. To unlock: scripts/pinak-unlock.sh"
