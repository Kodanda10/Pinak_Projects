#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"
BACKUP_ROOT="/Users/abhi-macmini/pinak-memory/backups"
LOG_FILE="/Users/abhi-macmini/Library/Logs/pinak-memory-backup.log"

mkdir -p "$BACKUP_ROOT"

stamp=$(date +"%Y%m%d-%H%M%S")
backup_dir="$BACKUP_ROOT/$stamp"
mkdir -p "$backup_dir"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

log "backup_start dir=$backup_dir"

# SQLite consistent backup
if [ -f "$DATA_DIR/memory.db" ]; then
  /Users/abhi-macmini/clawd-simba/Pinak_Projects/Pinak_Services/memory_service/.venv/bin/python - <<'PY'
import sqlite3
import shutil
from pathlib import Path

data_dir = Path("/Users/abhi-macmini/clawd-simba/Pinak_Projects/Pinak_Services/memory_service/data")
backup_dir = Path("/Users/abhi-macmini/pinak-memory/backups")
# Use latest timestamp dir
latest = sorted(backup_dir.glob("*/"))[-1]

src = data_dir / "memory.db"
dst = latest / "memory.db"

con = sqlite3.connect(src)
with sqlite3.connect(dst) as bck:
    con.backup(bck)
con.close()
PY
else
  log "backup_skip missing_db"
fi

# Copy vector index
if [ -f "$DATA_DIR/vectors.index.npy" ]; then
  cp "$DATA_DIR/vectors.index.npy" "$backup_dir/"
fi

# Tar archive
( cd "$BACKUP_ROOT" && tar -czf "$stamp.tar.gz" "$stamp" )

# Optional: sync to Google Drive via rclone if configured
if command -v rclone >/dev/null 2>&1; then
  if rclone listremotes | grep -q "gdrive:"; then
    rclone copy "$BACKUP_ROOT/$stamp.tar.gz" gdrive:pinak-memory/backups/ >> "$LOG_FILE" 2>&1
    log "backup_synced remote=gdrive"
  else
    log "backup_skip no_gdrive_remote"
  fi
else
  log "backup_skip rclone_missing"
fi

log "backup_done archive=$BACKUP_ROOT/$stamp.tar.gz"
