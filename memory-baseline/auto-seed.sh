#!/bin/zsh

# Usage: ./auto-seed.sh [project_dir]  # Default: current dir

SEED_DIR=/Users/abhijita/memory-baseline/blueprint-agent-memory
PROJ_DIR=${1:-$(pwd)}
FLAG_FILE="$PROJ_DIR/.blueprint-seeded"

if [ -f "$FLAG_FILE" ]; then
  echo "Project already seeded. Skipping."
  exit 0
fi

# Copy blueprint files
cp "$SEED_DIR/blueprint.md" "$PROJ_DIR/"
cp "$SEED_DIR/implementation.md" "$PROJ_DIR/"
cp "$SEED_DIR/memory.json" "$PROJ_DIR/"  # Config

# Install deps if Python project
if [ -f "$PROJ_DIR/requirements.txt" ]; then
  pip install -r "$PROJ_DIR/requirements.txt" || echo "Deps install failed; manual review needed."
else
  echo "No requirements.txt; skipping deps."
fi

# Git add if repo
if [ -d "$PROJ_DIR/.git" ]; then
  git add blueprint.md implementation.md memory.json
  git commit -m "Seed 2025 Agent Memory Blueprint" || echo "Commit skipped (no changes)."
fi

touch "$FLAG_FILE"
echo "Seeded: $PROJ_DIR"

# For all existing: find ~/projects -type d -maxdepth 1 -exec ./auto-seed.sh {} \;
