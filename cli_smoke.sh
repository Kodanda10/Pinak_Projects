#!/usr/bin/env bash
set -euo pipefail

export PINAK_TOKEN="TEST_TOKEN"
BASE="http://127.0.0.1:8000"

echo "== CLI --help ="
pinak-memory --help || true

echo "== CLI health --help ="
pinak-memory health --help || true

echo "== CLI health ="
pinak-memory health --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI episodic --help ="
pinak-memory episodic --help || true

echo "== CLI episodic list ="
pinak-memory episodic --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI procedural --help ="
pinak-memory procedural --help || true

echo "== CLI procedural list ="
pinak-memory procedural --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI rag --help ="
pinak-memory rag --help || true

echo "== CLI rag list ="
pinak-memory rag --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI events --help ="
pinak-memory events --help || true

echo "== CLI events list ="
pinak-memory events --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI session --help ="
pinak-memory session --help || true

echo "== CLI session list ="
pinak-memory session --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI working --help ="
pinak-memory working --help || true

echo "== CLI working list ="
pinak-memory working --limit 5 --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI search --help ="
pinak-memory search --help || true

echo "== CLI search ="
pinak-memory search "FDI Railways" --url "$BASE" --token "$PINAK_TOKEN" || true

echo "== CLI add --help ="
pinak-memory add --help || true

echo "CLI smoke done."