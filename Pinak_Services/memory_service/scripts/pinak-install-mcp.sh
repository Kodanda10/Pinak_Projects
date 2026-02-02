#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINAK_HOME="${PINAK_MCP_HOME:-${HOME}/pinak-memory}"
TARGET_DIR="${PINAK_HOME}/bin"
SCHEMA_DIR="${PINAK_HOME}/schemas"
TEMPLATE_DIR="${PINAK_HOME}/templates"

PYTHON_BIN="${PINAK_MCP_PYTHON:-python3}"

/bin/mkdir -p "$TARGET_DIR"
/bin/mkdir -p "$SCHEMA_DIR"
/bin/mkdir -p "$TEMPLATE_DIR"

/bin/cp "$BASE_DIR/client/pinak_memory_mcp.py" "$TARGET_DIR/pinak_memory_mcp.py"
/bin/cp -R "$BASE_DIR/schemas/." "$SCHEMA_DIR/"
/bin/cp -R "$BASE_DIR/templates/." "$TEMPLATE_DIR/"

/bin/cat > "$TARGET_DIR/pinak-mcp" <<'EOF'
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${PINAK_ENV_FILE:-$SCRIPT_DIR/../pinak.env}"

if [ -f "$ENV_FILE" ]; then
  set -a
  . "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PINAK_MCP_PYTHON:-python3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/pinak_memory_mcp.py" "$@"
EOF

/bin/chmod 755 "$TARGET_DIR/pinak-mcp"
/bin/chmod 644 "$TARGET_DIR/pinak_memory_mcp.py"

/bin/echo "Installed MCP client to $TARGET_DIR"
/bin/echo "Schemas synced to $SCHEMA_DIR"
/bin/echo "Templates synced to $TEMPLATE_DIR"
