# 🏹 Pinak Memory Service

**The Persistent Long-Term Context Substrate for AI Agents**

This service provides a unified, persistent memory layer for all AI agents operating within the Pinak ecosystem. It enables agents to "remember" past sessions, detect risks proactively, and maintain continuity across CLI and IDE environments.

## 🚀 Key Capabilities

### 1. **Persistent Long-Term Memory**
- **Semantic Layer**: Stores project identities and core constraints.
- **Episodic Layer**: Archives time-sequenced session summaries (Goals, Outcomes, Artifacts).
- **Procedural Layer**: Distills verified workflows and skill patterns.
- **Working Memory**: Ephemeral buffer for current task reasoning.

### 2. **Proactive Nudging** (New!)
- **Intent-Based Triggers**: Detects risky patterns in real-time based on agent intent.
- **Historical Risk Detection**: Surfaces past failures ("Nudges") relevant to the current task.
- **Cross-Layer Search**: Scans Semantic, Episodic, and Procedural layers simultaneously.

### 3. **Pinak Command Center (TUI)**
- **Agent Swarm View**: Real-time observability of all active and historical agent entities.
- **Deep Trace Analysis**: Browse session goals, outcomes, and timestamps.
- **System Health**: Live monitoring of database integrity and vector manifold state.

---

## 🛠️ Human Operations (Command Center)

### ✅ Install
```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install -e .
```

### 🟢 One-Command Startup
Launch the full Memory Command Center (Server + TUI) with a single command:
```bash
./pinak-memory
```
*This handles server startup, port checking, and UI visualization automatically.*
If `pinak-memory` is installed in PATH and cannot find the venv, set:
```bash
export PINAK_HOME="/path/to/memory_service"
```
Optional alias:
```bash
./pinak_memory
```

### 🔁 Auto‑Launch TUI on Login
```bash
cp scripts/com.pinak.memory.tui.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.pinak.memory.tui.plist
```

Disable:
```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.pinak.memory.tui.plist
```

### 🔍 Quick Context Search (CLI)
Query your persistent context directly without the UI:
```bash
export PINAK_JWT_SECRET="secret"
./.venv/bin/python -m cli.main search "disk cleanup strategies"
```

### 🩺 System Diagnostics
Run a full integrity check on the neural substrate:
```bash
./.venv/bin/python -m cli.main doctor
```

---

## 🧩 Client Schemas & Identity

### Schemas/Templates (no fetch needed)
- **Local canonical**: `~/pinak-memory/schemas` and `~/pinak-memory/templates`
- **API**: `/api/v1/memory/schema` and `/api/v1/memory/schema/{layer}`

Doctor syncs these automatically (`cli/main.py doctor --fix`).

### Client Identity Model
- **Client ID** is expected on write requests (non‑blocking).
- Missing `client_id` is logged as a **Client Issue** and can be backfilled by Doctor.
- Child agents should pass `X-Pinak-Child-Client-Id` so lineage is visible.

---

## 🧠 Embeddings Backend

Choose behavior via `PINAK_EMBEDDING_BACKEND`:
- `none` → keyword-only search (stable, no model downloads).
- `dummy` → deterministic embeddings (tests/dev).
- `qmd` → use QMD’s embedding pipeline (run `qmd embed` after indexing).

> QMD expects `embeddinggemma-300M-Q8_0.gguf`. Ensure the model is present before enabling.

---

## 🔐 Admin Lockdown (Source Protection)
To prevent non‑admin changes to the memory service source, run:
```bash
scripts/pinak-lockdown.sh
```
This requires macOS admin password and applies immutable flags. Unlock with:
```bash
scripts/pinak-unlock.sh
```

---

## 💾 Daily Backups (Google Drive)
- Backup script: `scripts/pinak-memory-backup.sh`
- LaunchAgent: `com.pinak.memory.backup.plist`
- Requires **rclone** with a `gdrive:` remote.

---

## 🤖 Agent Operations (MCP Integration)

Agents interface with memory via the **Model Context Protocol (MCP)**. Pinak Memory exposes an MCP server at:
```
client/pinak_memory_mcp.py --mcp
```

### Manual MCP Configuration (Recommended)
Point each agent to the MCP server script directly (no orchestrations or wrappers):
- **Command**: `.../memory_service/.venv/bin/python`
- **Args**: `.../memory_service/client/pinak_memory_mcp.py --mcp`
- **Env**:
  - `PINAK_API_URL=http://127.0.0.1:8000/api/v1`
  - `PINAK_JWT_SECRET=<secret>`
  - `PINAK_CLIENT_ID=<agent-id>`
  - `PINAK_CLIENT_NAME=<agent-name>`
  - `PINAK_PROJECT_ID=pinak-memory`

**Available Tools:**
- **`recall(query)`**: Semantic search across all memory layers. Used before starting tasks.
- **`remember_episode(goal, outcome, content, tags)`**: Stores execution results. Used after completing tasks.

---

## 🛡️ Pinak Memory Protocol (PMP)
*Standard operating procedure for all agents.*

1.  **Sync on Wake**: Call `recall()` at session start.
2.  **Trace on Completion**: Log results to `remember_episode`.
3.  **Distill on Close**: Save reusable patterns to procedural memory.

---

## 📄 Client Onboarding (Quickstart)

### 1) Register Client
```bash
export PINAK_JWT_SECRET="secret"
TOKEN=$(python -m cli.main mint)

curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/api/v1/memory/client/register \
  -d '{
    "client_id": "codex",
    "client_name": "codex-cli",
    "status": "trusted"
  }'
```

### 2) Write Memory (Semantic)
```bash
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Pinak-Client-Id: codex" \
  -H "X-Pinak-Client-Name: codex-cli" \
  http://127.0.0.1:8000/api/v1/memory/add \
  -d '{
    "content": "User prefers single-sprint delivery.",
    "tags": ["preference", "delivery"]
  }'
```

### 3) Child Agent (optional)
```bash
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Pinak-Client-Id: codex" \
  -H "X-Pinak-Child-Client-Id: codex-subagent-1" \
  http://127.0.0.1:8000/api/v1/memory/episodic/add \
  -d '{
    "content": "Summarized the deployment plan.",
    "timestamp": "2026-01-31T12:00:00Z"
  }'
```

### 4) Retrieve Context
```bash
curl -sS \
  -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/memory/retrieve_context?query=deployment plan"
```

### 5) Client Summary (Session Banner)
```bash
curl -sS \
  -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/memory/client/summary"
```

### Remote API (Tailscale)
Current server Tailscale IP:
```
http://100.66.59.92:8000/api/v1/memory
```
Headers:
- `Authorization: Bearer <token>`
- `X-Pinak-Client-Id`
- `X-Pinak-Client-Name`
- `X-Pinak-Child-Client-Id` (optional)

Manual approval: first‑time clients are `registered`. Mark them **trusted** in TUI → Clients tab.

Prefer `PINAK_JWT_TOKEN` for clients (no shared secret required).
Session banner is enabled by default (`PINAK_STARTUP_BANNER=1`). Set to `0` to disable.

### Schemas & Templates
- Local canonical: `~/pinak-memory/schemas`, `~/pinak-memory/templates`
- API: `/api/v1/memory/schema`, `/api/v1/memory/schema/{layer}`

Doctor syncs these automatically:
```bash
python -m cli.main doctor --fix
```

### Lockdown / Unlock (Source Protection)
```bash
sudo scripts/pinak-lockdown.sh
sudo scripts/pinak-unlock.sh
```

### Future: Token Rotation
Planned: short‑lived token issuer + rotation endpoint per client.

---

**Status**: ✅ Active & Stable | **Memories**: 19 | **Vector Engine**: Numpy-Accelerated
