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

### 🟢 One-Command Startup
Launch the full Memory Command Center (Server + TUI) with a single command:
```bash
./pinak-memory
```
*This handles server startup, port checking, and UI visualization automatically.*

### 🔍 Quick Context Search (CLI)
Query your persistent context directly without the UI:
```bash
export PINAK_JWT_SECRET="secret"
uv run python cli/main.py search "disk cleanup strategies"
```

### 🩺 System Diagnostics
Run a full integrity check on the neural substrate:
```bash
uv run python cli/main.py doctor
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

Agents interface with memory via the **Model Context Protocol (MCP)**. This abstracts authentication and vector complexity into simple tools.

### 1. Live Loading (Configuration)
The MCP server is now registered in your local Claude Desktop configuration. To activate it:

1.  **Restart Claude Desktop completely**.
2.  The tool `pinak-memory` will be available immediately.

### 2. Manual Loading (Dev/Debug)
You can also run the MCP server directly for testing or inspection:
```bash
uv run fastmcp run client/pinak_memory_mcp.py
```

**Available Tools:**
- **`recall(query)`**: Semantic search across all memory layers. Used before starting tasks.
- **`remember_episode(...)`**: Stores execution results. Used after completing tasks.

### 3. Agent-Specific Configuration Reference
The following files have been automatically updated to include `pinak-memory`:

| Agent | Configuration File |
| :--- | :--- |
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| **Google Gemini** | `~/.gemini/mcp.json` |
| **Google Antigravity** | `~/.gemini/antigravity/mcp_config.json` |
| **Codex CLI** | `~/.codex/mcp.json` (also uses `config.toml`, `auth.json`) |
| **Pi Coding Agent** | `~/.pi/agent/settings.json` |
| **AMP** | `~/.amp/mcp.json` |
| **OpenCode** | `~/.opencode/mcp_config.json` |
| **Cursor** | `~/.cursor/mcp.json` |

---

## 🛡️ Pinak Memory Protocol (PMP)
*Standard operating procedure for all agents.*

1.  **Sync on Wake**: Call `recall()` at session start.
2.  **Trace on Completion**: Log results to `remember_episode`.
3.  **Distill on Close**: Save reusable patterns to procedural memory.

---

## 📄 Client Onboarding
See `docs/CLIENT_ONBOARDING.md` for copy‑paste curl examples and schema guidance.

---

**Status**: ✅ Active & Stable | **Memories**: 19 | **Vector Engine**: Numpy-Accelerated
