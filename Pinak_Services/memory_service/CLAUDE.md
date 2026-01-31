# 🏹 Pinak Memory Protocol (PMP)

This file defines the mandatory interaction pattern for all agents (CLI/IDE) leveraging the Pinak Memory Substrate for long-term intelligence.

## <law> Neural Synchrony </law>
1. **Always Synchronize on Pulse**: Every new session MUST start with a context retrieval call to the Pinak Memory Service.
2. **Episodic Trace Mandatory**: Significant task outcomes MUST be logged to the Episodic layer upon completion.
3. **Procedural Distillation**: High-salience patterns identified during work MUST be distilled into the Procedural layer at session close.
4. **Tenant Isolation**: Use `TENANT=default` and `PROJECT_ID=pinak-memory` for this repository.

## 🛠️ Operational Commands

### 1. Ingest History (One-Time Setup)
```bash
# Ingest context from conversation_summaries
uv run python scripts/ingest_history.py
```

### 2. Live Dashboard
```bash
# Launch the Pinak OS TUI
uv run python cli/main.py tui
```

### 3. Verify Substrate
```bash
# Run integrity checks
curl -H "Authorization: Bearer <TOKEN>" http://localhost:8000/api/v1/memory/audit/list
```

## 🧠 Memory Map
- **Semantic**: Static project identities and core constraints.
- **Episodic**: Time-sequenced session summaries (The 19 ingested sessions live here).
- **Procedural**: Verified workflows and skill patterns.
- **Working**: Ephemeral buffers for current task reasoning.
