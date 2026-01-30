# 🏹 Pinak Memory Service

Enterprise-grade memory substrate for autonomous agents.

## 🌟 How it Works

The Pinak Memory Service uses a dual-engine approach to ensure high-fidelity context retrieval while maintaining strict ACID compliance.

### 1. The Storage Engines
- **SQLite (FTS5)**: Handles metadata, relations, and full-text keyword indexing.
- **FAISS (Vector)**: Stores high-dimensional semantic embeddings for similarity-based retrieval.

### 2. Hybrid Retrieval (RRF)
Search results are combined using **Reciprocal Rank Fusion**. This ensures that results appearing high in both keyword and vector searches are prioritized, providing better relevance than either engine alone.

## 🚀 Installation & Configuration

### Prerequisites
- **uv** (Recommended) or pip
- **System Dependencies**: FAISS requires `libomp-dev` on Linux/macOS.

### Setup
```bash
# Install dependencies
uv sync

# Configure shared secret for JWT validation
export JWT_SECRET="your-secret"
```

## 🛠️ Operating the System

### Starting the FastAPI Server
```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Running the CLI Tool
Manage memories, mint tokens, and search from the command line:
```bash
# Mint a development token
uv run python cli/main.py mint --tenant demo --project default

# Hybrid search via CLI
uv run python cli/main.py search "how does hybrid search work?"
```

### Opening the TUI Dashboard
Use the interactive Terminal UI to browse and audit memories in real-time:
```bash
uv run python cli/main.py tui
```

## 🧪 Development & Testing

We enforce a strict TDD workflow using the **Ironclad Testing Protocol**.

```bash
# Run unit, property, and chaos tests
uv run pytest --cov=app --cov-branch tests/
```

### Invariants Verified
1. **Retrieval Consistency**: Vector IDs must always align with SQLite primary keys.
2. **Tenant Isolation**: Cross-tenant data leak results in immediate process termination during audit.
3. **Atomic Writes**: Database and Index state are committed together via transaction hooks.
