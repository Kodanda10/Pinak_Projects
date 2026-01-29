# Architecture V2: Enterprise Memory Service

**Status:** Proposed
**Author:** Pinak Memory Architect
**Inspiration:** `qmd` (Quick Markdown Search), Enterprise Best Practices

## 1. Executive Summary

This document proposes a complete re-architecture of the Pinak Memory Service to transform it from a file-based (JSONL) prototype into a robust, thread-safe, enterprise-grade memory store for Autonomous Agents.

The core changes involve:
1.  **Storage Engine:** Migrating from raw JSONL files to **SQLite** (with FTS5) for metadata/logs and **FAISS** for vectors.
2.  **Search Strategy:** Moving from simple semantic-only search to **Hybrid Search** (Keyword + Semantic + RRF Fusion).
3.  **Agent Logic:** Introducing schemas specifically designed for Autonomous Agents (plans, outcomes, tool execution logs).
4.  **Concurrency:** Ensuring thread-safety for concurrent agent access.

## 2. Core Architecture

### 2.1 Storage Layer: SQLite + FAISS

We will use a "Hybrid Storage" approach to ensure zero external dependencies (no heavy DB servers) while gaining ACID compliance.

*   **SQLite (`memory.db`):**
    *   Stores all metadata, text content, logs, and structured agent data.
    *   Uses **FTS5** (Full Text Search) extension (built-in to Python's `sqlite3` usually) for high-performance keyword search.
    *   Handles concurrency via WAL (Write-Ahead Log) mode.
*   **FAISS (`vectors.index`):**
    *   Stores the high-dimensional embeddings.
    *   Maps internal integer IDs to SQLite UUIDs.
    *   Managed by a thread-safe wrapper.

### 2.2 Schema Design

#### Table: `memories_semantic` (Knowledge Base)
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID (PK) | Unique ID |
| `content` | TEXT | The raw knowledge snippet |
| `tags` | TEXT | JSON array of tags |
| `created_at` | DATETIME | Timestamp |
| `tenant` | TEXT | Multi-tenancy |
| `project_id` | TEXT | Project isolation |
| `embedding_id` | INT | Link to FAISS index ID |

#### Table: `memories_episodic` (Experiences)
*Critical for Autonomous Agents to learn from past actions.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID (PK) | Unique ID |
| `content` | TEXT | Summary of the episode |
| `goal` | TEXT | What was the agent trying to do? |
| `plan` | TEXT | JSON array of planned steps |
| `outcome` | TEXT | "Success", "Failure", "Partial" |
| `tool_logs` | TEXT | JSON summary of tools used and outputs |
| `salience` | INT | Importance score (1-10) |
| `created_at` | DATETIME | Timestamp |
| `tenant` | TEXT | |
| `project_id` | TEXT | |

#### Table: `memories_procedural` (Skills/SOPs)
*How-to knowledge for agents.*

| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | UUID (PK) | Unique ID |
| `skill_name` | TEXT | Name of the skill |
| `trigger` | TEXT | When should this be used? |
| `steps` | TEXT | JSON list of steps |
| `code_snippet` | TEXT | Optional executable code |
| `created_at` | DATETIME | |

#### Table: `logs_events` & `logs_session`
Standard audit and session logs, stored in SQLite for fast range queries (e.g., "events in last 1h").

### 2.3 Hybrid Search & Retrieval (The "Magic" Pipeline)

We will implement a retrieval pipeline inspired by `qmd` to provide the most relevant context.

**Algorithm: Reciprocal Rank Fusion (RRF)**

1.  **Query Analysis:**
    *   The user query is cleaned.
    *   (Future) Query expansion using an LLM.

2.  **Parallel Retrieval:**
    *   **Path A (Semantic):** Encode query -> Search FAISS -> Get top K (e.g., 20) -> Retrieve metadata from SQLite.
    *   **Path B (Keyword):** Search SQLite FTS5 (`MATCH query`) -> Get top K (e.g., 20).

3.  **Fusion (RRF):**
    *   Combine results.
    *   Score = `1 / (k + rank_semantic) + 1 / (k + rank_keyword)`.
    *   This boosts results that appear in *both* searches (meaning they match conceptually AND contain specific keywords).

4.  **Re-Ranking (Optional/Future):**
    *   Apply a Cross-Encoder to the top 10 fused results for final ordering.

5.  **Output:**
    *   Returns a unified `Context` object.

## 3. Implementation Plan

### 3.1 Directory Structure
```
app/
  core/
    database.py   # SQLite Manager
    config.py
  services/
    memory.py     # Main Orchestrator
    storage.py    # DB Operations
    vector.py     # FAISS Wrapper
    search.py     # Hybrid Search Logic
  api/
    v1/
      endpoints.py
```

### 3.2 Key Dependencies
- `sqlite3` (Standard Lib)
- `faiss-cpu`
- `sentence-transformers`
- `numpy`
- `pydantic`

## 4. Migration Strategy

1.  **Init:** New service starts, creates `data/memory.db`.
2.  **Migration Script:** A script `scripts/migrate_v1_v2.py` will read existing JSONL files and insert them into SQLite/FAISS.
3.  **Dual Run (Optional):** Not planned; we will do a hard cut-over for simplicity as this is dev/alpha stage.

## 5. Security & Multi-Tenancy

- **Row-Level Security:** Every SQL query *must* include `WHERE tenant=? AND project_id=?`.
- **Encryption:** (Future) SQLite supports encryption extensions (SEE) or we can encrypt content at app level.

## 6. Observability
- **Structured Logging:** All API hits and internal errors logged with JSON format.
- **Metrics:** (Future) Track "Search Latency", "Cache Hit Rate".
