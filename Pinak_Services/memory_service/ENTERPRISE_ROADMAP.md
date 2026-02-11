# Enterprise Roadmap: Pinak Memory Service

## 1. Executive Summary
The Pinak Memory Service is a capable, agentic memory system with a robust layered architecture (Semantic, Episodic, Procedural). It successfully implements Hybrid Search (combining SQLite FTS5 and Vector Search) and detailed observability.

However, to become "Enterprise Grade," it requires significant upgrades in scalability, reliability, and deployment infrastructure. The current `VectorStore` (NumPy-based) is a prototype suitable for <10k vectors but will fail under production loads.

## 2. Codebase Assessment

### ✅ The Good (Strengths)
*   **Architecture:** Clean separation of concerns. The `MemoryService` orchestrates `DatabaseManager` and `VectorStore` effectively.
*   **Hybrid Search:** The implementation of Reciprocal Rank Fusion (RRF) between Keyword and Vector search is advanced and provides high-quality retrieval.
*   **Observability:** Extensive logging of access events, client issues, and audit trails (`logs_access`, `logs_audit`) is built-in.
*   **Agentic Design:** The data model (Goal, Plan, Outcome, Steps) is specifically designed for autonomous agents, not just generic RAG.

### ❌ The Bad (Weaknesses)
*   **Vector Store Scalability:** The current `VectorStore` loads the entire index into memory and performs an $O(N)$ linear scan for every search. It uses `np.save` to dump the whole index to disk, which is slow and risky.
*   **Database Abstraction:** `DatabaseManager` relies on raw SQL strings and leaks `sqlite3` connection objects (via `get_cursor`). It lacks connection pooling.
*   **Dependency Hygiene:** `redis` is listed in `pyproject.toml` but unused in the codebase.
*   **Blocking I/O:** CPU-bound operations (like `model.encode`) are executed synchronously within async API endpoints, potentially blocking the event loop under load.

## 3. Implementation Plan

To achieve Enterprise Grade status, follow this phased roadmap:

### Phase 1: Scalability (The "Vector" Upgrade)
**Goal:** Replace the NumPy prototype with a production-ready Vector Engine.

1.  **Migrate to pgvector (Recommended)**:
    *   Since the goal is Enterprise, PostgreSQL is the standard.
    *   Replace `sqlite3` with `psycopg` (async).
    *   Use `pgvector` extension for embedding storage.
    *   *Why?* ACID compliance, true scalability, and simplified architecture (one DB for metadata + vectors).

2.  **Alternative: Fix FAISS**:
    *   If staying with the current architecture, re-integrate `faiss-cpu`.
    *   Ensure `IndexIVFFlat` is used for larger datasets.
    *   Wrap FAISS operations in a dedicated thread/process to avoid blocking the main loop.

### Phase 2: Reliability (The "Database" Hardening)
**Goal:** Ensure data integrity and robust connection management.

1.  **Adopt an ORM (SQLAlchemy/SQLModel)**:
    *   Replace raw SQL strings with Pydantic-based models (`SQLModel`).
    *   This provides type safety, prevents SQL injection (though `?` parameters help now), and supports multiple dialects.

2.  **Connection Pooling**:
    *   Implement `SQLAlchemy`'s connection pool to manage DB connections efficiently, preventing "too many clients" errors.

3.  **Migration System**:
    *   Introduce `alembic` for database schema migrations. The current `_init_db` approach is brittle for schema evolution.

### Phase 3: Performance & Infrastructure
**Goal:** High throughput and low latency.

1.  **Async/Non-blocking**:
    *   Offload embedding generation (`model.encode`) to a separate thread pool or a dedicated microservice (e.g., TEI - Text Embeddings Inference).
    *   Refactor `MemoryService` methods to be `async def` and use `await` for all I/O.

2.  **Caching**:
    *   Actually use the `redis` dependency to cache frequent queries or session contexts.

3.  **Docker & CI/CD**:
    *   Create a production `Dockerfile` (multi-stage build).
    *   Set up a `docker-compose.prod.yml` with Postgres (`pgvector`) and Redis.

### Phase 4: Security
**Goal:** Secure multi-tenant access.

1.  **Secret Management**:
    *   Move from `os.getenv` to a strict `pydantic-settings` configuration.
    *   Integrate with a secrets manager (e.g., Vault, AWS Secrets Manager) for production.

2.  **RBAC**:
    *   Enforce granular permissions beyond just "memory.read/write" scopes.

## 4. Verification
Run the verification script `scripts/verify_service_core.py` after each major refactor to ensure core logic (Hybrid Search, Memory Storage) remains intact.
