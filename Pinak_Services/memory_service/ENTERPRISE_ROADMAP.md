# Enterprise Roadmap: Pinak Memory Service

## 1. Executive Summary & Verification
The Pinak Memory Service is a functional prototype with a modular architecture, comprehensive test suite (72 passing tests), and thoughtful features like "Quarantine" and "Observability". However, its reliance on SQLite, in-memory NumPy vector storage, and synchronous execution makes it unsuitable for production enterprise workloads.

**Current Status:**
- **Tests:** 72 Passed, 0 Failed.
- **Architecture:** Modular (Service, DB, VectorStore layers).
- **Storage:** SQLite + Local .npy files.
- **Vector Search:** Linear scan ($O(N)$).

## 2. Weaknesses (The "Bad")
*   **Scalability:**
    *   **Vector Store:** The custom `VectorStore` loads all vectors into RAM and performs a linear scan. This is $O(N)$ and will become a bottleneck as the dataset grows (>10k vectors).
    *   **Database:** SQLite handles concurrency poorly (file locking) compared to PostgreSQL.
    *   **Concurrency:** The service methods are synchronous blocking calls. In a high-throughput environment, this will block the event loop.
*   **Persistence & Reliability:**
    *   **Data Loss Risk:** The `VectorStore` dumps the entire array to disk on save. A crash during save could corrupt the index.
    *   **No Connection Pooling:** `DatabaseManager` opens a new connection for every operation.
*   **Dependencies:**
    *   `faiss-cpu` is listed but unused (replaced by NumPy).
    *   `fastmcp` is missing from `pyproject.toml` but used in `client/`.

## 3. Enterprise Grade Roadmap

### Phase 1: Storage Migration (The "Ironclad" Foundation)
**Goal:** Move from local files to robust server-based storage.

1.  **Migrate to PostgreSQL:**
    *   Replace SQLite with PostgreSQL.
    *   Use `SQLAlchemy` or `SQLModel` for ORM to support both (or just Postgres).
    *   Implement connection pooling (e.g., `pgbouncer` or internal pool).
2.  **Adopt pgvector:**
    *   Replace the custom NumPy `VectorStore` with `pgvector` extension in PostgreSQL.
    *   This provides ACID compliance for vectors, HNSW indexing for speed ($O(\log N)$), and simplifies operations (one DB to backup).
    *   *Alternative:* Qdrant or Weaviate if vector scale is massive (>100M).

### Phase 2: Async Architecture (Performance)
**Goal:** Non-blocking high concurrency.

1.  **Async Refactor:**
    *   Convert `MemoryService` methods to `async def`.
    *   Use `asyncpg` or `databases` library for async DB access.
    *   Ensure vector operations (if kept in Python) run in a threadpool to avoid blocking the event loop.
2.  **Task Queue:**
    *   Move heavy "background" tasks (like `_rebuild_index`, `verify_and_recover`, or complex "intent sniffing") to a task queue (e.g., Celery, Redis Queue, or ARQ).

### Phase 3: Deployment & Security (Ops)
**Goal:** Production readiness.

1.  **Dockerization:**
    *   Create a multi-stage `Dockerfile` optimized for production (small image size).
    *   Add `docker-compose.yml` for local dev (Service + Postgres + Redis).
2.  **Security Hardening:**
    *   Replace `PINAK_JWT_SECRET` environment variable dependency with a secrets manager integration (Vault/AWS Secrets Manager) for rotation.
    *   Implement strict Role-Based Access Control (RBAC) scopes beyond simple "read/write".
    *   Audit `datetime.utcnow()` usage (deprecated) and replace with `datetime.now(timezone.utc)`.

### Phase 4: Developer Experience (DX)
**Goal:** Ease of use for agent developers.

1.  **SDK/Client:**
    *   Fix the missing `fastmcp` dependency.
    *   Publish a proper Python SDK (client library) to PyPI.
2.  **Documentation:**
    *   Generate OpenAPI (Swagger) docs automatically and host them.

## 4. Immediate "Quick Wins"
- [ ] Add `fastmcp` to `pyproject.toml`.
- [ ] Replace `datetime.utcnow()` with `datetime.now(datetime.UTC)`.
- [ ] Configure a proper logger (structlog) instead of standard `logging`.
