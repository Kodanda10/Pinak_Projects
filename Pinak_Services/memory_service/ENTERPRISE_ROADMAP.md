# Pinak Enterprise Memory Service Roadmap

This document outlines the architectural improvements required to elevate the Pinak Memory Service from a prototype to a robust, scalable, enterprise-grade system.

## 1. Database Architecture Upgrade

### Current State
- **Storage Engine**: SQLite with JSON columns.
- **Connection Management**: New connection per request (`sqlite3.connect`), leading to high latency and potential locking issues.
- **Transactions**: Implicit context manager based transactions.

### Target State
- **Storage Engine**: PostgreSQL 16+ (Required for concurrent writes and complex queries).
- **ORM/Query Builder**: SQLAlchemy (Async) or SQLModel.
- **Connection Pooling**: Use `asyncpg` with a robust connection pool (e.g., SQLAlchemy's `AsyncEngine`).
- **Migrations**: Alembic for schema version control.

### Implementation Plan
1.  Define SQLAlchemy models mirroring current SQLite schema.
2.  Implement `Alembic` for migration management.
3.  Replace `DatabaseManager` with an `AsyncSession` dependency injection pattern.
4.  Migrate data from SQLite to Postgres using a one-off script.

## 2. Vector Search Scalability

### Current State
- **Engine**: Custom NumPy implementation using exact L2 distance.
- **Complexity**: $O(N)$ search time (linear scan). Performance degrades significantly >10k vectors.
- **Persistence**: Entire index is serialized to disk (`np.save`) on every write (debounced), posing data loss risks and high I/O.
- **Concurrency**: `threading.RLock` limits throughput.

### Target State
- **Engine**: Dedicated Vector Database or Extension.
    - **Option A (Recommended for simplicity)**: **pgvector** extension for PostgreSQL. Allows unified transactional updates for metadata and vectors.
    - **Option B (Performance)**: **Qdrant** or **Weaviate** (Dockerized). Best for high-throughput, filtered semantic search.
- **Indexing**: HNSW (Hierarchical Navigable Small World) for $O(\log N)$ search.
- **Quantization**: Scalar Quantization (SQ8) or Product Quantization (PQ) for memory efficiency.

### Implementation Plan (pgvector path)
1.  Enable `vector` extension in PostgreSQL.
2.  Add `embedding vector(384)` column to `memories_*` tables.
3.  Create HNSW index on the embedding column.
4.  Rewrite `VectorStore` to simply execute SQL queries (`ORDER BY embedding <-> query_embedding LIMIT k`).
5.  Deprecate the NumPy `VectorStore` class.

## 3. Asynchronous & Non-Blocking Architecture

### Current State
- Mixed sync/async.
- `VectorStore` operations are synchronous and blocking.
- `DatabaseManager` uses blocking `sqlite3`.

### Target State
- **Fully Async**: All I/O operations (DB, Vector, API) must be `async def`.
- **Background Tasks**: Use `Celery` or `ARQ` for heavy lifting (e.g., bulk ingestion, re-indexing, summarization).

### Implementation Plan
1.  Replace `sqlite3` with `aiosqlite` (interim) or `asyncpg` (target).
2.  Ensure all API endpoints are `async def`.
3.  Offload "Proactive Nudging" and "Intent Sniffing" to background workers to reduce API latency.

## 4. Observability & Reliability

### Current State
- Custom logging tables (`logs_access`, `logs_audit`).
- Basic Python `logging`.

### Target State
- **Structured Logging**: JSON logs compatible with ELK/Datadog.
- **Tracing**: OpenTelemetry integration (instrument FastAPI, SQLAlchemy, HTTP clients).
- **Metrics**: Prometheus endpoint (request latency, vector search time, memory usage).
- **Health Checks**: Deep health check endpoint verifying DB and Vector Store connectivity.

## 5. Security Hardening

### Current State
- JWT Authentication (HS256).
- Tenant Isolation (Logic-based `WHERE` clauses).

### Target State
- **Key Rotation**: Support JWKS (JSON Web Key Sets) for key rotation.
- **Row Level Security (RLS)**: If using Postgres, enforce tenant isolation at the database level using RLS policies.
- **Secrets Management**: Integration with Vault or AWS Secrets Manager.

## 6. Testing & CI/CD

### Current State
- `pytest` suite.
- Dependent on local environment variables.

### Target State
- **Dockerized Tests**: Run tests against real Postgres/Qdrant containers using `testcontainers`.
- **Load Testing**: `locust` scripts to benchmark vector search at 100k+ vectors.
