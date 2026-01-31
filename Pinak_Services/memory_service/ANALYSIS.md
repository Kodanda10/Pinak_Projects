# Memory Service Analysis & Enterprise Roadmap

## 1. Status Report
**Test Status:** ✅ **PASSED**
- All 51 tests passed.
- **Note:** Minor warnings observed regarding `datetime.utcnow()` deprecation (Technical Debt).

## 2. Codebase Assessment

### What is Good
- **Modular Architecture:** Clean separation between API (`endpoints.py`), Service (`memory_service.py`), and Data (`database.py`, `vector_store.py`).
- **Hybrid Search:** Implementation of Reciprocal Rank Fusion (RRF) combining SQLite FTS5 (Keyword) and FAISS (Semantic) is a solid foundation for high-recall retrieval.
- **Multi-Tenancy:** Strict enforcement of `tenant` and `project_id` in database queries ensures data isolation.
- **Security:** JWT-based authentication context is integrated.
- **Data Integrity:** "Ironclad" startup checks (`verify_and_recover`) ensure synchronization between Metadata (SQLite) and Vectors (FAISS).

### What is Bad / Needs Improvement
- **Single Vector Index:** Currently, only "Semantic" memory uses the Vector Store. Episodic and Procedural memories rely solely on keyword search (FTS), which limits an agent's ability to find "similar" past experiences or skills based on meaning.
- **Synchronous Operations:** Database and Vector Store operations are synchronous. While FastAPI handles them in thread pools, high-concurrency enterprise workloads benefit from fully async pipelines (e.g., `aiosqlite`).
- **Concurrency & Process Safety:** The `VectorStore` uses `threading.Lock`, which is safe for threads but **unsafe** for multiple processes (e.g., multiple uvicorn workers). This is a blocker for horizontal scaling.
- **Unused Dependencies:** `Redis` is defined in infrastructure but not actively used for caching or queuing.
- **ID Generation:** Using `time.time_ns()` for vector IDs is simple but risky for distributed systems.

## 3. Implementation Guide: Making it an "Agentic" Memory Service

Agents require more than just knowledge retrieval; they need **Experience Recall** and **Skill Lookup**.

### Step 1: Vectorize Episodic & Procedural Layers
**Why:** To allow agents to answer "Have I done something like this before?" (Episodic) or "Do I have a tool for this?" (Procedural).

**Changes Required:**
1.  **Schema Update:** Add `embedding_id` column to `memories_episodic` and `memories_procedural` tables.
2.  **Vector Store Refactor:** Support multiple indices to avoid pollution between layers.
    -   `data/vectors_semantic.index`
    -   `data/vectors_episodic.index`
    -   `data/vectors_procedural.index`
3.  **Service Logic:**
    -   Update `add_episodic`: Generate embedding from `content` + `goal` + `outcome`. Add to Episodic Index.
    -   Update `add_procedural`: Generate embedding from `skill_name` + `trigger`. Add to Procedural Index.

### Step 2: Unified "Agent Context" Retrieval
**Why:** Agents need a single call to get all relevant context.

**New API Endpoint:** `POST /retrieve_context`
-   **Input:** Current Task/Query.
-   **Logic:**
    -   Search Semantic Index -> Facts.
    -   Search Episodic Index -> Relevant Past Plans/Outcomes.
    -   Search Procedural Index -> Relevant Skills.
    -   Combine results into a structured prompt context.

## 4. Enterprise-Grade Requirements

### Reliability & Scalability
-   **Async Database:** Migrate to `aiosqlite` or `sqlalchemy[asyncio]`.
-   **Distributed Vector Database:** Abstract `VectorStore` to support external providers (e.g., Qdrant, Weaviate, pgvector) for scale beyond local files.
-   **Process Safety:** If sticking to local files, ensure only **one** writer process exists, or use a file-locking mechanism (e.g., `portalocker`) compatible with multi-process access.

### Observability
-   **Metrics:** Instrument operations with Prometheus (latency, cache hits, vector search duration).
-   **Tracing:** Add OpenTelemetry to trace requests across layers.

### Infrastructure
-   **Redis Caching:** Implement Redis caching for high-frequency queries (e.g., fetching procedural skills which change rarely).
-   **Rate Limiting:** Protect APIs using `redis` and `fastapi-limiter`.
