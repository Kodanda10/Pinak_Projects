# Pinak Memory Service: Architecture Analysis & Read-Only → Agent Memory Service Roadmap

## 📊 Current State Assessment

### ✅ WHAT'S GOOD

#### 1. **Solid Security Foundation**
- JWT-based authentication with tenant/project isolation (file-system level)
- Multi-tenant architecture properly segregated
- Hash-chained audit logs for tamper-evidence
- Dependency injection pattern for clean testing
- Tests validate isolation between tenants

#### 2. **Clean Architecture Layers**
```
API Layer (FastAPI endpoints)
    ↓
Service Layer (MemoryService, add_episodic, list_episodic, etc.)
    ↓
Storage Layer (JSONL + FAISS + Redis)
    ↓
File System (tenant/project/layer structure)
```

#### 3. **8-Layer Memory System** (mostly implemented)
- ✅ Semantic (vector + FAISS)
- ✅ Episodic (JSONL with salience)
- ✅ Procedural (JSONL with steps)
- ✅ RAG (JSONL with sources)
- ✅ Events (audit-chained JSONL)
- ✅ Session (TTL-aware JSONL)
- ✅ Working (expiring JSONL)
- ⚠️ Changelog (partially - via audit hashing)

#### 4. **Good Testing Structure**
- Async test fixtures
- Token generation helpers
- Isolation tests between tenants
- Audit log verification tests

#### 5. **Sensible Defaults**
- Deterministic dummy encoder for testing (no external downloads)
- Redis optional (graceful fallback)
- Config-driven setup
- Docker-ready

---

### ❌ WHAT'S BAD (for Scaling to Agent Memory)

#### 1. **Write Operations are Limited**
- **Current**: Only adds (POST), searches (GET), and lists exist
- **Missing for agents**: 
  - Update/patch operations (e.g., refine memory, update importance)
  - Delete operations (e.g., forget stale memories)
  - Batch operations (write 100s of memories at once)
  - Conditional writes (write only if memory doesn't exist)

#### 2. **No Concurrency Control**
- In-memory vector store caching with simple dict keying `(tenant, project_id)`
- **Problem**: If two agents write simultaneously to same tenant/project:
  - Race condition on FAISS index (not thread-safe)
  - Metadata JSON overwrites
  - Potential data loss

#### 3. **Search is Basic**
- Semantic search only works on top-level `memory.add()` (semantic layer)
- **Missing**:
  - Keyword search (BM25) across layers
  - Tag-based filtering before vector search
  - Time-range filtering on semantic results
  - Hybrid search (semantic + keyword)
  - Composite queries ("async" in episodic + after:2025-01-01)

#### 4. **No Observability**
- Zero logging of API calls, search patterns, or performance
- No metrics on memory density, retrieval latency
- Makes it hard to debug or optimize for agents

#### 5. **Storage Inefficiency**
- JSONL files are read in full for list operations
- No indexing on episodic/procedural/RAG (just linear scan)
- Metadata JSON can grow unbounded
- No pagination on JSONL reads (memory bloat)

#### 6. **Temporal Operations Weak**
- Expiration checking is manual in-app logic (not DB-level)
- No "retrieve memories from last 24h" with index
- Session TTL is fragile (stores expiry time, requires app logic)

#### 7. **No Conflict Resolution**
- Duplicate memory check doesn't exist
- Same content written twice = two separate vectors
- Vector store and metadata can drift (FAISS + JSON mismatch)

#### 8. **Limited Error Handling**
- Silent failures in JSONL parsing (bare `except: pass`)
- No validation of embedding dimension mismatches
- No rollback if partial writes fail

---

## 🎯 Roadmap: Read-Only Service → Full Agent Memory Service

### **Phase 1: Make It Write-Safe (2-3 days)**

#### Goal
Enable safe concurrent writes from multiple agents without data loss.

#### Changes

**1. Add File Locking**
```python
# app/core/locking.py
import fcntl
import contextlib

class FileLock:
    def __init__(self, path: str, timeout: int = 30):
        self.path = path
        self.timeout = timeout
    
    @contextlib.contextmanager
    def acquire(self):
        # Exclusive lock on vector store during write
        with open(self.path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**2. Atomic Vector Store Writes**
```python
# In memory_service.py

def _save_vector_store(self, tenant: str, project_id: str) -> None:
    store = self._vector_stores.get((tenant, project_id))
    if not store:
        return
    
    # Write to temp, then atomic move
    vector_path = store["vector_path"]
    temp_vector_path = f"{vector_path}.tmp"
    temp_meta_path = f"{store['metadata_path']}.tmp"
    
    faiss.write_index(store["index"], temp_vector_path)
    with open(temp_meta_path, "w") as f:
        json.dump(store["metadata"], f)
    
    # Atomic rename
    os.rename(temp_vector_path, vector_path)
    os.rename(temp_meta_path, store["metadata_path"])
```

**3. Add DELETE + UPDATE Endpoints**
```python
# Endpoints to add
POST /memory/delete/{memory_id}          # Remove from semantic store
POST /memory/update/{memory_id}          # Update content + re-embed
POST /episodic/delete/{salience_id}      # Remove from episodic
GET  /memory/{memory_id}                 # Retrieve single memory
```

**4. Add Batch Operations**
```python
POST /memory/batch/add          # [{"content": "...", "tags": [...]}, ...]
POST /memory/batch/search       # [{"query": "...", "k": 5}, ...]
```

---

### **Phase 2: Search + Query Layer (3-4 days)**

#### Goal
Enable rich, composable queries across all memory layers.

#### Changes

**1. Add Full-Text Search (BM25)**
```python
# app/services/bm25_index.py
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self, layer: str, tenant: str, project_id: str):
        self.layer = layer
        self.docs = []
        self.bm25 = None
    
    def index(self, doc_ids: List[str], texts: List[str]):
        tokenized = [text.split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = doc_ids
    
    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        tokens = query.split()
        scores = self.bm25.get_scores(tokens)
        top_k_idx = np.argsort(-scores)[:k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_k_idx]
```

**2. Hybrid Search (Semantic + BM25)**
```python
def search_hybrid(self, query: str, tenant: str, project_id: str, 
                  semantic_weight: float = 0.7, k: int = 10):
    """
    Combine semantic + BM25 scoring.
    semantic_weight: 0.7 means 70% semantic, 30% keyword
    """
    semantic_results = self.search_memory(query, tenant, project_id, k)
    keyword_results = self.bm25_search(query, tenant, project_id, k)
    
    # Merge by ID, weighted score
    merged = {}
    for r in semantic_results:
        merged[r['id']] = semantic_weight * (1 - r['distance'] / 100)
    for r in keyword_results:
        merged[r['id']] = merged.get(r['id'], 0) + (1-semantic_weight) * r['score']
    
    return sorted(merged.items(), key=lambda x: -x[1])[:k]
```

**3. Rich Query DSL**
```python
# New endpoint: POST /memory/search_advanced
{
    "query": "async handling",
    "layer_filters": {
        "episodic": {"salience_gte": 5},
        "semantic": {"tag": "python"},
        "procedural": {}
    },
    "time_range": {
        "start": "2025-01-01",
        "end": "2025-01-31"
    },
    "limit": 20,
    "mode": "hybrid"  # or "semantic", "keyword", "unified"
}
```

**4. Layer-Specific Indices**
```python
# For each layer, build quick-lookup indices
self.indices = {
    "episodic": {
        "by_salience": {},      # salience -> [records]
        "by_date": {},          # YYYY-MM-DD -> [records]
        "by_keyword": {}        # keyword -> [record_ids]
    },
    "procedural": {
        "by_skill_id": {},      # skill_id -> [records]
    },
    "rag": {
        "by_source": {},        # source -> [records]
    }
}
```

---

### **Phase 3: Observability + Performance (2-3 days)**

#### Goal
Debug and optimize memory service for production agent workloads.

#### Changes

**1. Structured Logging**
```python
# app/core/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "tenant": getattr(record, "tenant", "unknown"),
            "project_id": getattr(record, "project_id", "unknown"),
        })

# Usage
logger.info("search_executed", extra={
    "tenant": ctx.tenant_id,
    "project_id": ctx.project_id,
    "query": query,
    "k": k,
    "latency_ms": elapsed_ms,
    "result_count": len(results),
})
```

**2. Prometheus Metrics**
```python
# app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge

memory_adds = Counter("memory_adds_total", "Total memories added", ["tenant"])
search_latency = Histogram("search_latency_ms", "Search latency", ["layer"])
vector_store_size = Gauge("vector_store_size", "FAISS index size", ["tenant", "project"])
```

**3. Health Check Endpoint**
```python
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "redis_connected": memory_service.redis_client is not None,
        "vector_stores_loaded": len(memory_service._vector_stores),
        "version": "0.1.0",
    }
```

---

### **Phase 4: Persistence + Scale (4-5 days)**

#### Goal
Replace JSONL + in-memory caching with proper database.

#### Changes

**1. Move to PostgreSQL + pgvector**
```sql
-- Storage becomes:
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    layer TEXT NOT NULL,  -- 'semantic', 'episodic', etc.
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    INDEX (tenant_id, project_id, layer)
) PARTITION BY LIST (layer);

-- Separate tables for semantic/episodic/procedural/rag
-- JSONL moves to indexes, not stored
```

**2. Background Expiration Cleanup**
```python
# app/background/cleanup.py
import asyncio

async def expire_memories():
    """Run every 5 minutes: delete expired entries"""
    while True:
        await asyncio.sleep(300)
        deleted = await db.query("""
            DELETE FROM memories 
            WHERE expires_at < NOW()
        """)
        logger.info(f"Expired {deleted} memories")
```

**3. Connection Pooling**
```python
# app/core/db.py
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
)
```

---

## 🔌 How Agents Would Use This

### Example: AI Agent Learning Loop

```python
# Agent code
from pinak_client import PinakMemoryClient

client = PinakMemoryClient(
    base_url="http://memory-service:8001",
    token=jwt_token,  # tenant + project_id baked in
)

# Step 1: Recall relevant context
context = await client.search(
    query="user preferences for async code",
    layers=["episodic", "procedural"],
    mode="hybrid",
    limit=10,
)

# Step 2: Do work (make API call, process data, etc.)
# ...

# Step 3: Store learnings
await client.add_memory(
    content="User prefers Promise-based async over callbacks",
    layer="episodic",
    salience=8,
    tags=["user-pref", "async"],
)

# Step 4: Update procedural memory if learned a new skill
await client.add_procedural(
    skill_id="async-promise-patterns",
    steps=["Use Promise.all for parallel", "Use async/await for sequences"],
)

# Step 5: Purge old working memory from today
await client.delete_working_memory(
    before="2025-01-30T23:59:59Z",  # Delete today's old notes
)

# Step 6: Monitor what we know
stats = await client.get_memory_stats()
print(f"Semantic store: {stats['semantic']['count']} memories")
print(f"Episodic salience avg: {stats['episodic']['avg_salience']}")
```

---

## 📋 Implementation Checklist

### Phase 1: Write Safety
- [ ] Add file locking around FAISS writes
- [ ] Implement atomic rename pattern
- [ ] Add DELETE /memory/{id} endpoint
- [ ] Add UPDATE /memory/{id} endpoint
- [ ] Add BATCH /memory/batch/add endpoint
- [ ] Write stress tests (100 concurrent writes)
- [ ] Verify no data loss under contention

### Phase 2: Search Layer
- [ ] Implement BM25 indexing for episodic
- [ ] Add hybrid search endpoint
- [ ] Create query DSL parser
- [ ] Add layer-specific filters
- [ ] Implement time-range filtering
- [ ] Write search benchmarks

### Phase 3: Observability
- [ ] Set up JSON logging
- [ ] Add Prometheus metrics
- [ ] Create Grafana dashboard
- [ ] Add request tracing (OpenTelemetry)
- [ ] Health check endpoint
- [ ] SLO targets (p99 search < 100ms)

### Phase 4: Database
- [ ] Migrate to PostgreSQL
- [ ] Set up pgvector extension
- [ ] Create migration scripts
- [ ] Implement connection pooling
- [ ] Background expiration task
- [ ] Performance tuning (indexes, partitioning)

---

## 🚀 Quick Start for Agent Integration (Today)

Even without Phase 1-4, you can start using this for agents **if you accept these constraints**:

```python
# Constraints:
# 1. Single agent per tenant/project (no concurrency)
# 2. Search only semantic layer
# 3. No updates/deletes
# 4. Manual TTL cleanup
# 5. Memory loss on service restart (in-memory FAISS cache)

# But you get:
# ✅ Secure multi-tenant isolation
# ✅ Vector semantic search
# ✅ Episodic + procedural + RAG storage
# ✅ Tamper-evident audit logs
# ✅ Working + session memory with TTL

# Usage for agent:
async def agent_loop():
    for task in tasks:
        # Recall context
        memories = await client.search(query=task.description, k=5)
        
        # Execute
        result = await execute_task(task, memories)
        
        # Learn
        await client.add_episodic(
            content=f"Task {task.id}: {result.summary}",
            salience=result.success_score,
        )
```

---

## 📊 Priority Matrix

| Phase | Time | Value | Criticality | Start When |
|-------|------|-------|-------------|-----------|
| 1: Write Safety | 2-3d | High | **CRITICAL** | Now (if >1 agent) |
| 2: Search Layer | 3-4d | High | Important | After Phase 1 |
| 3: Observability | 2-3d | Medium | Important | After Phase 2 |
| 4: Database | 4-5d | High | Post-MVP | After Phase 3 |

**Recommendation**: Do Phase 1 immediately if agents will write concurrently. Skip Phase 4 until you hit 1M+ memories.

---

## Summary Table

| Aspect | Current | Target (Phases 1-4) |
|--------|---------|-------------------|
| Write Ops | Add only | Add + Update + Delete + Batch |
| Concurrency | ❌ Unsafe | ✅ Thread-safe + file locking |
| Search | Semantic only | Hybrid + DSL + filters |
| Storage | JSONL + in-memory | PostgreSQL + pgvector |
| Observability | None | Logs + Metrics + Tracing |
| Agent Support | Single-tenant | Multi-agent multi-tenant |
| Max Memories | ~100K (memory limit) | Unlimited (DB) |
| Query Latency | ~50ms (single) | <100ms p99 (optimized) |
