# PR #15 Review: "Refactor Memory Service to Enterprise Grade (SQLite + Hybrid Search)"

**Status**: 🟡 **NEEDS FIXES** (Good direction, critical issues to resolve)

---

## 📋 Executive Summary

PR #15 is a **major refactor** that transforms the memory service from JSONL + in-memory FAISS caching to **SQLite + hybrid search**. This directly addresses Phase 1 & 2 from the architecture analysis. The approach is sound, but there are **critical issues blocking merge**:

1. ❌ **Concurrency bugs** in FAISS wrapper (race conditions)
2. ⚠️ **Incomplete hybrid search** (RRF not properly weighted)
3. ⚠️ **Missing delete/cleanup** for expired memories
4. ⚠️ **Test coverage too low** (~50% of new code)
5. ⚠️ **Database schema concerns** (denormalized design, FTS triggers may be fragile)

---

## ✅ WHAT'S GOOD

### 1. **SQLite + FTS5 is Solid Choice**
- ✅ WAL mode enabled (fixes concurrency issues from JSONL)
- ✅ FTS5 for full-text search (keyword + semantic hybrid)
- ✅ Proper schema per layer (semantic, episodic, procedural, RAG, events, session, working)
- ✅ Multi-tenant support maintained (tenant + project_id on all tables)

**Code:**
```python
# From database.py
conn.execute("PRAGMA journal_mode=WAL;")  # WAL = concurrent reads + writes
conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_semantic_fts
    USING fts5(content, ...)
""")
```

### 2. **VectorStore Wrapper Adds Thread Safety**
- ✅ Lock around FAISS operations (`self.lock = threading.Lock()`)
- ✅ Auto-load on init, explicit save (good pattern)
- ✅ IndexIDMap for mapping FAISS IDs to DB rows

**Code:**
```python
class VectorStore:
    def __init__(self, index_path: str, dimension: int):
        self.lock = threading.Lock()  # Thread safety
        self._load_index()
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        with self.lock:  # Exclusive lock during add
            self.index.add_with_ids(vectors, id_array)
```

### 3. **Hybrid Search (RRF) Attempts Multi-Modal Retrieval**
- ✅ Reciprocal Rank Fusion to combine FTS + semantic scores
- ✅ New `/retrieve_context` endpoint for unified queries
- ✅ Layer filtering (can query episodic+procedural together)

### 4. **Agent-Ready Memory Structures**
- ✅ `EpisodicCreate` now has `goal`, `plan`, `outcome`, `tool_logs` (agent loop tracking)
- ✅ `ContextResponse` returns multi-layer results in one call
- ✅ New endpoints for delete operations

### 5. **CLI + TUI for Local Development**
- ✅ `cli/main.py` and `cli/tui.py` for interactive testing
- ✅ Useful for debugging without API calls

---

## ❌ CRITICAL ISSUES (Must Fix)

### 1. **Race Condition in FAISS + SQLite Sync**

**Problem**: Vector store and database can drift after crashes.

```python
# Current code (WRONG):
def add_memory(self, memory_data, tenant, project_id):
    # Step 1: Add to DB (committed)
    memory_id = db.insert(...)
    
    # Step 2: If crash here, FAISS never gets the vector
    vector_store.add_vectors(embedding, [memory_id])
    
    # Step 3: If crash here, DB has memory but FAISS is missing it
    vector_store.save()
```

**Fix Required**:
```python
def add_memory(self, memory_data, tenant, project_id):
    # Step 1: Write to DB (SQLite auto-commits with save)
    with db.transaction():  # ACID wrapper
        memory_id = db.insert(...)
        embedding = model.encode(memory_data.content)
    
    # Step 2: Add to FAISS (this is in-memory, tolerable to lose on crash)
    vector_store.add_vectors(embedding, [memory_id])
    vector_store.save()  # Explicit save to disk
    
    # If crash between Step 1 and 2, we lose the vector but DB is clean
    # On restart, rebuild FAISS from DB
```

**Severity**: 🔴 **HIGH** — Data loss risk

---

### 2. **VectorStore.save() Not Called Automatically**

**Problem**: Comments in code suggest uncertainty:
```python
# Or maybe just periodically. Let's do explicit save calls from Service.
# Auto-save on modification? Or let caller handle it?
```

**Current State**: 
- `add_vectors()` does NOT auto-save
- Caller must remember to call `.save()`
- If forgotten, FAISS persists only to memory (loss on restart)

**Fix Required**:
```python
# Option A: Auto-save with debounce (fast enough)
def add_vectors(self, vectors, ids):
    with self.lock:
        self.index.add_with_ids(vectors, ids_array)
        self._schedule_save()  # Debounce to avoid 1000 saves/sec

def _schedule_save(self):
    if not hasattr(self, '_save_timer') or not self._save_timer.is_alive():
        self._save_timer = threading.Timer(5.0, self.save)  # Save in 5s
        self._save_timer.start()
```

**OR**

```python
# Option B: Make it explicit with context manager
@contextmanager
def batch_add(self):
    """For batch operations, bundle saves."""
    yield
    self.save()  # Save once at end
```

**Severity**: 🔴 **HIGH** — Silent data loss on service restart

---

### 3. **Hybrid Search Weighting is Wrong**

**Problem**: RRF (Reciprocal Rank Fusion) is equal-weight:

```python
# Current code (endpoints.py)
def retrieve_context(query, layers, semantic_weight=0.5):
    fts_results = db.fts_search(query, layer)
    vector_results = vector_store.search(embedding)
    
    # Merge scores — but RRF is not weighted!
    # RRF = 1/(k + rank1) + 1/(k + rank2)
    # This gives equal weight to FTS and vectors
```

**Problem**: `semantic_weight` parameter is accepted but **NOT USED**.

**Fix Required**:
```python
def retrieve_context(query, layers, semantic_weight=0.7):
    """
    semantic_weight: 0.0 = pure keyword, 1.0 = pure semantic
    """
    fts_results = db.fts_search(query, layer, k=50)  # Get more to merge
    vector_results = vector_store.search(embedding, k=50)
    
    # Normalize scores to [0, 1]
    fts_scores = {r['id']: 1.0 - (i / len(fts_results)) for i, r in enumerate(fts_results)}
    vector_scores = {r['id']: 1.0 - (r['distance'] / 100) for r in vector_results}
    
    # Weighted fusion
    merged = {}
    for mem_id in set(fts_scores.keys()) | set(vector_scores.keys()):
        merged[mem_id] = (
            semantic_weight * vector_scores.get(mem_id, 0) +
            (1 - semantic_weight) * fts_scores.get(mem_id, 0)
        )
    
    return sorted(merged.items(), key=lambda x: -x[1])[:k]
```

**Severity**: 🟡 **MEDIUM** — Feature doesn't work as intended

---

### 4. **No Expiration Cleanup Job**

**Problem**: Session/working memories with TTL are stored but **never deleted**. Code checks expiry at query time:

```python
# In endpoints.py
if exp:
    try:
        if datetime.datetime.fromisoformat(exp) < datetime.datetime.utcnow():
            continue  # Skip expired, but never delete
    except Exception:
        pass
```

**Result**: DB grows unbounded with dead records.

**Fix Required**:
```python
# background_tasks.py (new file)
import asyncio
from datetime import datetime, timedelta

async def cleanup_expired_memories(db, interval_seconds=3600):
    """Run every hour: delete expired session/working memories."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            now = datetime.utcnow().isoformat()
            
            deleted_sessions = db.execute("""
                DELETE FROM memories_session 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (now,))
            
            deleted_working = db.execute("""
                DELETE FROM memories_working 
                WHERE expires_at IS NOT NULL AND expires_at < ?
            """, (now,))
            
            logger.info(f"Expired: {deleted_sessions + deleted_working} memories")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

# In main.py
@app.on_event("startup")
async def startup():
    asyncio.create_task(cleanup_expired_memories(db))
```

**Severity**: 🟡 **MEDIUM** — Operational concern (disk space)

---

### 5. **Database Triggers for FTS Are Fragile**

**Problem**: FTS5 triggers can become inconsistent if:
- INSERT fails mid-transaction (trigger fires, table doesn't)
- Manual SQL bypasses triggers
- DELETE on FTS virtual table fails silently

**Current Code**:
```python
CREATE TRIGGER IF NOT EXISTS memories_semantic_ai AFTER INSERT ON memories_semantic BEGIN
  INSERT INTO memories_semantic_fts(rowid, content) VALUES (new.rowid, new.content);
END;
```

**Risk**: FTS and table can drift → search returns deleted records.

**Better Approach**:
```python
# Use app-level FTS sync instead of triggers
def add_memory(self, content, tenant, project_id):
    with db.transaction():  # ACID
        memory_id = db.insert("memories_semantic", {
            "id": uuid.uuid4(),
            "content": content,
            "tenant": tenant,
            "project_id": project_id,
            "created_at": datetime.utcnow().isoformat(),
        })
    
    # Sync FTS after commit succeeds
    db.execute("""
        INSERT INTO memories_semantic_fts(rowid, content)
        SELECT rowid, content FROM memories_semantic WHERE id = ?
    """, (memory_id,))
```

**Severity**: 🟡 **MEDIUM** — Edge case, but trust in search quality

---

### 6. **Tests Incomplete**

**Current Tests**: 
- `test_memory_api.py`: 53 lines (covers basic add/search)
- `test_updates.py`: 53 lines (covers update/delete)
- **Missing**:
  - Concurrent write tests (race conditions)
  - Large batch tests (performance)
  - Expiration tests
  - Hybrid search weighting tests
  - Vector sync tests (FAISS ↔ DB consistency)

**Coverage Estimate**: ~40% of new code

**Fix Required**:
```python
# test_concurrency.py (new file)
@pytest.mark.asyncio
async def test_concurrent_adds_no_data_loss():
    """100 concurrent writes, all should succeed without duplicates."""
    tasks = [
        add_memory(f"content_{i}", tenant="test")
        for i in range(100)
    ]
    results = await asyncio.gather(*tasks)
    
    # Verify DB count
    db_count = db.execute("SELECT COUNT(*) FROM memories_semantic WHERE tenant=?", ("test",))
    assert db_count == 100, "Data loss detected"
    
    # Verify FAISS count
    vector_count = vector_store.index.ntotal
    assert vector_count == 100, "Vector store out of sync"

@pytest.mark.asyncio
async def test_vector_store_faiss_sync():
    """Add memory, kill FAISS, restart, verify consistency."""
    mem_id = add_memory("test content", tenant="test")
    
    # Kill FAISS (don't save)
    vector_store.index = None
    
    # Restart (rebuild from DB)
    vector_store._rebuild_from_db()
    
    # Search should work
    results = vector_store.search(query_vec, k=1)
    assert results[0]['id'] == mem_id
```

**Severity**: 🟡 **MEDIUM** — Can't merge without test coverage

---

## ⚠️ MODERATE ISSUES (Should Fix)

### 7. **No Rollback on Delete (Soft Delete Alternative)**

If a user deletes a memory by mistake, it's gone forever. Consider soft deletes:

```python
# Schema change
CREATE TABLE memories_semantic (
    id TEXT PRIMARY KEY,
    ...
    deleted_at TEXT NULL,  -- NULL = active, timestamp = deleted
);

# Delete operation becomes:
def delete_memory(self, memory_id):
    db.execute("""
        UPDATE memories_semantic SET deleted_at = ? WHERE id = ?
    """, (datetime.utcnow().isoformat(), memory_id))
    
    # Hide from search
    db.execute("""
        UPDATE memories_semantic_fts SET content = NULL 
        WHERE rowid = (SELECT rowid FROM memories_semantic WHERE id = ?)
    """, (memory_id,))

# Search should filter
WHERE deleted_at IS NULL
```

**Severity**: 🟢 **LOW** — Can be added in Phase 2

---

### 8. **No Schema Versioning**

If we need to add columns later (e.g., `importance` score), there's no migration system.

**Fix**: Add `schema_version` check:
```python
def _init_db(self):
    version = self._get_schema_version()  # Check DB version
    if version < 2:
        self._migrate_v1_to_v2()  # Run migration
        self._set_schema_version(2)
```

**Severity**: 🟢 **LOW** — Can defer to Phase 3

---

### 9. **Missing Logging in Critical Paths**

API endpoints have no request/response logging. Hard to debug in production.

```python
# Add structured logging
@router.post("/add")
def add_memory(...):
    logger.info("add_memory", extra={
        "tenant": ctx.tenant_id,
        "project_id": ctx.project_id,
        "content_length": len(memory.content),
    })
    result = service.add_memory(...)
    logger.info("add_memory_success", extra={"memory_id": result.id})
    return result
```

**Severity**: 🟢 **LOW** — Can add after merge

---

## 📊 Summary Table

| Issue | Severity | Type | Fix Effort |
|-------|----------|------|-----------|
| FAISS ↔ DB sync race | 🔴 HIGH | Bug | 4 hours |
| Missing auto-save | 🔴 HIGH | Bug | 2 hours |
| Hybrid search unweighted | 🟡 MEDIUM | Feature | 1 hour |
| No expiration cleanup | 🟡 MEDIUM | Ops | 2 hours |
| FTS triggers fragile | 🟡 MEDIUM | Risk | 3 hours |
| Incomplete tests | 🟡 MEDIUM | QA | 6 hours |
| No soft deletes | 🟢 LOW | UX | 2 hours |
| No schema versioning | 🟢 LOW | DevOps | 2 hours |
| Missing logging | 🟢 LOW | Ops | 1 hour |

**Total Fix Effort**: ~23 hours

---

## ✨ RECOMMENDATIONS

### **Merge Strategy**

**Option A: Fix Then Merge (Recommended)**
1. ✅ Fix race conditions (#1, #2)
2. ✅ Fix hybrid search (#3)
3. ✅ Add expiration cleanup (#4)
4. ✅ Add concurrency tests
5. Merge → Main

**Timeline**: ~2-3 days

---

### **Option B: Merge as WIP, Fix in Phase 2**
1. Merge with label `[WIP]`
2. Document known issues
3. Fix in separate PRs
4. Risk: Bugs in production

---

## 🎯 Detailed Fix Checklist

```markdown
# Pre-Merge Checklist

## Critical Fixes Required
- [ ] Implement FAISS←→DB sync recovery on startup
  - Rebuild FAISS from DB if mismatch detected
  - Log reconciliation events
  - Add test case
  
- [ ] Add explicit save protocol with debounce
  - Auto-save every 5 seconds during batch ops
  - Add context manager: `with vector_store.batch_add(): ...`
  - Document in code
  
- [ ] Fix hybrid search weighting
  - Implement semantic_weight parameter
  - Add unit tests for RRF scoring
  - Document formula in docstring
  
- [ ] Add expiration cleanup job
  - Background task to delete expired records
  - Configurable interval (default 1h)
  - Startup hook in main.py

## Test Coverage
- [ ] 100 concurrent add operations
- [ ] Vector store consistency after failure
- [ ] Hybrid search scoring tests (weights: 0.0, 0.5, 1.0)
- [ ] Expiration behavior (add, wait, query, verify not found)
- [ ] Delete cascade behavior

## Documentation
- [ ] Add troubleshooting guide (FTS desync, etc.)
- [ ] Document backup/restore procedure
- [ ] Add performance tuning notes (WAL, triggers)
```

---

## 💬 Questions for Author

1. **Why SQLite instead of PostgreSQL?**
   - ✅ Good for local-first MVP
   - ⚠️ Scaling to 10M+ memories is hard without PostgreSQL
   - Plan: Phase 4 migration?

2. **How will FAISS sync recovery work on prod?**
   - Current code doesn't handle it
   - Suggest: Background task to verify DB ↔ FAISS match on startup

3. **Why manual FTS triggers instead of app-level sync?**
   - Current: Triggers (fragile, trust SQLite)
   - Alternative: App-level after each commit (more control)

---

## 🚀 Post-Merge Next Steps

1. **Phase 2a**: Add PostgreSQL compatibility layer (same API, different backend)
2. **Phase 2b**: Hybrid search optimization (BM25 preprocessing, caching)
3. **Phase 3**: Observability (metrics, logs, tracing)
4. **Phase 4**: Scale to distributed (Elasticsearch, Pinecone)

---

## Final Verdict

### 🟡 **REQUEST CHANGES**

This PR is a **strong architectural improvement** but has **critical bugs that must be fixed**:

1. Data loss risk (FAISS ↔ DB sync)
2. Silent failures (auto-save not working)
3. Incomplete features (hybrid search)

**Recommendation**: 
- ✅ Merge direction and design: APPROVED
- ❌ Current implementation: NEEDS FIXES
- 📅 Target merge: After addressing critical issues (~2-3 days of work)

**Effort to fix**: ~23 hours total
**Effort to review**: ~4 hours (already done 👍)

---

## Code Quality Notes

### What's Well Done
- ✅ Clean separation of concerns (DB, VectorStore, Service)
- ✅ Proper type hints throughout
- ✅ Good schema design (normalized, indexed)
- ✅ Pydantic models for validation

### What Needs Polish
- ⚠️ Comments instead of docstrings (update for clarity)
- ⚠️ Some exception handling is too broad (`except: pass`)
- ⚠️ No connection pooling configured
- ⚠️ Logging is inconsistent

---

*Review completed with ❤️ by Amp (Rush Mode)*
