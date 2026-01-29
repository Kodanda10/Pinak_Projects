# PR #15 Quick Review Summary

**Status**: 🟡 **REQUEST CHANGES** — Good direction, critical bugs block merge

---

## 🎯 What This PR Does

Refactors memory service from **JSONL + in-memory FAISS** → **SQLite + thread-safe FAISS wrapper**

### Changes
- ✅ 1,653 lines added, 586 removed
- ✅ New `DatabaseManager` class (431 lines) — SQLite with FTS5
- ✅ New `VectorStore` wrapper (105 lines) — Thread-safe FAISS with locking
- ✅ Hybrid search (RRF) combining FTS + semantic
- ✅ Updated schemas for agent memory structures (goal, plan, outcome, tool_logs)
- ✅ CLI + TUI for interactive testing
- ✅ New delete endpoints + update endpoints

---

## ✅ WHAT'S GOOD

| Feature | Status | Notes |
|---------|--------|-------|
| SQLite + WAL mode | ✅ | Fixes JSONL concurrency issues |
| Thread-safe FAISS | ✅ | Lock-based synchronization |
| FTS5 indexing | ✅ | Full-text search for keyword matching |
| Multi-tenant support | ✅ | Preserved from original design |
| Hybrid search endpoint | ✅ | Combines FTS + vector search |
| Agent memory structures | ✅ | Plan, outcome, tool_logs fields |
| Schema per layer | ✅ | semantic, episodic, procedural, rag, events, session, working |

---

## ❌ CRITICAL ISSUES (Must Fix Before Merge)

### 1. **FAISS ↔ DB Sync Race Condition** 🔴 HIGH
```python
# WRONG: DB write succeeds, FAISS write fails mid-flight
# Result: Memory in DB but not searchable (data loss)
db.insert(...)  # ← Success
vector_store.add_vectors(...)  # ← Crash here?
vector_store.save()  # ← Never runs
```

**Fix**: Implement recovery on startup
- Detect FAISS ↔ DB mismatch
- Rebuild FAISS index from DB if needed
- Add test case for crash recovery

**Effort**: 4 hours

---

### 2. **Missing Auto-Save on Vector Operations** 🔴 HIGH
```python
# Current code comment:
# "Or maybe just periodically. Let's do explicit save calls from Service."

# Problem: If save() is forgotten, FAISS only in memory
# Crash = vector loss (in-memory only)
```

**Fix**: Add auto-save with debounce
```python
def add_vectors(self, vectors, ids):
    with self.lock:
        self.index.add_with_ids(vectors, ids_array)
        self._schedule_save()  # Save in 5 seconds (debounced)
```

**Effort**: 2 hours

---

### 3. **Hybrid Search Weight Parameter Unused** 🟡 MEDIUM
```python
def retrieve_context(query, layers, semantic_weight=0.7):
    # semantic_weight is ACCEPTED but NOT USED!
    # Both FTS and vectors get equal weight
    # Feature broken
```

**Fix**: Implement weighted RRF scoring
```python
merged[mem_id] = (
    semantic_weight * vector_score +
    (1 - semantic_weight) * fts_score
)
```

**Effort**: 1 hour

---

### 4. **No Expiration Cleanup Job** 🟡 MEDIUM
Session/working memories with `expires_at` timestamp are never deleted.
- DB grows unbounded with dead records
- Soft queries skip expired records (slow, no cleanup)

**Fix**: Background task to delete expired records
```python
async def cleanup_expired_memories(db):
    while True:
        await asyncio.sleep(3600)  # Every hour
        db.delete("memories_session", where="expires_at < NOW()")
        db.delete("memories_working", where="expires_at < NOW()")
```

**Effort**: 2 hours

---

### 5. **FTS Triggers Are Fragile** 🟡 MEDIUM
- Triggers can become inconsistent if INSERT/DELETE fails mid-transaction
- Manual SQL bypass bypasses triggers
- No consistency checking

**Better approach**: Sync FTS in app after commit (not triggers)

**Effort**: 3 hours

---

### 6. **Test Coverage Too Low** 🟡 MEDIUM
- Current: ~53 lines in test_memory_api.py, ~53 in test_updates.py
- **Missing**:
  - 100 concurrent writes (race condition tests)
  - Vector store crash recovery
  - Hybrid search weighting validation
  - Expiration behavior
  - DB ↔ FAISS consistency

**Coverage**: ~40% of new code

**Effort**: 6 hours

---

## 🟢 MODERATE ISSUES (Nice to Have)

- No soft deletes (can't undelete)
- No schema versioning (hard to migrate later)
- Missing structured logging in API endpoints
- No connection pooling configured
- Some exception handling too broad (`except: pass`)

---

## 📊 Fix Effort Breakdown

| Issue | Hours | Priority |
|-------|-------|----------|
| FAISS ↔ DB sync | 4 | **CRITICAL** |
| Auto-save | 2 | **CRITICAL** |
| Hybrid search weight | 1 | MEDIUM |
| Expiration cleanup | 2 | MEDIUM |
| FTS triggers | 3 | MEDIUM |
| Tests | 6 | MEDIUM |
| **Total** | **~18 hours** | — |

---

## 🎯 Recommendation

### **Merge Strategy**: Fix Then Merge (2-3 days)

1. **Day 1**: Fix critical issues (#1, #2)
2. **Day 2**: Fix hybrid search (#3), add tests
3. **Day 3**: Fix expiration (#4), cleanup code

### **OR**: Merge as [WIP], fix in Phase 2
- Risk: Bugs in production

---

## 📝 Full Review Document

See `PR_15_REVIEW.md` for detailed analysis:
- Code examples
- Line-by-line fixes
- Design questions
- Post-merge roadmap

---

## ✨ What Comes After

Once merged (with fixes):

**Phase 2**: Advanced search
- BM25 preprocessing
- Query DSL parser
- Layer-specific indexes

**Phase 3**: Observability
- JSON logging
- Prometheus metrics
- Request tracing

**Phase 4**: Scale
- PostgreSQL migration
- Distributed deployment
- Federated learning

---

## 💬 Questions for Team

1. **Why not PostgreSQL now?** → Good: local-first MVP. Plan Phase 4 migration?
2. **How to handle FAISS sync on prod?** → Need explicit recovery job
3. **Auto-save or explicit calls?** → Recommend debounced auto-save for safety
4. **When to add soft deletes?** → Phase 2 (nice to have, not critical)

---

**Verdict**: 🟡 Strong foundation, needs ~18 hours of fixes, then ✅ LGTM

*Review completed with ❤️ by Amp*
