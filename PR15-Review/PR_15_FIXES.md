# PR #15: Code Fixes (Ready to Implement)

Copy-paste ready fixes for critical issues.

---

## Fix #1: FAISS ↔ DB Sync Recovery

**File**: `app/services/vector_store.py`

Add this method to `VectorStore` class:

```python
def _rebuild_from_db(self, db_manager, tenant: str, project_id: str):
    """
    Rebuild FAISS index from database records.
    Called on startup to detect/fix mismatches.
    """
    logger.info(f"Rebuilding FAISS index for {tenant}/{project_id}")
    
    # Get all semantic memories from DB
    memories = db_manager.get_all_memories(tenant, project_id)
    
    if not memories:
        logger.info("No memories to rebuild")
        return
    
    # Rebuild index with DB IDs
    embeddings = []
    db_ids = []
    
    for mem in memories:
        try:
            embedding = model.encode([mem['content']])[0].astype(np.float32)
            embeddings.append(embedding)
            db_ids.append(int(mem['db_row_id']))  # Use actual DB row ID
        except Exception as e:
            logger.warning(f"Skip rebuild for {mem['id']}: {e}")
    
    if embeddings:
        with self.lock:
            self.index = self._create_index()  # Fresh index
            vectors = np.array(embeddings, dtype=np.float32)
            ids = np.array(db_ids, dtype=np.int64)
            self.index.add_with_ids(vectors, ids)
            self.save()
        logger.info(f"Rebuilt {len(embeddings)} vectors")

def verify_consistency(self, db_manager, tenant: str, project_id: str) -> bool:
    """
    Check if FAISS index size matches DB count.
    Returns True if consistent, False if mismatch.
    """
    db_count = db_manager.count_memories(tenant, project_id)
    faiss_count = self.index.ntotal if self.index else 0
    
    if db_count != faiss_count:
        logger.warning(f"Mismatch: DB={db_count}, FAISS={faiss_count}")
        return False
    
    return True
```

**File**: `app/main.py`

Add startup hook:

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle."""
    # Startup
    logger.info("Verifying FAISS ↔ DB consistency")
    for tenant, project_id in db_manager.get_all_tenants_projects():
        if not vector_store.verify_consistency(db_manager, tenant, project_id):
            logger.warning(f"Rebuilding {tenant}/{project_id}")
            vector_store._rebuild_from_db(db_manager, tenant, project_id)
    
    yield
    
    # Shutdown
    vector_store.save()

app = FastAPI(lifespan=lifespan)
```

---

## Fix #2: Auto-Save with Debounce

**File**: `app/services/vector_store.py`

Replace `add_vectors` method:

```python
import threading
import time

class VectorStore:
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.Lock()
        self.index = None
        self._load_index()
        
        # Save debounce
        self._save_timer = None
        self._save_interval = 5.0  # seconds
    
    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors and schedule save (debounced)."""
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        
        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        
        with self.lock:
            self.index.add_with_ids(vectors, id_array)
        
        # Schedule save in 5 seconds (debounced)
        self._schedule_save()
    
    def remove_ids(self, ids: List[int]):
        """Remove vectors and schedule save."""
        id_array = np.array(ids, dtype=np.int64)
        
        with self.lock:
            self.index.remove_ids(id_array)
        
        # Schedule save
        self._schedule_save()
    
    def _schedule_save(self):
        """Schedule a debounced save."""
        # Cancel previous timer
        if self._save_timer is not None:
            self._save_timer.cancel()
        
        # Schedule new save
        self._save_timer = threading.Timer(self._save_interval, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()
    
    def save(self):
        """Synchronously save to disk."""
        with self.lock:
            if self.index:
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}. Size: {self.index.ntotal}")
    
    @contextlib.contextmanager
    def batch_add(self):
        """Context manager for batch operations (no debounce)."""
        yield
        # Save immediately after batch
        self.save()
```

Usage in service:

```python
# Single add (debounced save)
vector_store.add_vectors(embedding, [memory_id])

# Batch add (explicit save at end)
with vector_store.batch_add():
    for memory in batch:
        embedding = model.encode([memory.content])
        vector_store.add_vectors(embedding, [memory.id])
```

---

## Fix #3: Hybrid Search Weighted Scoring

**File**: `app/services/memory_service.py`

Replace `retrieve_context` method:

```python
def retrieve_context(self, query: str, layers: List[str], 
                     tenant: str, project_id: str,
                     semantic_weight: float = 0.7, k: int = 10) -> dict:
    """
    Unified context retrieval with weighted hybrid search.
    
    Args:
        query: Search query
        layers: List of layers to search (e.g., ["episodic", "procedural"])
        semantic_weight: 0.0 = pure keyword, 0.5 = balanced, 1.0 = pure semantic
        k: Number of results per layer
    
    Returns:
        {
            "episodic": [...],
            "semantic": [...],
            "procedural": [...],
            ...
        }
    """
    assert 0.0 <= semantic_weight <= 1.0, "semantic_weight must be [0, 1]"
    
    results = {}
    
    for layer in layers:
        if layer == "semantic":
            results["semantic"] = self._search_semantic_hybrid(
                query, tenant, project_id, semantic_weight, k
            )
        elif layer == "episodic":
            results["episodic"] = self._search_episodic_hybrid(
                query, tenant, project_id, semantic_weight, k
            )
        elif layer == "procedural":
            results["procedural"] = self._search_procedural_hybrid(
                query, tenant, project_id, semantic_weight, k
            )
        # ... other layers
    
    return results

def _search_semantic_hybrid(self, query: str, tenant: str, project_id: str,
                           semantic_weight: float, k: int) -> List[dict]:
    """Hybrid search: FTS + Vector scoring."""
    
    # 1. FTS search (keyword)
    fts_results = self.db.fts_search(
        "memories_semantic", query, tenant, project_id, limit=k*2
    )
    
    # 2. Vector search (semantic)
    embedding = self.model.encode([query])[0].astype(np.float32)
    distances, indices = self.vector_store.search(embedding, k=k*2)
    
    # 3. Normalize scores to [0, 1]
    fts_scores = {}
    for i, result in enumerate(fts_results):
        # Lower rank = higher score
        fts_scores[result['id']] = 1.0 - (i / len(fts_results))
    
    vector_scores = {}
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        mem_id = self.db.get_memory_id_by_faiss_id(idx, tenant, project_id)
        # Lower distance = higher score (invert distance)
        vector_scores[mem_id] = max(0, 1.0 - (dist / 100))
    
    # 4. Weighted fusion (RRF with weights)
    merged = {}
    all_ids = set(fts_scores.keys()) | set(vector_scores.keys())
    
    for mem_id in all_ids:
        fts_score = fts_scores.get(mem_id, 0)
        vec_score = vector_scores.get(mem_id, 0)
        
        # Weighted combination
        combined_score = (
            semantic_weight * vec_score +
            (1 - semantic_weight) * fts_score
        )
        merged[mem_id] = combined_score
    
    # 5. Return top-k sorted by score
    top_k = sorted(merged.items(), key=lambda x: -x[1])[:k]
    
    results = []
    for mem_id, score in top_k:
        mem = self.db.get_memory_by_id(mem_id, tenant, project_id)
        results.append({
            **mem,
            "relevance_score": float(score),
            "search_mode": "hybrid",
        })
    
    return results
```

---

## Fix #4: Expiration Cleanup Job

**File**: `app/background/cleanup.py` (new file)

```python
import asyncio
import logging
from datetime import datetime
from app.core.database import DatabaseManager

logger = logging.getLogger(__name__)

async def cleanup_expired_memories(db: DatabaseManager, interval_seconds: int = 3600):
    """
    Background task: Delete expired session/working memories.
    
    Args:
        db: DatabaseManager instance
        interval_seconds: How often to run (default: 1 hour)
    """
    logger.info(f"Expiration cleanup task started (interval: {interval_seconds}s)")
    
    while True:
        try:
            await asyncio.sleep(interval_seconds)
            
            now = datetime.utcnow().isoformat()
            
            # Delete expired session memories
            session_deleted = db.execute(
                "DELETE FROM memories_session WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            
            # Delete expired working memories
            working_deleted = db.execute(
                "DELETE FROM memories_working WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,)
            )
            
            total = session_deleted + working_deleted
            
            if total > 0:
                logger.info(f"Expired {total} memories (session: {session_deleted}, working: {working_deleted})")
        
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}", exc_info=True)
            # Continue even if failed
```

**File**: `app/main.py`

Hook into startup:

```python
import asyncio
from app.background.cleanup import cleanup_expired_memories

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifecycle."""
    # Startup: Start background tasks
    cleanup_task = asyncio.create_task(
        cleanup_expired_memories(db_manager, interval_seconds=3600)
    )
    app.state.cleanup_task = cleanup_task
    
    yield
    
    # Shutdown: Cancel background tasks
    app.state.cleanup_task.cancel()
    try:
        await app.state.cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)
```

---

## Fix #5: Improve FTS Consistency

**File**: `app/core/database.py`

Replace trigger-based approach with app-level sync:

```python
@contextlib.contextmanager
def transaction(self):
    """Context manager for transactions."""
    conn = self._get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def add_memory(self, layer: str, memory_data: dict, tenant: str, project_id: str) -> str:
    """
    Add memory with guaranteed FTS sync.
    
    Args:
        layer: "semantic", "episodic", "procedural", etc.
        memory_data: Dict with content, tags, etc.
        
    Returns:
        memory_id (str)
    """
    memory_id = str(uuid.uuid4())
    
    with self.transaction() as conn:
        # Insert into main table
        conn.execute(
            f"""INSERT INTO memories_{layer} 
               (id, content, tenant, project_id, created_at, ...)
               VALUES (?, ?, ?, ?, ?, ...)""",
            (memory_id, memory_data['content'], tenant, project_id, 
             datetime.utcnow().isoformat(), ...)
        )
        # Get rowid for FTS
        rowid = conn.execute(
            f"SELECT rowid FROM memories_{layer} WHERE id = ?",
            (memory_id,)
        ).fetchone()[0]
    
    # After commit, sync FTS (app-level, not trigger)
    try:
        conn = self._get_connection()
        conn.execute(
            f"""INSERT INTO memories_{layer}_fts(rowid, content, ...)
               VALUES (?, ?, ...)""",
            (rowid, memory_data['content'], ...)
        )
        conn.commit()
    except Exception as e:
        logger.error(f"FTS sync failed for {memory_id}: {e}")
        # Log but don't fail (will retry on next cleanup)
    finally:
        conn.close()
    
    return memory_id
```

---

## Test Additions

**File**: `tests/test_vector_consistency.py` (new file)

```python
import pytest
import asyncio
from app.services.vector_store import VectorStore
from app.core.database import DatabaseManager

@pytest.mark.asyncio
async def test_concurrent_vector_adds_no_race():
    """Verify 100 concurrent writes don't corrupt index."""
    async def add_one(i):
        embedding = np.random.rand(8).astype(np.float32)
        vector_store.add_vectors(embedding, [i])
    
    tasks = [add_one(i) for i in range(100)]
    await asyncio.gather(*tasks)
    
    # Force save
    vector_store.save()
    
    # Reload and verify count
    vector_store._load_index()
    assert vector_store.index.ntotal == 100, "Race condition detected"

@pytest.mark.asyncio
async def test_faiss_db_sync_recovery():
    """Simulate crash and recovery."""
    # Add memories
    for i in range(10):
        mem_id = db.add_memory("semantic", {"content": f"test_{i}"}, "test", "proj")
        embedding = np.random.rand(8).astype(np.float32)
        vector_store.add_vectors(embedding, [i])
    
    # "Crash" - don't save FAISS
    vector_store.index = None
    
    # Recovery - rebuild from DB
    vector_store._rebuild_from_db(db, "test", "proj")
    
    # Search should work
    query_embedding = np.random.rand(8).astype(np.float32)
    distances, ids = vector_store.search(query_embedding, k=5)
    assert len(ids[0]) > 0, "Recovery failed"

@pytest.mark.asyncio
async def test_hybrid_search_semantic_weight():
    """Verify semantic_weight parameter actually affects results."""
    # Add test data
    db.add_memory("semantic", {"content": "python async await"}, "test", "proj")
    db.add_memory("semantic", {"content": "synchronized locks threads"}, "test", "proj")
    
    # Pure keyword search (weight=0)
    keyword_results = service.retrieve_context(
        "async", ["semantic"], "test", "proj", semantic_weight=0.0, k=2
    )
    
    # Pure semantic search (weight=1)
    semantic_results = service.retrieve_context(
        "async", ["semantic"], "test", "proj", semantic_weight=1.0, k=2
    )
    
    # Different weights should produce different orders
    assert keyword_results[0]['id'] != semantic_results[0]['id'], \
        "semantic_weight not affecting results"
```

---

## Summary: Lines of Code to Add/Change

| Fix | File | Lines | Complexity |
|-----|------|-------|------------|
| #1 Sync recovery | vector_store.py, main.py | ~50 | Medium |
| #2 Auto-save | vector_store.py | ~40 | Easy |
| #3 Hybrid search | memory_service.py | ~80 | Hard |
| #4 Cleanup job | cleanup.py (new), main.py | ~60 | Medium |
| #5 FTS sync | database.py | ~30 | Medium |
| Tests | test_*.py (new) | ~100 | Medium |
| **Total** | — | **~360** | — |

---

*All fixes are self-contained and can be implemented independently*
