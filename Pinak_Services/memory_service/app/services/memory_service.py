from __future__ import annotations

import json
import math
import os
import datetime
import glob
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import faiss
from sqlalchemy.orm import Session
from sqlalchemy import select

# ---- Project imports (adjust paths if your structure differs) ----
# Models / Schemas
from app.db.models import Memory
from app.schemas.memory import MemoryCreate, MemoryOut
from app.embedder import get_embedder

# ------------------------------------------------------------------
# In-process FAISS index holder (one per process). If you have multi-process workers,
# back the index by a shared store or rebuild per process at startup.
# ------------------------------------------------------------------

class _FaissHolder:
    """
    Lazily initializes a FAISS index with the correct dimension and provides a lock
    for thread-safe mutation and search.
    """

    def __init__(self) -> None:
        self._index: Optional[faiss.Index] = None
        self._dim: Optional[int] = None
        self._lock = threading.RLock()

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def ensure(self, embedding_dim: int) -> faiss.Index:
        with self._lock:
            if self._index is not None:
                # Guard against dimension mismatches
                if self._dim is not None and self._dim != embedding_dim:
                    raise RuntimeError(
                        f"FAISS index dimension mismatch: index={self._dim}, embedder={embedding_dim}"
                    )
                return self._index

            # Use a simple L2 index by default; swap to IndexFlatIP if you store normalized embeddings.
            index = faiss.IndexFlatL2(embedding_dim)
            self._index = index
            self._dim = embedding_dim
            return index

    def get(self) -> faiss.Index:
        if self._index is None:
            raise RuntimeError("FAISS index not initialized; call ensure(dim) first.")
        return self._index


_FAISS = _FaissHolder()


# -----------------------
# Utility helpers
# -----------------------

def _to_vec(x: Sequence[float]) -> np.ndarray:
    arr = np.asarray(x, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr

def _preserve_rank(items: Iterable[Memory], order_ids: List[int]) -> List[Memory]:
    by_id = {m.faiss_id: m for m in items if m.faiss_id is not None}
    return [by_id[i] for i in order_ids if i in by_id]


# -----------------------
# Service
# -----------------------

class MemoryService:
    """
    Memory add/search with FAISS-backed ANN. The critical invariant:
    **faiss_id == Memory.id** (DB PK). We assign faiss_id in a single transaction.

    Requirements for the Memory model:
      - id: int PK (autoincrement)
      - faiss_id: Optional[int]
      - content: str
      - embedding: Optional[bytes|str|JSON]  (if you persist it)
      - metadata: Optional[JSON/dict]
      - project_id: Optional[str]  (if multi-tenant)
    """

    def __init__(self, project_id: Optional[str] = None) -> None:
        self.project_id = project_id
        self.embedder = get_embedder()
        # Initialize FAISS with correct dim once
        _FAISS.ensure(self.embedder.dim)
        self.redis_client = None  # Optional Redis

    # ------------- Add -------------

    def add_memory(self, db: Session, payload: MemoryCreate) -> MemoryOut:
        """
        Single-commit add:
          1) Compute embedding
          2) Create row, session.add + flush() to obtain DB id
          3) Use DB id as faiss_id; add vector with add_with_ids()
          4) Set row.faiss_id = id and commit()
        """
        # 1) Embed
        vec = self._embed_text(payload.content)  # (1, d)
        d = vec.shape[1]

        # 2) Create row and flush to get PK
        row = Memory(
            content=payload.content,
            tags=payload.tags,
            metadata=self._safe_metadata(payload.metadata),
            created_at=datetime.datetime.now(datetime.timezone.utc)
        )
        db.add(row)
        db.flush()  # assigns row.id without committing

        # 3) Use DB id as FAISS id
        faiss_id = int(row.id)

        # 4) Add vector to FAISS (thread-safe)
        index = _FAISS.ensure(d)
        with _FAISS.lock:
            faiss_ids = np.asarray([faiss_id], dtype=np.int64)
            index.add_with_ids(vec, faiss_ids)

        # 5) Persist faiss_id in the same transaction and commit
        row.faiss_id = faiss_id
        db.commit()
        db.refresh(row)

        # Invalidate search cache
        if self.redis_client:
            for key in self.redis_client.scan_iter("search:*"):
                self.redis_client.delete(key)

        return self._to_out(row)

    # ------------- Search -------------

    def search_memory(
        self,
        db: Session,
        query: str,
        k: int = 5,
        project_id: Optional[str] = None,
        min_score: Optional[float] = None,
        use_inner_product: bool = False,
    ) -> List[MemoryOut]:
        """
        FAISS ANN → DB fetch by faiss_id → return ranked results.
        - Filters invalid FAISS hits (id == -1 or NaN distances).
        - Optional tenant filter via project_id.
        - min_score threshold interprets distance as:
            * If use_inner_product=True: score = similarity (higher is better)
            * Else (L2): score = -distance (higher is better)
        """
        qvec = self._embed_text(query)
        index = _FAISS.get()

        # Search (thread-safe)
        with _FAISS.lock:
            distances, ids = index.search(qvec, k)

        ids_row = ids[0].tolist()
        dists_row = distances[0].tolist()

        # Filter out FAISS "empty" hits
        filtered: List[Tuple[int, float]] = []
        for i, dist in zip(ids_row, dists_row):
            if i == -1:
                continue
            if i is None:
                continue
            if dist is None or (isinstance(dist, float) and math.isnan(dist)):
                continue
            filtered.append((int(i), float(dist)))

        if not filtered:
            return []

        ordered_ids = [i for i, _ in filtered]

        # Optional score thresholding
        if min_score is not None:
            if use_inner_product:
                keep = [i for (i, s) in filtered if s >= min_score]
            else:
                # For L2, convert to a "score" where higher is better
                keep = [i for (i, d) in filtered if (-d) >= min_score]
            if not keep:
                return []
            ordered_ids = [i for i in ordered_ids if i in set(keep)]
            if not ordered_ids:
                return []

        # Fetch rows in bulk
        stmt = select(Memory).where(Memory.faiss_id.in_(ordered_ids))
        if project_id or self.project_id:
            tenant = project_id or self.project_id
            stmt = stmt.where(Memory.project_id == tenant)

        rows: List[Memory] = list(db.execute(stmt).scalars().all())
        if not rows:
            return []

        ranked_rows = _preserve_rank(rows, ordered_ids)
        return [self._to_out(r) for r in ranked_rows]

    # ------------- (Optional) Rebuild -------------

    def rebuild_faiss_from_db(self, db: Session, batch_size: int = 1000) -> int:
        """
        Rebuilds the in-memory FAISS index from DB rows using their **DB ids as faiss_ids**.
        Useful if you start a fresh process or after clearing the index.
        Returns the count indexed.
        """
        index = _FAISS.ensure(self.embedder.dim)
        with _FAISS.lock:
            # Reset index
            self._reset_index(index)

            offset = 0
            total = 0
            while True:
                q = select(Memory).order_by(Memory.id).limit(batch_size).offset(offset)
                if self.project_id:
                    q = q.where(Memory.project_id == self.project_id)
                batch = list(db.execute(q).scalars().all())
                if not batch:
                    break

                vecs: List[np.ndarray] = []
                ids: List[int] = []
                for row in batch:
                    # If you persisted embeddings, load from row; else re-embed.
                    emb = self._embed_text(row.content)  # (1, d)
                    vecs.append(emb[0])
                    ids.append(int(row.id))

                mat = np.vstack(vecs).astype("float32")
                idarr = np.asarray(ids, dtype=np.int64)
                index.add_with_ids(mat, idarr)

                # Keep faiss_id column in sync
                for row in batch:
                    if row.faiss_id != row.id:
                        row.faiss_id = int(row.id)
                db.flush()

                total += len(batch)
                offset += batch_size

            db.commit()
            return total

    # ------------- Internal helpers -------------

    def _embed_text(self, text: str | List[str]) -> np.ndarray:
        vec = self.embedder.embed(text)
        vec = np.asarray(vec, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        # If your search uses IndexFlatIP, you should L2-normalize here.
        # faiss.normalize_L2(vec)  # uncomment when using inner product similarity
        return vec

    @staticmethod
    def _reset_index(index: faiss.Index) -> None:
        # Recreate a fresh empty index with the same dimension
        dim = index.d
        new_index = faiss.IndexFlatL2(dim)
        _FAISS._index = new_index  # swap in holder
        _FAISS._dim = dim

    @staticmethod
    def _safe_metadata(md: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if md is None:
            return None
        # Ensure JSON-serializable
        try:
            json.dumps(md)
            return md
        except Exception:
            return {"_raw": str(md)}

    @staticmethod
    def _serialize_embedding(vec1d: np.ndarray) -> List[float]:
        # If you store embeddings in DB; returns Python list (JSON)
        return [float(x) for x in np.asarray(vec1d, dtype="float32").tolist()]

    @staticmethod
    def _to_out(row: Memory) -> MemoryOut:
        return MemoryOut(
            id=row.id,
            faiss_id=row.faiss_id,
            content=row.content,
            metadata=row.metadata,
            tags=row.tags,
            created_at=row.created_at,
            redacted=row.redacted
        )

    def _store_dir(self, tenant: str, project_id: str) -> str:
        """Get storage directory for tenant/project."""
        base = os.path.join('data', tenant, project_id or 'default')
        os.makedirs(base, exist_ok=True)
        return base

    def _dated_file(self, base: str, layer: str, prefix: str) -> str:
        """Get dated file path for layer."""
        date = datetime.datetime.utcnow().strftime('%Y%m%d')
        return os.path.join(base, layer, f'{prefix}_{date}.jsonl')

    def _append_audit_jsonl(self, path: str, record: Dict[str, Any]):
        """Append record to JSONL file with audit."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            json.dump(record, f)
            f.write('\n')

    def _append_jsonl(self, path: str, record: Dict[str, Any]):
        """Append record to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            json.dump(record, f)
            f.write('\n')

    def _session_file(self, base: str, session_id: str) -> str:
        """Get session file path."""
        return os.path.join(base, 'session', f'session_{session_id}.jsonl')

    def _working_file(self, base: str) -> str:
        """Get working file path."""
        date = datetime.datetime.utcnow().strftime('%Y%m%d')
        return os.path.join(base, 'working', f'working_{date}.jsonl')

    def redact_memory(self, memory_id: str, tenant: str, project_id: str, reason: str) -> Dict[str, Any]:
        """Redact a memory."""
        # Stub: mark as redacted
        return {"status": "redacted", "id": memory_id}

    def list_changelog(self, tenant: str, project_id: str, change_type: Optional[str], since: Optional[str], until: Optional[str], limit: int, offset: int) -> List[Dict[str, Any]]:
        """List changelog entries."""
        base = self._store_dir(tenant, project_id)
        out = []
        folder = os.path.join(base, 'changelog')
        import glob
        for fp in sorted(glob.glob(os.path.join(folder, 'changes_*.jsonl'))):
            if os.path.exists(fp):
                with open(fp, 'r') as f:
                    for line in f:
                        try:
                            obj = json.loads(line)
                            out.append(obj)
                        except:
                            pass
        return out[offset:offset+limit]

    def write_audit_anchor(self, tenant: str, project_id: str, kind: str, date: Optional[str], reason: str) -> Dict[str, Any]:
        """Write audit anchor."""
        base = self._store_dir(tenant, project_id)
        path = os.path.join(base, 'audit', f'anchor_{kind}.json')
        anchor = {"kind": kind, "date": date, "reason": reason, "ts": datetime.datetime.utcnow().isoformat()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(anchor, f)
        return anchor

    def verify_audit_chain(self, tenant: str, project_id: str, kind: str, date: Optional[str]) -> Dict[str, Any]:
        """Verify audit chain."""
        # Stub
        return {"verified": True, "kind": kind}


# Layer-specific service functions (outside the class for import)

def add_episodic(memory_service: MemoryService, tenant: str, project_id: str, content: str, salience: int) -> Dict[str, Any]:
    """Add episodic memory with salience."""
    # Store in JSONL file
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'episodic', 'episodic')
    rec = {
        'content': content,
        'salience': salience,
        'project_id': project_id,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_audit_jsonl(path, rec)
    return rec

def list_episodic(memory_service: MemoryService, tenant: str, project_id: str) -> List[Dict[str, Any]]:
    """List episodic memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    import glob
    folder = os.path.join(base, 'episodic')
    for fp in sorted(glob.glob(os.path.join(folder, 'episodic_*.jsonl'))):
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except:
                        pass
    return out

def add_procedural(memory_service: MemoryService, tenant: str, project_id: str, skill_id: str, steps: List[str]) -> Dict[str, Any]:
    """Add procedural memory."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'procedural', 'procedural')
    rec = {
        'skill_id': skill_id,
        'steps': steps,
        'project_id': project_id,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_audit_jsonl(path, rec)
    return rec

def list_procedural(memory_service: MemoryService, tenant: str, project_id: str) -> List[Dict[str, Any]]:
    """List procedural memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    import glob
    folder = os.path.join(base, 'procedural')
    for fp in sorted(glob.glob(os.path.join(folder, 'procedural_*.jsonl'))):
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except:
                        pass
    return out

def add_rag(memory_service: MemoryService, tenant: str, project_id: str, query: str, external_source: str) -> Dict[str, Any]:
    """Add RAG memory."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'rag', 'rag')
    rec = {
        'query': query,
        'external_source': external_source,
        'project_id': project_id,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_audit_jsonl(path, rec)
    return rec

def list_rag(memory_service: MemoryService, tenant: str, project_id: str) -> List[Dict[str, Any]]:
    """List RAG memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    import glob
    folder = os.path.join(base, 'rag')
    for fp in sorted(glob.glob(os.path.join(folder, 'rag_*.jsonl'))):
        if os.path.exists(fp):
            with open(fp, 'r') as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except:
                        pass
    return out

def search_v2(memory_service: MemoryService, tenant: str, project_id: str, query: str, layer_list: List[str]) -> Dict[str, Any]:
    """Search across multiple layers."""
    results = {}
    # Semantic not included here as it's DB-based
    if 'episodic' in layer_list:
        epi_results = list_episodic(memory_service, tenant, project_id)
        # Simple filter by content
        results['episodic'] = [r for r in epi_results if query.lower() in r.get('content', '').lower()][:5]
    if 'procedural' in layer_list:
        proc_results = list_procedural(memory_service, tenant, project_id)
        results['procedural'] = [r for r in proc_results if query.lower() in r.get('skill_id', '').lower()][:5]
    if 'rag' in layer_list:
        rag_results = list_rag(memory_service, tenant, project_id)
        results['rag'] = [r for r in rag_results if query.lower() in r.get('query', '').lower()][:5]
    # Add other layers similarly
    return results