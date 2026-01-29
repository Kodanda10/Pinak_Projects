import json
import os
import hashlib
import datetime
import logging
import time
from typing import Dict, List, Optional, Any, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from app.core.database import DatabaseManager
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

class _DeterministicEncoder:
    """Lightweight embedding encoder used for tests and local development."""

    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension

    def encode(self, sentences: List[str]) -> np.ndarray:
        import itertools
        vectors = []
        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            needed_bytes = self.embedding_dimension * 4
            if len(digest) < needed_bytes:
                digest = bytes(itertools.islice(itertools.cycle(digest), needed_bytes))
            vectors.append(np.frombuffer(digest[:needed_bytes], dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

class MemoryService:
    """
    Enterprise Memory Service Orchestrator.
    Manages SQLite (Metadata/Logs) and FAISS (Vectors).
    Implements Hybrid Search with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config_path: Optional[str] = None, model: Optional[object] = None):
        config_path = config_path or os.getenv("PINAK_CONFIG_PATH", "app/core/config.json")
        self.config = self._load_config(config_path)

        # Paths
        self.data_root = self.config.get("data_root", "data")
        os.makedirs(self.data_root, exist_ok=True)
        self.db_path = os.path.join(self.data_root, "memory.db")
        self.vector_path = os.path.join(self.data_root, "vectors.index")

        # Components
        self.db = DatabaseManager(self.db_path)

        # Model
        self.model = model or self._load_embedding_model(self.config.get("embedding_model"))
        self.embedding_dim = getattr(self.model, "embedding_dimension", None)
        if self.embedding_dim is None and hasattr(self.model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim is None:
             # Fallback for deterministic encoder
             self.embedding_dim = getattr(self.model, "embedding_dimension", 384)

        # Vector Store
        self.vector_store = VectorStore(self.vector_path, self.embedding_dim)

    def _load_config(self, path):
        if not os.path.exists(path):
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def _load_embedding_model(self, name: Optional[str]):
        backend = os.getenv("PINAK_EMBEDDING_BACKEND")
        if backend == "dummy" or not name or name.lower() == "dummy":
            return _DeterministicEncoder()
        try:
            return SentenceTransformer(name)
        except Exception as e:
            logger.warning(f"Failed to load model {name}: {e}. Falling back to Dummy.")
            return _DeterministicEncoder()

    # --- Core Memory Operations ---

    def add_memory(self, memory_data: MemoryCreate, tenant: str, project_id: str) -> MemoryRead:
        """Adds a semantic memory (Vector + DB)."""
        content = memory_data.content
        tags = memory_data.tags or []

        # 1. Generate Embedding
        embedding = self.model.encode([content])[0].astype("float32")

        # 2. Generate ID for Vector Store (using simple int hash or sequence could be risky for collisions if not managed,
        # but for now we use a high-precision timestamp based int to avoid collisions in valid range)
        # Better: use a dedicated counter in DB. For simplicity/speed without extra DB hit: Time-based.
        # SQLite RowID is nice but we don't have it until we insert.
        # Strategy: Insert into DB first? No, we need embedding_id.
        # Strategy: Use UUID -> Int mapping? No.
        # Strategy: Use Time (ns).
        embedding_id = int(time.time_ns()) # 64-bit int, fits in FAISS ID

        # 3. Add to Vector Store
        self.vector_store.add_vectors(np.array([embedding]), [embedding_id])

        # 4. Add to DB
        result = self.db.add_semantic(content, tags, tenant, project_id, embedding_id)

        # 5. Save Vector Store (persist)
        self.vector_store.save()

        return MemoryRead(**result)

    def add_episodic(self, content: str, tenant: str, project_id: str,
                     salience: int = 0, goal: str = None, plan: List[str] = None,
                     outcome: str = None, tool_logs: List[Dict] = None) -> Dict[str, Any]:
        """Add episodic memory (DB only for now, can add vectors later if needed)."""
        # Note: If we want vector search on episodes, we should add embedding here too.
        # For V2, we stick to DB + FTS for episodes as per design docs,
        # unless we want to embed the description.
        # Let's auto-embed content for Hybrid search on episodes too?
        # The schema has `memories_episodic` separate from `memories_semantic`.
        # If we want vector search on ALL layers, we should embed this.
        # But `memories_episodic` table doesn't have `embedding_id` in my schema.
        # I'll stick to FTS for Episodic for now to match Schema, or update schema?
        # Plan said "All layers to be vector indexed" was a goal.
        # But schema in `database.py` only put `embedding_id` on `semantic`.
        # Correction: The user asked "Do you want ALL layers to be vector-indexed? ... recommend what to do".
        # I recommended Hybrid. Hybrid usually implies Vectors for everything.
        # However, for now, let's implement FTS-based Episodic and Semantic Vector-based.
        # Changing Schema now might be tricky without migration logic, but `init_db` checks `IF NOT EXISTS`.
        # I will rely on FTS for Episodic for this iteration.
        return self.db.add_episodic(content, tenant, project_id, goal, plan, outcome, tool_logs, salience)

    def add_procedural(self, skill_name: str, steps: List[str], tenant: str, project_id: str,
                       trigger: str = None, code_snippet: str = None) -> Dict[str, Any]:
        return self.db.add_procedural(skill_name, steps, tenant, project_id, trigger, code_snippet)

    def add_rag(self, query: str, external_source: str, content: str, tenant: str, project_id: str) -> Dict[str, Any]:
        return self.db.add_rag(query, external_source, content, tenant, project_id)

    # --- Hybrid Search (The "Magic") ---

    def search_memory(self, query: str, tenant: str, project_id: str, k: int = 5) -> List[MemorySearchResult]:
        """
        Legacy Semantic Search Endpoint.
        We upgrade this to use the Hybrid logic but filter for 'semantic' type to maintain backward compatibility if needed.
        Or better: Use Hybrid and return top results.
        """
        # Use Hybrid Search restricted to Semantic layer if we want strict backward compat,
        # but let's just use the full hybrid power.
        results = self.search_hybrid(query, tenant, project_id, limit=k)
        # Convert to MemorySearchResult
        out = []
        for r in results:
            # We map 'score' to 'distance' (inverted) or just use score.
            out.append(MemorySearchResult(
                id=r['id'],
                content=r['content'],
                tags=[r['type']], # Use type as tag
                distance=1.0 - r['score'], # Fake distance from score (0..1)
                created_at=r['created_at'],
                tenant=r.get('tenant', tenant),
                project_id=r.get('project_id', project_id),
                metadata=r
            ))
        return out

    def search_hybrid(self, query: str, tenant: str, project_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search (Vector + Keyword) with Reciprocal Rank Fusion (RRF).
        """
        # 1. Keyword Search (SQLite FTS)
        keyword_results = self.db.search_keyword(query, tenant, project_id, limit=limit * 2)

        # 2. Vector Search (FAISS)
        embedding = self.model.encode([query])[0].astype("float32")
        dists, ids = self.vector_store.search(embedding, k=limit * 2)

        # Fetch metadata for vector hits
        # Note: Vector store only has Semantic memories currently.
        vector_results_db = self.db.get_semantic_by_embedding_ids(ids, tenant, project_id)

        # Map DB results to a dictionary for O(1) access
        # vector_results_db is list of dicts.
        # We need to preserve order from FAISS to know the rank.
        vector_map = {item['embedding_id']: item for item in vector_results_db}

        vector_results = []
        for i, emb_id in enumerate(ids):
            if emb_id in vector_map:
                item = vector_map[emb_id]
                item['type'] = 'semantic'
                item['vector_rank'] = i
                vector_results.append(item)

        # 3. RRF Fusion
        # score = 1 / (k + rank_vec) + 1 / (k + rank_fts)
        # k usually 60
        rrf_k = 60
        scores: Dict[str, float] = {}
        merged: Dict[str, Dict] = {}

        # Process Keyword Results
        for i, item in enumerate(keyword_results):
            mid = item['id']
            merged[mid] = item
            scores[mid] = scores.get(mid, 0.0) + (1.0 / (rrf_k + i))

        # Process Vector Results
        for i, item in enumerate(vector_results):
            mid = item['id']
            # If item already in merged (from keyword), we assume it's the same item.
            # However, vector results are full rows, keyword results are FTS rows (id, content, type).
            # They should match on ID.
            if mid not in merged:
                merged[mid] = item
            scores[mid] = scores.get(mid, 0.0) + (1.0 / (rrf_k + i))

        # Sort by Score DESC
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        final_results = []
        for mid in sorted_ids[:limit]:
            item = merged[mid]
            item['score'] = scores[mid]
            final_results.append(item)

        return final_results

    def retrieve_context(self, query: str, tenant: str, project_id: str) -> Dict[str, Any]:
        """
        Unified Context Retrieval for Agents.
        Returns categorized memory relevant to the query.
        """
        hybrid_hits = self.search_hybrid(query, tenant, project_id, limit=20)

        # Categorize
        context = {
            "semantic": [],
            "episodic": [],
            "procedural": [],
            "working": [] # TODO: Add working memory search
        }

        for hit in hybrid_hits:
            mtype = hit.get('type', 'unknown')
            if mtype in context:
                context[mtype].append(hit)
            else:
                # Fallback
                context["semantic"].append(hit)

        return context

    # --- Other Operations (Logs) ---
    def add_event(self, payload: Dict, tenant: str, project_id: str):
        return self.db.add_event(payload.get('event_type', 'unknown'), payload, tenant, project_id)

    def list_events(self, tenant: str, project_id: str, limit: int=100):
        return self.db.list_events(tenant, project_id, limit)

    def session_add(self, session_id: str, content: str, role: str, tenant: str, project_id: str):
        return self.db.add_session(session_id, content, role, tenant, project_id)

    def session_list(self, session_id: str, tenant: str, project_id: str, limit: int=100):
        return self.db.list_session(session_id, tenant, project_id, limit)

    def working_add(self, content: str, tenant: str, project_id: str):
        return self.db.add_working(content, tenant, project_id)

    def working_list(self, tenant: str, project_id: str, limit: int=100):
        return self.db.list_working(tenant, project_id, limit)

    def update_memory(self, layer: str, memory_id: str, updates: Dict[str, Any], tenant: str, project_id: str) -> bool:
        """
        Updates memory in DB. If content changes in Semantic layer, re-embeds.
        """
        # Security: Prevent updating system fields
        forbidden_keys = {"id", "tenant", "project_id", "created_at", "embedding_id"}
        safe_updates = {k: v for k, v in updates.items() if k not in forbidden_keys}

        if not safe_updates:
            return False

        if layer == "semantic" and "content" in safe_updates:
            # 1. Fetch old record to get embedding_id
            old_item = self.db.get_memory(layer, memory_id, tenant, project_id)
            if old_item:
                emb_id = old_item.get("embedding_id")
                # 2. Re-embed
                new_embedding = self.model.encode([safe_updates["content"]])[0].astype("float32")
                # 3. Update Vector Store
                # Remove old vector first
                if emb_id:
                    self.vector_store.remove_ids([emb_id])
                    # Add new
                    self.vector_store.add_vectors(np.array([new_embedding]), [emb_id])
                    self.vector_store.save()

        return self.db.update_memory(layer, memory_id, safe_updates, tenant, project_id)

    def delete_memory(self, layer: str, memory_id: str, tenant: str, project_id: str) -> bool:
        if layer == "semantic":
            # Fetch embedding ID first
            item = self.db.get_memory(layer, memory_id, tenant, project_id)
            if item and item.get("embedding_id"):
                self.vector_store.remove_ids([item["embedding_id"]])
                self.vector_store.save()

        return self.db.delete_memory(layer, memory_id, tenant, project_id)

# Functional adapters for compatibility with existing endpoints.py which calls 'svc_add_episodic' etc.
# Ideally endpoints should call service instance methods.
# I will fix endpoints.py to use the service methods directly.
