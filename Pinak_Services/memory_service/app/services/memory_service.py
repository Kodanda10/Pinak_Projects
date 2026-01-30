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

    def __init__(self, dimension: int = 384):
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
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            self.embedding_dim = getattr(self.model, "embedding_dimension", 384)

        # Vector Store
        print(f"DEBUG: Initializing VectorStore with dimension {self.embedding_dim}")
        self.vector_store = VectorStore(self.vector_path, self.embedding_dim)

    def verify_and_recover(self):
        """
        Check consistency between DB and Vector Store. Rebuild if necessary.
        """
        # Simple heuristic: Count check
        # We need to count ALL semantic memories in DB.
        db_count = 0
        with self.db.get_cursor() as cur:
            cur.execute("SELECT count(*) FROM memories_semantic")
            db_count = cur.fetchone()[0]

        vec_count = self.vector_store.total

        if db_count != vec_count:
            logger.warning(f"Consistency Mismatch! DB: {db_count}, Vector: {vec_count}. Rebuilding Index...")
            self._rebuild_index()
        else:
            logger.info(f"System Consistent. {vec_count} memories loaded.")

    def _rebuild_index(self):
        """Re-encode all semantic memories and populate FAISS."""
        # 1. Reset Index (handled by create in store or we can manually reset if we expose it)
        # VectorStore doesn't expose reset, but we can delete file and reload or just loop add.
        # Better: Create new index in memory and swap.

        # We will use batch_add context
        with self.vector_store.batch_add():
            # Clear existing logic?
            # self.vector_store.index.reset() # We need to expose this or access protected.
            with self.vector_store.lock:
                self.vector_store.index.reset()

            # Page through DB
            offset = 0
            limit = 100
            while True:
                with self.db.get_cursor() as cur:
                    cur.execute("SELECT content, embedding_id FROM memories_semantic LIMIT ? OFFSET ?", (limit, offset))
                    rows = cur.fetchall()

                if not rows:
                    break

                # Batch Encode
                texts = [r['content'] for r in rows]
                ids = [r['embedding_id'] for r in rows]

                embeddings = self.model.encode(texts)
                self.vector_store.add_vectors(embeddings, ids)

                offset += len(rows)
                logger.info(f"Rebuilt {offset} vectors...")

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
        try:
            content = memory_data.content
            tags = memory_data.tags or []

            # 1. Generate Embedding
            embedding = self.model.encode([content])[0].astype("float32")

            # 2. Generate ID for Vector Store
            # We use a hash or a simpler counter to avoid huge int issues
            embedding_id = hash(content + str(time.time())) % (2**31 - 1)

            # 3. Add to Vector Store
            self.vector_store.add_vectors(np.array([embedding]), [embedding_id])

            # 4. Add to DB
            result = self.db.add_semantic(content, tags, tenant, project_id, embedding_id)

            # 5. Save Vector Store (persist)
            self.vector_store.save()

            return MemoryRead(**result)
        except Exception as e:
            import traceback
            print(f"CRITICAL ERROR in add_memory: {e}")
            traceback.print_exc()
            raise

    def add_episodic(self, content: str, tenant: str, project_id: str,
                     salience: int = 0, goal: str = None, plan: List[str] = None,
                     outcome: str = None, tool_logs: List[Dict] = None) -> Dict[str, Any]:
        """Add episodic memory (Vector + DB)."""
        # 1. Generate Embedding from content + goal + outcome
        search_blob = f"{content} {goal or ''} {outcome or ''}"
        embedding = self.model.encode([search_blob])[0].astype("float32")
        embedding_id = hash(search_blob + str(time.time())) % (2**31 - 1)

        # 2. Add to Vector Store
        self.vector_store.add_vectors(np.array([embedding]), [embedding_id])

        # 3. Add to DB
        return self.db.add_episodic(content, tenant, project_id, goal, plan, outcome, tool_logs, salience, embedding_id)

    def add_procedural(self, skill_name: str, steps: List[str], tenant: str, project_id: str,
                       description: str = None, trigger: str = None, code_snippet: str = None) -> Dict[str, Any]:
        """Add procedural memory (Vector + DB)."""
        # 1. Generate Embedding from skill_name + trigger + description
        search_blob = f"{skill_name} {trigger or ''} {description or ''}"
        embedding = self.model.encode([search_blob])[0].astype("float32")
        embedding_id = hash(search_blob + str(time.time())) % (2**31 - 1)

        # 2. Add to Vector Store
        self.vector_store.add_vectors(np.array([embedding]), [embedding_id])

        # 3. Add to DB
        return self.db.add_procedural(skill_name, steps, tenant, project_id, description, trigger, code_snippet, embedding_id)

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

    def search_hybrid(self, query: str, tenant: str, project_id: str, limit: int = 10, semantic_weight: float = 0.5) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search (Vector + Keyword) with Weighted Fusion.
        semantic_weight: 0.0 = pure keyword, 1.0 = pure semantic.
        """
        # 1. Keyword Search (SQLite FTS)
        keyword_results = self.db.search_keyword(query, tenant, project_id, limit=limit * 2)

        # 2. Vector Search (FAISS)
        embedding = self.model.encode([query])[0].astype("float32")
        dists, ids = self.vector_store.search(embedding, k=limit * 2)

        # Filter out -1 ids (no matches)
        valid_ids = [int(i) for i in ids[0] if i != -1]

        # Unified result retrieval (Semantic, Episodic, Procedural)
        vector_results_db = self.db.get_memories_by_embedding_ids(valid_ids, tenant, project_id)
        vector_map = {item['embedding_id']: item for item in vector_results_db}

        # 4. Score Normalization & Weighted Fusion
        fts_scores = {}
        for i, item in enumerate(keyword_results):
            # Rank based score: 1.0 for first, decaying
            score = 1.0 - (i / len(keyword_results))
            fts_scores[item['id']] = score

        vector_scores = {}
        for i, idx_raw in enumerate(ids[0]):
            idx = int(idx_raw)
            if idx == -1: continue
            
            if idx in vector_map:
                item = vector_map[idx]
                mid = item['id']
                dist = dists[0][i]
                
                # Heuristic: 1 / (1 + dist)
                # Handle inf or huge dists (as 0.1 score baseline)
                if np.isinf(dist) or dist > 1e12:
                    score = 0.1
                else:
                    score = 1.0 / (1.0 + dist)
                
                vector_scores[mid] = score

        # 4. Weighted Fusion
        merged: Dict[str, Dict] = {}
        all_ids = set(fts_scores.keys()) | set(vector_scores.keys())
        final_scores = {}

        for mid in all_ids:
            # Get item data from either source
            item = None
            if mid in fts_scores:
                # Find in keyword_results
                item = next((x for x in keyword_results if x['id'] == mid), None)

            if not item and mid in vector_scores:
                # Find in vector results
                item = next((x for x in vector_results_db if x['id'] == mid), None)

            if item:
                if mid not in merged:
                    merged[mid] = item

                s_vec = vector_scores.get(mid, 0.0)
                s_fts = fts_scores.get(mid, 0.0)

                score = (semantic_weight * s_vec) + ((1.0 - semantic_weight) * s_fts)
                final_scores[mid] = score

        # Sort by Score DESC
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)

        final_results = []
        for mid in sorted_ids[:limit]:
            item = merged[mid]
            item['score'] = final_scores[mid]
            final_results.append(item)

        return final_results

    def intent_sniff(self, content: str, tenant: str, project_id: str) -> List[Dict[str, Any]]:
        """
        Background Observer Logic: Detects if the current 'Working' intent
        collides with known risks in ANY memory layer.
        """
        # 1. Intent Extraction (Keyword-based for speed)
        content_lower = content.lower()
        
        # Define intent clusters
        intents = {
            "deployment": ["deploy", "vercel", "production", "ship", "push"],
            "security": ["auth", "token", "secret", "key", "access", "password"],
            "infrastructure": ["database", "db", "server", "cluster", "aws", "gcp"]
        }
        
        found_intents = [name for name, keywords in intents.items() if any(k in content_lower for k in keywords)]
        
        if not found_intents:
            return []

        # 2. Targeted Risk Search
        # We construct a query that combines the found intent keywords with high-risk terms
        risk_terms = "failed error expired warning deprecated issue problem"
        search_query = f"{' '.join(found_intents)} {risk_terms}"
        
        # Use a high semantic weight to catch conceptual collisions
        results = self.search_hybrid(search_query, tenant, project_id, limit=5, semantic_weight=0.6)

        nudges = []
        seen_messages = set()
        
        for r in results:
            # Combine all textual fields for a thorough risk check
            text_to_check = " ".join([
                str(r.get('content', '')),
                str(r.get('description', '')),
                str(r.get('goal', '')),
                str(r.get('outcome', ''))
            ]).lower()
            
            # If historical context contains negative signals related to the intents
            if any(term in text_to_check for term in ["expired", "failed", "warning", "error", "deprecated"]):
                msg = f"Collision with {r.get('type', 'memory')}: {r['content'][:100]}..."
                if msg not in seen_messages:
                    nudges.append({
                        "type": "proactive_nudge",
                        "strength": "HIGH" if any(t in text_to_check for t in ["expired", "error"]) else "MEDIUM",
                        "message": msg,
                        "source_id": r['id'],
                        "layer": r.get('type', 'procedural') if r.get('type') == 'procedural' else r.get('type', 'semantic')
                    })
                    seen_messages.add(msg)

        return nudges

    def working_add(self, content: str, tenant: str, project_id: str):
        # Add to DB
        res = self.db.add_working(content, tenant, project_id)
        # Sniff for nudges
        nudges = self.intent_sniff(content, tenant, project_id)
        if nudges:
            res['nudges'] = nudges
        return res

    def retrieve_context(self, query: str, tenant: str, project_id: str, semantic_weight: float = 0.5) -> Dict[str, Any]:
        """
        Unified Context Retrieval for Agents.
        Returns categorized memory relevant to the query.
        """
        hybrid_hits = self.search_hybrid(query, tenant, project_id, limit=20, semantic_weight=semantic_weight)

        # Categorize
        context = {
            "semantic": [],
            "episodic": [],
            "procedural": [],
            "working": []
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
