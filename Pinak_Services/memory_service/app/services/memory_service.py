
import json
import os
import uuid
import datetime
import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from typing import List, Optional, Tuple, Dict, Any

class MemoryService:
    """The core logic for the memory service, handling vector search and storage."""

    def __init__(self, config_path='app/core/config.json'):
        print("Initializing MemoryService with real logic...")
        self.config = self._load_config(config_path)
        self._ensure_data_directory()
        self.model = SentenceTransformer(self.config['embedding_model'])
        # tenant -> (index, metadata)
        self._tenant_stores: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._current_tenant: str = "default"
        self.index, self.metadata = self._load_or_create_vector_db()
        self.redis_client = self._connect_to_redis()
        print("MemoryService initialized.")

    def _connect_to_redis(self):
        """Connects to the Redis server, prioritizing environment variables."""
        redis_host = os.getenv("REDIS_HOST", self.config.get("redis_host", "localhost"))
        redis_port = int(os.getenv("REDIS_PORT", self.config.get("redis_port", 6379)))
        
        try:
            client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            client.ping()
            print(f"Successfully connected to Redis at {redis_host}:{redis_port}.")
            return client
        except redis.exceptions.ConnectionError as e:
            raise Exception(f"Could not connect to Redis at {redis_host}:{redis_port}. Error: {e}")

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _ensure_data_directory(self):
        data_dir = os.path.dirname(self.config['vector_db_path'])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _paths_for_tenant(self, tenant: Optional[str] = None) -> Tuple[str, str]:
        t = tenant or self._current_tenant
        base = os.path.dirname(self.config['vector_db_path'])
        if t == "default":
            return (self.config['vector_db_path'], self.config['metadata_db_path'])
        tdir = os.path.join(base, 'tenants', t)
        os.makedirs(tdir, exist_ok=True)
        return (os.path.join(tdir, 'memory.faiss'), os.path.join(tdir, 'metadata.json'))

    def switch_tenant(self, tenant: str) -> None:
        if not tenant:
            tenant = "default"
        self._current_tenant = tenant
        if tenant in self._tenant_stores:
            self.index, self.metadata = self._tenant_stores[tenant]
            return
        self.index, self.metadata = self._load_or_create_vector_db()
        self._tenant_stores[tenant] = (self.index, self.metadata)

    def _load_or_create_vector_db(self):
        db_path, meta_path = self._paths_for_tenant()
        if os.path.exists(db_path) and os.path.exists(meta_path):
            index = faiss.read_index(db_path)
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            embedding_dim = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_dim)
            metadata = {}
        return index, metadata

    def _save_vector_db(self):
        db_path, meta_path = self._paths_for_tenant()
        faiss.write_index(self.index, db_path)
        with open(meta_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_memory(self, memory_data: MemoryCreate) -> MemoryRead:
        memory_id = str(uuid.uuid4())
        embedding = self.model.encode([memory_data.content])[0].astype('float32')
        self.index.add(np.array([embedding]))
        
        new_meta = {
            "id": memory_id,
            "content": memory_data.content,
            "tags": memory_data.tags,
            "created_at": datetime.datetime.utcnow().isoformat(),
        }
        self.metadata[str(self.index.ntotal - 1)] = new_meta
        self._save_vector_db()
        return MemoryRead.model_validate(new_meta)

    def set_expiry_by_id(self, memory_id: str, ttl_seconds: Optional[int]) -> None:
        if not ttl_seconds or ttl_seconds <= 0:
            return
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(seconds=ttl_seconds)).isoformat()
        changed = False
        for k, v in self.metadata.items():
            if isinstance(v, dict) and v.get("id") == memory_id:
                v["expires_at"] = expires_at
                changed = True
                break
        if changed:
            self._save_vector_db()

    def search_memory(self, query: str, k: int = 5) -> List[MemorySearchResult]:
        query_embedding = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for i in range(len(indices[0])):
            index_pos = str(indices[0][i])
            if index_pos in self.metadata:
                meta = self.metadata[index_pos]
                exp = meta.get("expires_at")
                if exp:
                    try:
                        if datetime.datetime.fromisoformat(exp) < datetime.datetime.utcnow():
                            continue
                    except Exception:
                        pass
                result_with_dist = {**meta, "distance": float(distances[0][i])}
                results.append(MemorySearchResult.model_validate(result_with_dist))
        return results

# Singleton instance
memory_service = MemoryService()
