import json
import os
import re
import uuid
import hashlib
import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer

from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult


class _DeterministicEncoder:
    """Lightweight embedding encoder used for tests and local development."""

    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension

    def encode(self, sentences: List[str]) -> np.ndarray:
        vectors = []
        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode("utf-8")).digest()
            needed_bytes = self.embedding_dimension * 4
            if len(digest) < needed_bytes:
                digest = (digest * ((needed_bytes // len(digest)) + 1))[:needed_bytes]
            vectors.append(np.frombuffer(digest[:needed_bytes], dtype=np.float32))
        return np.array(vectors, dtype=np.float32)

class MemoryService:
    """The core logic for the memory service, handling vector search and storage."""

    def __init__(self, config_path: Optional[str] = None, model: Optional[object] = None):
        config_path = config_path or os.getenv("PINAK_CONFIG_PATH", "app/core/config.json")
        self.config = self._load_config(config_path)
        self.data_root = self._determine_data_root()
        os.makedirs(self.data_root, exist_ok=True)
        self.model = model or self._load_embedding_model(self.config.get("embedding_model"))
        self.embedding_dim = getattr(self.model, "embedding_dimension", None)
        if self.embedding_dim is None and hasattr(self.model, "get_sentence_embedding_dimension"):
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        if self.embedding_dim is None:
            raise RuntimeError("Unable to determine embedding dimension for the configured model")
        self._vector_stores: Dict[Tuple[str, str], Dict[str, object]] = {}
        self.redis_client = self._connect_to_redis()

    def _determine_data_root(self) -> str:
        if "data_root" in self.config:
            return self.config["data_root"]
        vector_path = self.config.get("vector_db_path", "data/memory.faiss")
        return os.path.dirname(vector_path) or "data"

    def _load_embedding_model(self, name: Optional[str]):
        backend = os.getenv("PINAK_EMBEDDING_BACKEND")
        if backend == "dummy" or not name or name.lower() == "dummy":
            return _DeterministicEncoder()
        return SentenceTransformer(name)

    def _connect_to_redis(self):
        """Connects to the Redis server, prioritizing environment variables."""
        redis_host = os.getenv("REDIS_HOST", self.config.get("redis_host", "localhost"))
        redis_port = int(os.getenv("REDIS_PORT", self.config.get("redis_port", 6379)))
        
        try:
            client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            client.ping()
            print(f"Successfully connected to Redis at {redis_host}:{redis_port}.")
            return client
        except Exception as e:
            print(f"Redis not available at {redis_host}:{redis_port}. Continuing without Redis. Error: {e}")
            return None

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _vector_paths(self, tenant: str, project_id: str) -> Tuple[str, str]:
        base = self._store_dir(tenant, project_id)
        semantic_dir = os.path.join(base, "semantic")
        os.makedirs(semantic_dir, exist_ok=True)
        return (
            os.path.join(semantic_dir, "memory.faiss"),
            os.path.join(semantic_dir, "metadata.json"),
        )

    def _load_vector_store(self, tenant: str, project_id: str) -> Dict[str, object]:
        vector_path, metadata_path = self._vector_paths(tenant, project_id)
        if os.path.exists(vector_path) and os.path.exists(metadata_path):
            index = faiss.read_index(vector_path)
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        else:
            index = faiss.IndexFlatL2(self.embedding_dim)
            metadata = {}
        return {"index": index, "metadata": metadata, "vector_path": vector_path, "metadata_path": metadata_path}

    def _get_vector_store(self, tenant: str, project_id: str) -> Dict[str, object]:
        key = (tenant, project_id)
        if key not in self._vector_stores:
            self._vector_stores[key] = self._load_vector_store(tenant, project_id)
        return self._vector_stores[key]

    def _save_vector_store(self, tenant: str, project_id: str) -> None:
        store = self._vector_stores.get((tenant, project_id))
        if not store:
            return
        faiss.write_index(store["index"], store["vector_path"])
        with open(store["metadata_path"], "w", encoding="utf-8") as fh:
            json.dump(store["metadata"], fh, indent=2)

    def add_memory(self, memory_data: MemoryCreate, tenant: str, project_id: str) -> MemoryRead:
        store = self._get_vector_store(tenant, project_id)
        memory_id = str(uuid.uuid4())
        embedding = self.model.encode([memory_data.content])[0].astype("float32")
        store["index"].add(np.array([embedding]))

        tags = memory_data.tags or []
        new_meta = {
            "id": memory_id,
            "content": memory_data.content,
            "tags": tags,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "tenant": tenant,
            "project_id": project_id,
        }
        store["metadata"][str(store["index"].ntotal - 1)] = new_meta
        self._save_vector_store(tenant, project_id)
        return MemoryRead.model_validate(new_meta)

    def search_memory(self, query: str, tenant: str, project_id: str, k: int = 5) -> List[MemorySearchResult]:
        store = self._get_vector_store(tenant, project_id)
        if store["index"].ntotal == 0:
            return []
        k = min(k, store["index"].ntotal)
        query_embedding = self.model.encode([query])[0].astype("float32")
        distances, indices = store["index"].search(np.array([query_embedding]), k)

        results: List[MemorySearchResult] = []
        for i in range(len(indices[0])):
            index_pos = str(indices[0][i])
            if index_pos in store["metadata"]:
                meta = store["metadata"][index_pos]
                result_with_dist = {**meta, "distance": float(distances[0][i])}
                results.append(MemorySearchResult.model_validate(result_with_dist))
        return results

    def _sanitize_component(self, value: Optional[str]) -> str:
        if not value:
            return "default"
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
        return safe or "default"

    def _store_dir(self, tenant: str, project_id: str) -> str:
        """Get storage directory for tenant/project."""
        tenant_component = self._sanitize_component(tenant)
        project_component = self._sanitize_component(project_id)
        base = os.path.join(self.data_root, tenant_component, project_component)
        os.makedirs(base, exist_ok=True)
        return base

    def _dated_file(self, base: str, layer: str, prefix: str) -> str:
        """Get dated file path for layer."""
        date = datetime.datetime.utcnow().strftime('%Y%m%d')
        layer_dir = os.path.join(base, layer)
        os.makedirs(layer_dir, exist_ok=True)
        return os.path.join(layer_dir, f'{prefix}_{date}.jsonl')

    def _last_audit_record(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        last_record = None
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    last_record = json.loads(line)
                except json.JSONDecodeError:
                    continue
        return last_record

    def _compute_audit_hash(self, payload: dict) -> str:
        to_hash = {k: v for k, v in payload.items() if k != 'hash'}
        serialized = json.dumps(to_hash, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def _append_audit_jsonl(self, path: str, record: dict) -> dict:
        """Append record to JSONL file with tamper-evident hashing."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        prev_entry = self._last_audit_record(path)
        prev_hash = prev_entry.get('hash') if prev_entry else None
        payload = {**record, 'prev_hash': prev_hash}
        payload['hash'] = self._compute_audit_hash(payload)
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(payload, f, sort_keys=True)
            f.write('\n')
        return payload

    def _session_file(self, base: str, session_id: str) -> str:
        """Get session file path."""
        return os.path.join(base, 'session', f'session_{session_id}.jsonl')

    def _working_file(self, base: str) -> str:
        """Get working file path."""
        date = datetime.datetime.utcnow().strftime('%Y%m%d')
        return os.path.join(base, 'working', f'working_{date}.jsonl')

    def _append_jsonl(self, path: str, record: dict):
        """Append record to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(record, f, sort_keys=True)
            f.write('\n')

# Layer-specific service functions

def add_episodic(memory_service: MemoryService, tenant: str, project_id: str, content: str, salience: int) -> dict:
    """Add episodic memory with salience."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'episodic', 'episodic')
    rec = {
        'content': content,
        'salience': salience,
        'project_id': project_id,
        'tenant': tenant,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_episodic(memory_service: MemoryService, tenant: str, project_id: str) -> list:
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

def add_procedural(memory_service: MemoryService, tenant: str, project_id: str, skill_id: str, steps: list) -> dict:
    """Add procedural memory."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'procedural', 'procedural')
    rec = {
        'skill_id': skill_id,
        'steps': steps,
        'project_id': project_id,
        'tenant': tenant,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_procedural(memory_service: MemoryService, tenant: str, project_id: str) -> list:
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

def add_rag(memory_service: MemoryService, tenant: str, project_id: str, query: str, external_source: str) -> dict:
    """Add RAG memory."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'rag', 'rag')
    rec = {
        'query': query,
        'external_source': external_source,
        'project_id': project_id,
        'tenant': tenant,
        'ts': datetime.datetime.utcnow().isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_rag(memory_service: MemoryService, tenant: str, project_id: str) -> list:
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

def search_v2(memory_service: MemoryService, tenant: str, project_id: str, query: str, layer_list: list) -> dict:
    """Search across multiple layers."""
    results = {}
    if 'episodic' in layer_list:
        epi_results = list_episodic(memory_service, tenant, project_id)
        results['episodic'] = [r for r in epi_results if query.lower() in r.get('content', '').lower()][:5]
    if 'procedural' in layer_list:
        proc_results = list_procedural(memory_service, tenant, project_id)
        results['procedural'] = [r for r in proc_results if query.lower() in r.get('skill_id', '').lower()][:5]
    if 'rag' in layer_list:
        rag_results = list_rag(memory_service, tenant, project_id)
        results['rag'] = [r for r in rag_results if query.lower() in r.get('query', '').lower()][:5]
    return results
