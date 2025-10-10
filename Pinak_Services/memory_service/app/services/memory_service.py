import json
import os
import uuid
import datetime
import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from typing import List

class MemoryService:
    """The core logic for the memory service, handling vector search and storage."""

    def __init__(self, config_path='app/core/config.json'):
        print("Initializing MemoryService with real logic...")
        self.config = self._load_config(config_path)
        self._ensure_data_directory()
        self.model = SentenceTransformer(self.config['embedding_model'])
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
        except Exception as e:
            print(f"Redis not available at {redis_host}:{redis_port}. Continuing without Redis. Error: {e}")
            return None

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _ensure_data_directory(self):
        data_dir = os.path.dirname(self.config['vector_db_path'])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _load_or_create_vector_db(self):
        db_path = self.config['vector_db_path']
        meta_path = self.config['metadata_db_path']
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
        faiss.write_index(self.index, self.config['vector_db_path'])
        with open(self.config['metadata_db_path'], 'w') as f:
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

    def search_memory(self, query: str, k: int = 5) -> List[MemorySearchResult]:
        query_embedding = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for i in range(len(indices[0])):
            index_pos = str(indices[0][i])
            if index_pos in self.metadata:
                meta = self.metadata[index_pos]
                result_with_dist = {**meta, "distance": float(distances[0][i])}
                results.append(MemorySearchResult.model_validate(result_with_dist))
        return results

    def _store_dir(self, tenant: str, project_id: str) -> str:
        """Get storage directory for tenant/project."""
        base = os.path.join('data', tenant, project_id or 'default')
        os.makedirs(base, exist_ok=True)
        return base

    def _dated_file(self, base: str, layer: str, prefix: str) -> str:
        """Get dated file path for layer."""
        date = datetime.datetime.utcnow().strftime('%Y%m%d')
        return os.path.join(base, layer, f'{prefix}_{date}.jsonl')

    def _append_audit_jsonl(self, path: str, record: dict):
        """Append record to JSONL file with audit."""
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

    def _append_jsonl(self, path: str, record: dict):
        """Append record to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a') as f:
            json.dump(record, f)
            f.write('\n')

# Singleton instance
memory_service = MemoryService()

# Layer-specific service functions

def add_episodic(memory_service: MemoryService, tenant: str, project_id: str, content: str, salience: int) -> dict:
    """Add episodic memory with salience."""
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._dated_file(base, 'episodic', 'episodic')
    rec = {
        'content': content,
        'salience': salience,
        'project_id': project_id,
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
