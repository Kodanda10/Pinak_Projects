
import json
import os
import uuid
import datetime
import numpy as np
import faiss
import redis
from typing import List
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # optional in tests
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult

class MemoryService:
    """The core logic for the memory service, handling vector search and storage."""

    def __init__(self, config_path='app/core/config.json'):
        print("Initializing MemoryService with real logic...")
        self.config = self._load_config(config_path)
        self._ensure_data_directory()
        # Allow mock embeddings in test/CI to avoid heavy model downloads
        if os.getenv('USE_MOCK_EMBEDDINGS', '').lower() in {'1','true','yes','on'}:
            class _MockModel:
                def get_sentence_embedding_dimension(self):
                    return 384

                def encode(self, texts):
                    return np.zeros((len(texts), 384), dtype='float32')
            self.model = _MockModel()
        else:
            if SentenceTransformer is None:
                raise RuntimeError('sentence-transformers not available and USE_MOCK_EMBEDDINGS is false')
            self.model = SentenceTransformer(self.config['embedding_model'])
        self.index, self.metadata = self._load_or_create_vector_db()
        self.redis_client = self._connect_to_redis()
        print("MemoryService initialized.")

    def _connect_to_redis(self):
        """Connects to the Redis server, prioritizing environment variables. Best-effort in dev/tests."""
        redis_host = os.getenv("REDIS_HOST", self.config.get("redis_host", "localhost"))
        redis_port = int(os.getenv("REDIS_PORT", self.config.get("redis_port", 6379)))
        
        try:
            client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            client.ping()
            print(f"Successfully connected to Redis at {redis_host}:{redis_port}.")
            return client
        except Exception as e:
            print(f"Warning: Redis unavailable at {redis_host}:{redis_port}. Proceeding without Redis. Error: {e}")
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

    # Storage helpers per tenant/project
    def _store_dir(self, tenant: str, project_id: object) -> str:
        base = os.path.join(os.path.dirname(self.config['vector_db_path']), 'tenants', tenant or 'default', str(project_id or 'default'))
        os.makedirs(base, exist_ok=True)
        return base

    def _append_jsonl(self, path: str, obj: dict) -> None:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(obj) + "\n")

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
        # Append changelog (append-only)
        try:
            tenant = 'default'
            proj = 'default'
            base = self._store_dir(tenant, proj)
            self._append_jsonl(os.path.join(base, 'changelog.jsonl'), {
                'change_type':'create','target_id': new_meta['id'], 'ts': datetime.datetime.utcnow().isoformat(), 'reason': 'add_memory'
            })
        except Exception:
            pass
        return MemoryRead.model_validate(new_meta)

    def search_memory(self, query: str, k: int = 5) -> List[MemorySearchResult]:
        query_embedding = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k)
        
        results = []
        for i in range(len(indices[0])):
            index_pos = str(indices[0][i])
            if index_pos in self.metadata:
                meta = self.metadata[index_pos]
                if meta.get('redacted'):
                    continue
                result_with_dist = {**meta, "distance": float(distances[0][i])}
                results.append(MemorySearchResult.model_validate(result_with_dist))
        return results

    # Redact/tombstone support
    def redact_memory(self, memory_id: str, tenant: str, project_id: object, reason: str = 'redact') -> dict:
        # Mark redacted in metadata
        for k, v in self.metadata.items():
            if isinstance(v, dict) and v.get('id') == memory_id:
                v['redacted'] = True
                self._save_vector_db()
                break
        base = self._store_dir(tenant, project_id)
        self._append_jsonl(os.path.join(base, 'changelog.jsonl'), {
            'change_type':'redact','target_id': memory_id, 'ts': datetime.datetime.utcnow().isoformat(), 'reason': reason
        })
        return {'status':'ok','redacted_id': memory_id}

    # Changelog reader
    def list_changelog(self, tenant: str, project_id: object, change_type=None, since=None, until=None, limit: int = 100, offset: int = 0) -> list:
        import json, datetime
        base = self._store_dir(tenant or 'default', project_id)
        p = os.path.join(base, 'changelog.jsonl')
        out=[]
        def parse_ts(ts: str):
            try:
                return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
            except Exception:
                return None
        t_since = parse_ts(since) if since else None
        t_until = parse_ts(until) if until else None
        if os.path.exists(p):
            with open(p,'r',encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if change_type and obj.get('change_type') != change_type:
                            continue
                        ts = parse_ts(obj.get('ts',''))
                        if t_since and ts and ts < t_since:
                            continue
                        if t_until and ts and ts > t_until:
                            continue
                        out.append(obj)
                    except Exception:
                        pass
        return out[offset:offset+limit]

    # Singleton instance
memory_service = MemoryService()

def add_episodic(self, tenant: str, project_id: object, content: str, salience: int = 0) -> dict:
    key = (tenant or 'default', project_id or 'default')
    store = getattr(self, '_episodic_store', {})
    rec = {"content": content, "project_id": project_id, "salience": salience}
    store.setdefault(key, []).append(rec)
    self._episodic_store = store
    return rec

def list_episodic(self, tenant: str, project_id: object) -> list:
    store = getattr(self, '_episodic_store', {})
    return list(store.get((tenant or 'default', project_id or 'default'), []))

def add_procedural(self, tenant: str, project_id: object, skill_id: str, steps=None) -> dict:
    key = (tenant or 'default', project_id or 'default')
    store = getattr(self, '_procedural_store', {})
    rec = {"skill_id": skill_id, "steps": steps or [], "project_id": project_id}
    store.setdefault(key, []).append(rec)
    self._procedural_store = store
    return rec

def list_procedural(self, tenant: str, project_id: object):
    store = getattr(self, '_procedural_store', {})
    return list(store.get((tenant or 'default', project_id or 'default'), []))

def add_rag(self, tenant: str, project_id: object, query: str, external_source=None) -> dict:
    key = (tenant or 'default', project_id or 'default')
    store = getattr(self, '_rag_store', {})
    rec = {"query": query, "external_source": external_source, "project_id": project_id}
    store.setdefault(key, []).append(rec)
    self._rag_store = store
    return rec

def list_rag(self, tenant: str, project_id: object):
    store = getattr(self, '_rag_store', {})
    return list(store.get((tenant or 'default', project_id or 'default'), []))

def search_v2(self, tenant: str, project_id: object, query: str, layers: list[str]) -> dict:
    out: dict[str, list] = {}
    if 'semantic' in layers:
        out['semantic'] = [m for m in self.search_memory(query, k=20)]
    if 'episodic' in layers:
        out['episodic'] = [r for r in list_episodic(self, tenant, project_id) if query.lower() in (r.get('content','').lower())]
    if 'procedural' in layers:
        out['procedural'] = [r for r in list_procedural(self, tenant, project_id) if query.lower() in (' '.join(r.get('steps',[])).lower())]
    if 'rag' in layers:
        out['rag'] = [r for r in list_rag(self, tenant, project_id) if query.lower() in (r.get('query','').lower())]
    return out
