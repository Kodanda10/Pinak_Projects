import json
import os
import re
import uuid
import hashlib
import datetime
import shutil
import glob
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer
from filelock import FileLock

from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult


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
                # Use itertools.cycle for efficient padding
                digest = bytes(itertools.islice(itertools.cycle(digest), needed_bytes))
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
        lock_path = vector_path + ".lock"

        with FileLock(lock_path):
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

        vector_path = store["vector_path"]
        metadata_path = store["metadata_path"]
        lock_path = vector_path + ".lock"

        with FileLock(lock_path):
            # Atomic write for FAISS index
            tmp_vector_path = vector_path + ".tmp"
            faiss.write_index(store["index"], tmp_vector_path)
            shutil.move(tmp_vector_path, vector_path)

            # Atomic write for metadata
            tmp_metadata_path = metadata_path + ".tmp"
            with open(tmp_metadata_path, "w", encoding="utf-8") as fh:
                json.dump(store["metadata"], fh, indent=2)
            shutil.move(tmp_metadata_path, metadata_path)

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
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
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
        date = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d')
        layer_dir = os.path.join(base, layer)
        os.makedirs(layer_dir, exist_ok=True)
        return os.path.join(layer_dir, f'{prefix}_{date}.jsonl')

    def _last_audit_record(self, path: str) -> Optional[dict]:
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as fh:
                fh.seek(0, os.SEEK_END)
                position = fh.tell()
                buffer = b''
                while position > 0:
                    read_size = min(4096, position)
                    position -= read_size
                    fh.seek(position)
                    buffer = fh.read(read_size) + buffer
                    lines = buffer.split(b'\n')
                    # If we have more than one line, process from the end
                    for line in reversed(lines):
                        line = line.strip()
                        if line:
                            try:
                                return json.loads(line.decode('utf-8'))
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                continue
                    # If not enough lines, continue reading backwards
                    buffer = lines[0]  # keep the partial first line
                # If we reach here, try the remaining buffer
                line = buffer.strip()
                if line:
                    try:
                        return json.loads(line.decode('utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
        except Exception:
            pass
        return None

    def _compute_audit_hash(self, payload: dict) -> str:
        to_hash = {k: v for k, v in payload.items() if k != 'hash'}
        serialized = json.dumps(to_hash, sort_keys=True, separators=(',', ':')).encode('utf-8')
        return hashlib.sha256(serialized).hexdigest()

    def _append_audit_jsonl(self, path: str, record: dict) -> dict:
        """Append record to JSONL file with tamper-evident hashing."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            prev_entry = self._last_audit_record(path)
            prev_hash = prev_entry.get('hash') if prev_entry else None
            payload = {**record, 'prev_hash': prev_hash}
            payload['hash'] = self._compute_audit_hash(payload)
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(payload, f, sort_keys=True)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            return payload

    def _session_file(self, base: str, session_id: str) -> str:
        """Get session file path."""
        safe_id = self._sanitize_component(session_id)
        return os.path.join(base, 'session', f'session_{safe_id}.jsonl')

    def _working_file(self, base: str) -> str:
        """Get working file path."""
        date = datetime.datetime.now(datetime.UTC).strftime('%Y%m%d')
        return os.path.join(base, 'working', f'working_{date}.jsonl')

    def _append_jsonl(self, path: str, record: dict):
        """Append record to JSONL file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lock_path = path + ".lock"
        with FileLock(lock_path):
            with open(path, 'a', encoding='utf-8') as f:
                json.dump(record, f, sort_keys=True)
                f.write('\n')

    @staticmethod
    def parse_ts(ts: str) -> Optional[datetime.datetime]:
        if not ts:
            return None
        try:
            dt = datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.UTC)
            return dt
        except Exception:
            return None

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """Read all records from a JSONL file."""
        if not os.path.exists(path):
            return []

        # We don't necessarily need a lock for reading unless strict consistency is required.
        # But for robustness, we can use a shared lock if FileLock supported it (it's exclusive).
        # We'll skip lock for read for performance, acknowledging slight race.
        out = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass # Graceful failure
        return out

    def get_layer_files(self, base: str, layer: str, prefix: str) -> List[str]:
        folder = os.path.join(base, layer)
        return sorted(glob.glob(os.path.join(folder, f'{prefix}_*.jsonl')))

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
        'ts': datetime.datetime.now(datetime.UTC).isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_episodic(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list:
    """List episodic memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    for fp in memory_service.get_layer_files(base, 'episodic', 'episodic'):
        for obj in memory_service._read_jsonl(fp):
            ts = memory_service.parse_ts(obj.get('ts', ''))
            if t_since and ts and ts < t_since:
                continue
            if t_until and ts and ts > t_until:
                continue
            out.append(obj)
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
        'ts': datetime.datetime.now(datetime.UTC).isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_procedural(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list:
    """List procedural memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    for fp in memory_service.get_layer_files(base, 'procedural', 'procedural'):
        for obj in memory_service._read_jsonl(fp):
            ts = memory_service.parse_ts(obj.get('ts', ''))
            if t_since and ts and ts < t_since:
                continue
            if t_until and ts and ts > t_until:
                continue
            out.append(obj)
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
        'ts': datetime.datetime.now(datetime.UTC).isoformat(),
    }
    memory_service._append_jsonl(path, rec)
    return rec

def list_rag(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> list:
    """List RAG memories."""
    base = memory_service._store_dir(tenant, project_id)
    out = []
    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    for fp in memory_service.get_layer_files(base, 'rag', 'rag'):
        for obj in memory_service._read_jsonl(fp):
            ts = memory_service.parse_ts(obj.get('ts', ''))
            if t_since and ts and ts < t_since:
                continue
            if t_until and ts and ts > t_until:
                continue
            out.append(obj)
    return out

def search_v2(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    query: str,
    layer_list: list,
    since: Optional[str] = None,
    until: Optional[str] = None
) -> dict:
    """Search across multiple layers."""
    results = {}
    query_lower = query.lower()

    if 'episodic' in layer_list:
        epi_results = list_episodic(memory_service, tenant, project_id, since, until)
        results['episodic'] = [r for r in epi_results if query_lower in r.get('content', '').lower()][:5]
    if 'procedural' in layer_list:
        proc_results = list_procedural(memory_service, tenant, project_id, since, until)
        results['procedural'] = [r for r in proc_results if query_lower in r.get('skill_id', '').lower()][:5]
    if 'rag' in layer_list:
        rag_results = list_rag(memory_service, tenant, project_id, since, until)
        results['rag'] = [r for r in rag_results if query_lower in r.get('query', '').lower()][:5]
    return results

def add_event(memory_service: MemoryService, tenant: str, project_id: str, payload: dict) -> dict:
    base = memory_service._store_dir(tenant, project_id)
    event_payload = {
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        **payload,
        "tenant": tenant,
        "project_id": project_id,
    }
    ep = memory_service._dated_file(base, 'events', 'events')
    memory_service._append_audit_jsonl(ep, event_payload)
    return {"status": "ok"}

def list_events(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    q: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    base = memory_service._store_dir(tenant, project_id)
    out = []
    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    for fp in memory_service.get_layer_files(base, 'events', 'events'):
        for obj in memory_service._read_jsonl(fp):
            if q and q not in json.dumps(obj):
                continue
            ts = memory_service.parse_ts(obj.get('ts', ''))
            if t_since and ts and ts < t_since:
                continue
            if t_until and ts and ts > t_until:
                continue
            out.append(obj)
    return out[offset:offset+limit]

def add_session(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    session_id: str,
    content: str,
    ttl: Optional[int] = None,
    expires_at: Optional[str] = None,
    ts: Optional[str] = None
) -> dict:
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._session_file(base, session_id)
    rec = {
        'session_id': session_id,
        'content': content,
        'project_id': project_id,
        'tenant': tenant,
        'ts': ts or datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if ttl:
        rec['expires_at'] = (datetime.datetime.now(datetime.UTC)+datetime.timedelta(seconds=int(ttl))).isoformat()
    if expires_at:
        rec['expires_at'] = expires_at
    memory_service._append_jsonl(path, rec)
    return {'status': 'ok'}

def list_session(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._session_file(base, session_id)
    out = []

    # Check legacy path too (as per original code)
    if not os.path.exists(path):
        legacy = os.path.join(base, f'session_{session_id}.jsonl')
        if os.path.exists(legacy):
            path = legacy

    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    now = datetime.datetime.now(datetime.UTC)

    for obj in memory_service._read_jsonl(path):
        exp = obj.get('expires_at')
        if exp:
            try:
                if datetime.datetime.fromisoformat(exp) < now:
                    continue
            except Exception:
                pass

        ts = memory_service.parse_ts(obj.get('ts', ''))
        if t_since and ts and ts < t_since:
            continue
        if t_until and ts and ts > t_until:
            continue
        out.append(obj)

    return out[offset:offset+limit]

def add_working(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    content: str,
    ttl: Optional[int] = None,
    expires_at: Optional[str] = None,
    ts: Optional[str] = None
) -> dict:
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._working_file(base)
    rec = {
        'content': content,
        'project_id': project_id,
        'tenant': tenant,
        'ts': ts or datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if ttl:
        rec['expires_at'] = (datetime.datetime.now(datetime.UTC)+datetime.timedelta(seconds=int(ttl))).isoformat()
    if expires_at:
        rec['expires_at'] = expires_at
    memory_service._append_jsonl(path, rec)
    return {'status': 'ok'}

def list_working(
    memory_service: MemoryService,
    tenant: str,
    project_id: str,
    limit: int = 100,
    offset: int = 0,
    since: Optional[str] = None,
    until: Optional[str] = None,
) -> List[Dict[str, Any]]:
    base = memory_service._store_dir(tenant, project_id)
    path = memory_service._working_file(base)
    out = []

    t_since = memory_service.parse_ts(since) if since else None
    t_until = memory_service.parse_ts(until) if until else None

    now = datetime.datetime.now(datetime.UTC)

    for obj in memory_service._read_jsonl(path):
        exp = obj.get('expires_at')
        if exp:
            try:
                if datetime.datetime.fromisoformat(exp) < now:
                    continue
            except Exception:
                pass

        ts = memory_service.parse_ts(obj.get('ts', ''))
        if t_since and ts and ts < t_since:
            continue
        if t_until and ts and ts > t_until:
            continue
        out.append(obj)

    return out[offset:offset+limit]
