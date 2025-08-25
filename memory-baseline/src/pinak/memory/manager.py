
import json
import os
import uuid
import datetime
import numpy as np
import faiss
import redis
from sentence_transformers import SentenceTransformer

class MemoryManager:
    """A robust, local-first memory management system for AI agents."""

    def __init__(self, config_path=None):
        """Initializes the MemoryManager by loading configuration and resources."""
        if config_path is None:
            # Build a path relative to this file: src/../config/memory_config.json
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_dir, '..', 'config', 'memory_config.json')

        """Initializes the MemoryManager by loading configuration and resources."""
        print("Initializing MemoryManager...")
        self.config = self._load_config(config_path)
        self.model = self._load_embedding_model()
        self.redis_client = self._connect_to_redis()
        self.index, self.metadata = self._load_or_create_vector_db()
        self._ensure_data_directory_exists()
        print("MemoryManager initialized successfully.")

    def _load_config(self, path):
        """Loads configuration from a JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise Exception(f"Configuration file not found at {path}. Aborting.")

    def _ensure_data_directory_exists(self):
        """Ensures the data directory from the config exists."""
        data_dir = os.path.dirname(self.config['vector_db_path'])
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

    def _load_embedding_model(self):
        """Loads the SentenceTransformer model."""
        print(f"Loading embedding model: {self.config['embedding_model']}...")
        return SentenceTransformer(self.config['embedding_model'])

    def _connect_to_redis(self):
        """Connects to the local Redis server for working memory."""
        try:
            client = redis.Redis(
                host=self.config['redis_host'],
                port=self.config['redis_port'],
                decode_responses=True
            )
            client.ping()
            print("Successfully connected to Redis.")
            return client
        except redis.exceptions.ConnectionError as e:
            raise Exception(f"Could not connect to Redis at {self.config['redis_host']}:{self.config['redis_port']}. Please ensure Redis is running. Error: {e}")

    def _load_or_create_vector_db(self):
        """Loads the FAISS index and metadata from disk, or creates them if they don't exist."""
        db_path = self.config['vector_db_path']
        meta_path = self.config['metadata_db_path']
        
        if os.path.exists(db_path) and os.path.exists(meta_path):
            print(f"Loading existing vector database from {db_path}...")
            index = faiss.read_index(db_path)
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print("Vector database loaded.")
        else:
            print("No existing vector database found. Creating a new one.")
            embedding_dim = self.model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(embedding_dim)
            metadata = {}
        return index, metadata

    def _save_vector_db(self):
        """Saves the FAISS index and metadata to disk."""
        faiss.write_index(self.index, self.config['vector_db_path'])
        with open(self.config['metadata_db_path'], 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _log_change(self, event_type, memory_id, content):
        """Logs an event to the changelog file."""
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": event_type,
            "memory_id": memory_id,
            "content": content
        }
        with open(self.config['changelog_path'], 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def add_memory(self, content: str, tags: list = None):
        """Adds a new memory to the system."""
        if not isinstance(content, str) or not content:
            raise ValueError("Memory content must be a non-empty string.")

        memory_id = str(uuid.uuid4())
        embedding = self.model.encode([content])[0].astype('float32')
        
        self.index.add(np.array([embedding]))
        
        self.metadata[str(self.index.ntotal - 1)] = {
            "id": memory_id,
            "content": content,
            "tags": tags or [],
            "created_at": datetime.datetime.utcnow().isoformat(),
            "deleted": False
        }
        
        self._save_vector_db()
        self._log_change("ADD", memory_id, content)
        print(f"Added new memory: {memory_id}")
        return memory_id

    def retrieve_memory(self, query: str, k: int = 5):
        """Retrieves the k most relevant memories for a given query."""
        if not isinstance(query, str) or not query:
            raise ValueError("Query must be a non-empty string.")

        query_embedding = self.model.encode([query])[0].astype('float32')
        distances, indices = self.index.search(np.array([query_embedding]), k * 2) # Retrieve more to filter out deleted
        
        results = []
        for i in range(len(indices[0])):
            if len(results) >= k:
                break
            
            index_pos = str(indices[0][i])
            if index_pos in self.metadata and not self.metadata[index_pos].get('deleted', False):
                result = self.metadata[index_pos]
                result['distance'] = float(distances[0][i])
                results.append(result)
                
        return results

    def delete_memory(self, memory_id: str):
        """(Guardrail) Soft-deletes a memory by its UUID. It is not removed from the index but is flagged and excluded from retrieval."""
        found = False
        for index_pos, meta in self.metadata.items():
            if meta.get('id') == memory_id:
                if meta.get('deleted') is True:
                    print(f"Memory {memory_id} is already deleted.")
                    return True # Idempotent
                
                self.metadata[index_pos]['deleted'] = True
                self._save_vector_db()
                self._log_change("DELETE_SOFT", memory_id, meta['content'])
                print(f"Soft-deleted memory: {memory_id}")
                found = True
                break
        if not found:
            print(f"Could not find memory with ID: {memory_id}")
        return found

    def purge_deleted_memories(self):
        """(Robustness) Permanently removes all soft-deleted memories from the database. This is an expensive operation and should be used sparingly."""
        print("Starting memory purge...")
        
        # Identify indices to keep
        indices_to_keep = [int(idx) for idx, meta in self.metadata.items() if not meta.get('deleted')]
        if len(indices_to_keep) == self.index.ntotal:
            print("No soft-deleted memories to purge.")
            return

        # Create a new index with only the vectors to keep
        new_index = faiss.IndexFlatL2(self.index.d)
        new_metadata = {}
        
        # Reconstruct the index and metadata
        for new_idx, old_idx in enumerate(indices_to_keep):
            vector = self.index.reconstruct(old_idx)
            new_index.add(np.array([vector]))
            meta = self.metadata[str(old_idx)]
            new_metadata[str(new_idx)] = meta

        # Replace old with new
        self.index = new_index
        self.metadata = new_metadata
        self._save_vector_db()
        self._log_change("PURGE", "all_deleted", f"Removed {len(indices_to_keep)} memories.")
        print(f"Purge complete. New memory count: {self.index.ntotal}")

    def set_working_memory(self, key: str, value: str):
        """Sets a key-value pair in the working memory (Redis)."""
        self.redis_client.set(key, value)

    def get_working_memory(self, key: str):
        """Gets a value from working memory."""
        return self.redis_client.get(key)
