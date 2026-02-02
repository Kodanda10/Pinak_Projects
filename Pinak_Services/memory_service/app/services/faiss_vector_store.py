import os
import numpy as np
import threading
import logging
import faiss
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class FaissVectorStore:
    """
    Thread-safe Vector Store using FAISS for similarity search.
    Enterprise-grade scalability using IndexIDMap + IndexFlatL2.
    """
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()

        self.index = None
        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    @property
    def ntotal(self):
        with self.lock:
            return self.index.ntotal if self.index else 0

    @property
    def total(self):
        return self.ntotal

    def _load_index(self):
        """Loads FAISS index from disk or creates a new one."""
        with self.lock:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded FAISS Index from {self.index_path}. Size: {self.index.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to load FAISS index: {e}. Creating new one.")
                    self._create_new_index()
            else:
                self._create_new_index()

    def _create_new_index(self):
        # IndexIDMap allows add_with_ids
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def _schedule_save(self):
        """Schedule a debounced save."""
        if self._save_timer is not None:
            self._save_timer.cancel()

        self._save_timer = threading.Timer(self._save_interval, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def save(self):
        """Synchronously save to disk."""
        with self.lock:
            if self.needs_save and self.index:
                dirpath = os.path.dirname(self.index_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                self.needs_save = False
                logger.info(f"Saved FAISS Index to {self.index_path}. Size: {self.index.ntotal}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        id_array = np.array(ids, dtype=np.int64)

        with self.lock:
            self.index.add_with_ids(vectors, id_array)
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if self.index.ntotal == 0:
                return [], []

            query_vector = np.ascontiguousarray(query_vector.astype(np.float32))
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            # FAISS returns distances (L2 squared) and indices
            distances, indices = self.index.search(query_vector, k)

            # Flatten results for single query
            return (
                distances[0].tolist(),
                indices[0].tolist(),
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            id_array = np.array(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            self.needs_save = True
        self._schedule_save()

    def reset(self):
        with self.lock:
            self.index.reset()
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            try:
                # FAISS can throw runtime error if ID not found in IDMap
                vec = self.index.reconstruct(vector_id)
                return np.array(vec, dtype=np.float32)
            except RuntimeError:
                return None
            except Exception as e:
                logger.warning(f"Failed to reconstruct vector {vector_id}: {e}")
                return None

    @contextmanager
    def batch_add(self):
        # For FAISS, we can just save after the block
        # Could optimize by disabling auto-save during batch
        old_interval = self._save_interval
        self._save_interval = 300 # Pause auto-save
        try:
            yield
        finally:
            self._save_interval = old_interval
            self.save()
