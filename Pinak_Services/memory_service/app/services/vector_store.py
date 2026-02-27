import os
import numpy as np
import threading
import logging
import time
import json
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Thread-safe Vector Store using Numpy for similarity search.
    Provides identical API to the previous FAISS implementation but without segfaults.
    """
    def __init__(self, index_path: str, dimension: int, initial_capacity: int = 1024):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        # In-memory storage with dynamic array resizing (amortized O(1))
        self.capacity = initial_capacity
        self.size = 0
        self.vectors = np.empty((self.capacity, dimension), dtype=np.float32)
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.norms = np.empty(self.capacity, dtype=np.float32)
        
        self._save_timer = None
        self._load_index()

        self._save_interval = 5.0  # seconds
        self.needs_save = False

    def _resize(self, min_capacity: int):
        """Internal method to resize arrays when capacity is exceeded."""
        new_capacity = max(self.capacity * 2, min_capacity)

        new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_vectors[:self.size] = self.vectors[:self.size]
        self.vectors = new_vectors

        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_ids[:self.size] = self.ids[:self.size]
        self.ids = new_ids

        new_norms = np.empty(new_capacity, dtype=np.float32)
        new_norms[:self.size] = self.norms[:self.size]
        self.norms = new_norms

        self.capacity = new_capacity

    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        return self.size

    def _load_index(self):
        """Loads vectors and IDs from a numpy file."""
        with self.lock:
            load_path = None
            if os.path.exists(self.index_path):
                load_path = self.index_path
            elif os.path.exists(f"{self.index_path}.npy"):
                load_path = f"{self.index_path}.npy"
            if load_path:
                try:
                    data = np.load(load_path, allow_pickle=True).item()
                    loaded_vectors = data['vectors'].astype(np.float32)
                    loaded_ids = data['ids'].astype(np.int64)
                    self.size = len(loaded_ids)
                    self.capacity = max(self.capacity, self.size)
                    self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
                    self.ids = np.empty(self.capacity, dtype=np.int64)
                    self.norms = np.empty(self.capacity, dtype=np.float32)

                    self.vectors[:self.size] = loaded_vectors
                    self.ids[:self.size] = loaded_ids
                    self.norms[:self.size] = np.sum(np.square(loaded_vectors), axis=1)
                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self.size}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.size = 0
                    self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
                    self.ids = np.empty(self.capacity, dtype=np.int64)
                    self.norms = np.empty(self.capacity, dtype=np.float32)

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
            if self.needs_save:
                dirpath = os.path.dirname(self.index_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(self.index_path, "wb") as handle:
                    np.save(handle, {'vectors': self.vectors[:self.size], 'ids': self.ids[:self.size]})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self.size}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        num_new = vectors.shape[0]
        if num_new != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            if self.size + num_new > self.capacity:
                self._resize(self.size + num_new)

            end_idx = self.size + num_new
            self.vectors[self.size:end_idx] = vectors
            self.ids[self.size:end_idx] = id_array
            self.norms[self.size:end_idx] = new_norms
            self.size = end_idx
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if self.size == 0:
                return [], []

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            active_vectors = self.vectors[:self.size]
            active_norms = self.norms[:self.size]
            active_ids = self.ids[:self.size]

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dot_product = np.dot(active_vectors, query_vector.T).flatten()
            query_norm_sq = float(np.sum(np.square(query_vector)))
            sq_dists = active_norms + query_norm_sq - (2.0 * dot_product)
            sq_dists = np.maximum(sq_dists, 0.0)

            # Get top K indices
            actual_k = min(k, self.size)
            if actual_k < self.size:
                top_k_partition = np.argpartition(sq_dists, actual_k - 1)[:actual_k]
                sorted_idx_in_top_k = np.argsort(sq_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(sq_dists)

            # Return in FAISS compatibility format (2D arrays)
            return (
                [float(d) for d in sq_dists[top_k_idx].tolist()],
                [int(i) for i in active_ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            if self.size == 0:
                return
            mask = ~np.isin(self.ids[:self.size], ids)
            new_size = np.sum(mask)
            if new_size < self.size:
                self.vectors[:new_size] = self.vectors[:self.size][mask]
                self.ids[:new_size] = self.ids[:self.size][mask]
                self.norms[:new_size] = self.norms[:self.size][mask]
                self.size = new_size
                self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self.size

    def reset(self):
        with self.lock:
            self.size = 0
            self.capacity = max(1024, self.capacity)
            self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
            self.ids = np.empty(self.capacity, dtype=np.int64)
            self.norms = np.empty(self.capacity, dtype=np.float32)
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            matches = np.where(self.ids[:self.size] == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
