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
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        # Internal buffers with initial capacity
        self._capacity = 1024
        self._size = 0
        self._vectors_buffer = np.empty((self._capacity, dimension), dtype=np.float32)
        self._ids_buffer = np.empty(self._capacity, dtype=np.int64)
        self._norms_buffer = np.empty(self._capacity, dtype=np.float32)
        
        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    def _resize(self, new_capacity: int):
        """Resize internal buffers to new capacity."""
        if new_capacity <= self._capacity:
            return

        # Double capacity or match request
        target_capacity = max(self._capacity * 2, new_capacity)

        new_vectors = np.empty((target_capacity, self.dimension), dtype=np.float32)
        new_ids = np.empty(target_capacity, dtype=np.int64)
        new_norms = np.empty(target_capacity, dtype=np.float32)

        # Copy existing data
        if self._size > 0:
            new_vectors[:self._size] = self._vectors_buffer[:self._size]
            new_ids[:self._size] = self._ids_buffer[:self._size]
            new_norms[:self._size] = self._norms_buffer[:self._size]

        self._vectors_buffer = new_vectors
        self._ids_buffer = new_ids
        self._norms_buffer = new_norms
        self._capacity = target_capacity

    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        return self._size

    @property
    def vectors(self):
        return self._vectors_buffer[:self._size]

    @vectors.setter
    def vectors(self, value):
        val = np.array(value, dtype=np.float32)
        if val.ndim == 1:
            val = val.reshape(1, -1)
        # Handle empty case
        if val.size == 0:
            self._size = 0
            return

        count = val.shape[0]
        if count > self._capacity:
            self._resize(count)

        self._vectors_buffer[:count] = val
        self._size = count

    @property
    def ids(self):
        return self._ids_buffer[:self._size]

    @ids.setter
    def ids(self, value):
        val = np.array(value, dtype=np.int64)
        count = val.shape[0]
        if count > self._capacity:
            self._resize(count)
        self._ids_buffer[:count] = val

    @property
    def norms(self):
        return self._norms_buffer[:self._size]

    @norms.setter
    def norms(self, value):
        val = np.array(value, dtype=np.float32)
        count = val.shape[0]
        if count > self._capacity:
            self._resize(count)
        self._norms_buffer[:count] = val

    def _load_index(self):
        """Loads vectors and IDs from a numpy file."""
        with self.lock:
            load_path = None
            if os.path.exists(self.index_path):
                load_path = self.index_path
            elif os.path.exists(f"{self.index_path}.npy"):
                load_path = f"{self.index_path}.npy"

            self._size = 0

            if load_path:
                try:
                    data = np.load(load_path, allow_pickle=True).item()
                    vectors = data['vectors'].astype(np.float32)
                    ids = data['ids'].astype(np.int64)

                    count = len(ids)
                    if count > self._capacity:
                        self._resize(count)

                    self._vectors_buffer[:count] = vectors
                    self._ids_buffer[:count] = ids
                    self._norms_buffer[:count] = np.sum(np.square(vectors), axis=1)
                    self._size = count

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {count}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self._size = 0

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
                    np.save(handle, {'vectors': self.vectors, 'ids': self.ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {len(self.ids)}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        count = len(ids)

        with self.lock:
            if self._size + count > self._capacity:
                self._resize(self._size + count)

            self._vectors_buffer[self._size : self._size + count] = vectors
            self._ids_buffer[self._size : self._size + count] = id_array
            self._norms_buffer[self._size : self._size + count] = new_norms

            self._size += count
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if len(self.ids) == 0:
                return [], []

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dot_product = np.dot(self.vectors, query_vector.T).flatten()
            query_norm_sq = float(np.sum(np.square(query_vector)))
            sq_dists = self.norms + query_norm_sq - (2.0 * dot_product)
            sq_dists = np.maximum(sq_dists, 0.0)

            # Get top K indices
            actual_k = min(k, len(self.ids))
            if actual_k < len(self.ids):
                top_k_partition = np.argpartition(sq_dists, actual_k - 1)[:actual_k]
                sorted_idx_in_top_k = np.argsort(sq_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(sq_dists)

            # Return in FAISS compatibility format (2D arrays)
            return (
                [float(d) for d in sq_dists[top_k_idx].tolist()],
                [int(i) for i in self.ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            # Operate on valid data slice
            valid_ids = self._ids_buffer[:self._size]
            mask = ~np.isin(valid_ids, ids)

            # If nothing to remove, return early
            if np.all(mask):
                return

            # Count remaining items
            new_count = np.count_nonzero(mask)

            # Copy kept elements
            new_vectors = self._vectors_buffer[:self._size][mask]
            new_ids = valid_ids[mask]
            new_norms = self._norms_buffer[:self._size][mask]

            # Place back at start of buffer
            self._vectors_buffer[:new_count] = new_vectors
            self._ids_buffer[:new_count] = new_ids
            self._norms_buffer[:new_count] = new_norms

            self._size = new_count
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self._size

    def reset(self):
        with self.lock:
            self._size = 0
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            matches = np.where(self.ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
