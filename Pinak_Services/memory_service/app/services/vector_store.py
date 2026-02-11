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
        
        # In-memory storage
        self._size = 0
        self.capacity = 1000  # Initial capacity
        self.vectors = np.empty((self.capacity, dimension), dtype=np.float32)
        self.ids = np.empty(self.capacity, dtype=np.int64)
        self.norms = np.empty(self.capacity, dtype=np.float32)
        
        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        return self._size

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

                    count = len(loaded_ids)
                    self._size = count

                    # Ensure capacity covers loaded data
                    if count > self.capacity:
                        self.capacity = max(count * 2, 1000)
                        self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
                        self.ids = np.empty(self.capacity, dtype=np.int64)
                        self.norms = np.empty(self.capacity, dtype=np.float32)

                    # Copy data into pre-allocated arrays
                    self.vectors[:count] = loaded_vectors
                    self.ids[:count] = loaded_ids
                    self.norms[:count] = np.sum(np.square(loaded_vectors), axis=1)

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self._size}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self._size = 0
                    # Capacity arrays are already initialized in __init__

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
                    # Save only the active slice of data
                    np.save(handle, {
                        'vectors': self.vectors[:self._size],
                        'ids': self.ids[:self._size]
                    })
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self._size}")

    def _expand_capacity(self, min_capacity):
        """Expand the capacity of the internal arrays."""
        new_capacity = max(self.capacity * 2, min_capacity)

        new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_norms = np.empty(new_capacity, dtype=np.float32)

        # Copy existing data
        if self._size > 0:
            new_vectors[:self._size] = self.vectors[:self._size]
            new_ids[:self._size] = self.ids[:self._size]
            new_norms[:self._size] = self.norms[:self._size]

        self.vectors = new_vectors
        self.ids = new_ids
        self.norms = new_norms
        self.capacity = new_capacity

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

        n = len(ids)

        with self.lock:
            new_size = self._size + n
            if new_size > self.capacity:
                self._expand_capacity(new_size)

            self.vectors[self._size : new_size] = vectors
            self.ids[self._size : new_size] = id_array
            self.norms[self._size : new_size] = new_norms

            self._size = new_size
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if self._size == 0:
                return [], []

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            # Compute L2 distance using dot product on active slice
            # ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            active_vectors = self.vectors[:self._size]
            active_norms = self.norms[:self._size]

            dot_product = np.dot(active_vectors, query_vector.T).flatten()
            query_norm_sq = float(np.sum(np.square(query_vector)))
            sq_dists = active_norms + query_norm_sq - (2.0 * dot_product)
            sq_dists = np.maximum(sq_dists, 0.0)

            # Get top K indices
            actual_k = min(k, self._size)
            if actual_k < self._size:
                top_k_partition = np.argpartition(sq_dists, actual_k - 1)[:actual_k]
                sorted_idx_in_top_k = np.argsort(sq_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(sq_dists)

            # Return in FAISS compatibility format (2D arrays)
            # Map back to original IDs using self.ids
            result_ids = self.ids[:self._size][top_k_idx].tolist()
            result_dists = sq_dists[top_k_idx].tolist()

            return (
                [float(d) for d in result_dists],
                [int(i) for i in result_ids],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            # Identify which indices to keep from the active part
            # np.isin returns a boolean mask of same shape as first argument
            active_ids = self.ids[:self._size]
            keep_mask = ~np.isin(active_ids, ids)

            # If we are keeping everything, do nothing
            new_count = np.sum(keep_mask)
            if new_count == self._size:
                return

            # Compact the arrays in place
            # Copy kept elements to the beginning of arrays
            self.vectors[:new_count] = self.vectors[:self._size][keep_mask]
            self.ids[:new_count] = self.ids[:self._size][keep_mask]
            self.norms[:new_count] = self.norms[:self._size][keep_mask]

            self._size = new_count
            self.needs_save = True

        self._schedule_save()

    @property
    def total(self):
        return self._size

    def reset(self):
        with self.lock:
            self._size = 0
            # We can keep the capacity, effectively clearing the store without reallocating
            self.needs_save = True
        self._schedule_save()

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            # Search only in active IDs
            matches = np.where(self.ids[:self._size] == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
