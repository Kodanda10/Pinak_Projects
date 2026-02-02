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
        self.capacity = 0
        self._size = 0
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        
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
                    self.vectors = data['vectors'].astype(np.float32)
                    self.ids = data['ids'].astype(np.int64)
                    self.norms = np.sum(np.square(self.vectors), axis=1)

                    self._size = len(self.ids)
                    self.capacity = self._size

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self._size}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)
                    self.ids = np.array([], dtype=np.int64)
                    self.norms = np.array([], dtype=np.float32)
                    self._size = 0
                    self.capacity = 0

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
                    # Only save valid data
                    np.save(handle, {
                        'vectors': self.vectors[:self._size],
                        'ids': self.ids[:self._size]
                    })
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self._size}")

    def _expand(self, min_capacity: int):
        """Expand internal storage arrays."""
        new_capacity = max(self.capacity * 2, min_capacity, 1024)

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

        n_new = len(ids)

        with self.lock:
            new_size = self._size + n_new
            if new_size > self.capacity:
                self._expand(new_size)

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

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            # Use views of active data
            vectors_view = self.vectors[:self._size]
            norms_view = self.norms[:self._size]
            ids_view = self.ids[:self._size]

            dot_product = np.dot(vectors_view, query_vector.T).flatten()
            query_norm_sq = float(np.sum(np.square(query_vector)))
            sq_dists = norms_view + query_norm_sq - (2.0 * dot_product)
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
            return (
                [float(d) for d in sq_dists[top_k_idx].tolist()],
                [int(i) for i in ids_view[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            # Mask against active IDs
            current_ids = self.ids[:self._size]
            mask = ~np.isin(current_ids, ids)

            # This creates new arrays for storage (shrinking capacity to fit)
            self.vectors = self.vectors[:self._size][mask]
            self.ids = current_ids[mask]
            self.norms = self.norms[:self._size][mask]

            self._size = len(self.ids)
            self.capacity = self._size
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self._size

    def reset(self):
        with self.lock:
            # We can release memory or just reset size
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.ids = np.array([], dtype=np.int64)
            self.norms = np.array([], dtype=np.float32)
            self._size = 0
            self.capacity = 0
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            # Search in active slice
            matches = np.where(self.ids[:self._size] == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
