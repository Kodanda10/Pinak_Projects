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
    Implements dynamic array resizing (amortized O(1)) for high-performance updates.
    """
    def __init__(self, index_path: str, dimension: int, initial_capacity: int = 1000):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        # Internal state
        self.size = 0
        self.capacity = initial_capacity

        # Pre-allocate arrays
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
        return self.size

    def _resize(self, min_capacity: int):
        """Resize internal arrays to at least min_capacity."""
        new_capacity = max(min_capacity, self.capacity * 2)

        # Create new larger arrays
        new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_norms = np.empty(new_capacity, dtype=np.float32)

        # Copy existing data
        new_vectors[:self.size] = self.vectors[:self.size]
        new_ids[:self.size] = self.ids[:self.size]
        new_norms[:self.size] = self.norms[:self.size]

        # Replace
        self.vectors = new_vectors
        self.ids = new_ids
        self.norms = new_norms
        self.capacity = new_capacity

        logger.debug(f"VectorStore resized to {new_capacity}")

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
                    loaded_size = len(loaded_ids)

                    # Ensure capacity
                    if loaded_size > self.capacity:
                        self._resize(loaded_size)

                    # Copy data
                    self.vectors[:loaded_size] = loaded_vectors
                    self.ids[:loaded_size] = loaded_ids
                    self.norms[:loaded_size] = np.sum(np.square(loaded_vectors), axis=1)

                    self.size = loaded_size

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self.size}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.size = 0

    def _schedule_save(self):
        """Schedule a debounced save."""
        if self._save_timer is not None:
            if self._save_timer.is_alive():
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

                # Save only the active portion
                with open(self.index_path, "wb") as handle:
                    np.save(handle, {
                        'vectors': self.vectors[:self.size],
                        'ids': self.ids[:self.size]
                    })
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self.size}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        num_new = len(ids)
        if vectors.shape[0] != num_new:
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            if self.size + num_new > self.capacity:
                self._resize(self.size + num_new)

            # Direct assignment to pre-allocated arrays
            self.vectors[self.size : self.size + num_new] = vectors
            self.ids[self.size : self.size + num_new] = id_array
            self.norms[self.size : self.size + num_new] = new_norms

            self.size += num_new
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

            # Operate on sliced views of active data
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

            # Identify which elements to keep (operating on active slice)
            active_ids = self.ids[:self.size]
            keep_mask = ~np.isin(active_ids, ids)

            # If nothing to remove, exit
            if np.all(keep_mask):
                return

            new_size = np.sum(keep_mask)

            # Compact the arrays in-place
            # We can just overwrite the beginning of the arrays with kept elements
            # This is O(N) but avoids reallocation
            self.vectors[:new_size] = self.vectors[:self.size][keep_mask]
            self.ids[:new_size] = self.ids[:self.size][keep_mask]
            self.norms[:new_size] = self.norms[:self.size][keep_mask]

            self.size = new_size
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self.size

    def reset(self):
        with self.lock:
            self.size = 0
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            active_ids = self.ids[:self.size]
            matches = np.where(active_ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
