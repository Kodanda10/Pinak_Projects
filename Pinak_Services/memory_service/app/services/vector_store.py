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
        self.capacity = 1000
        self.size = 0
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

                    n_loaded = len(loaded_ids)
                    self.size = n_loaded

                    # Ensure capacity is enough
                    if self.capacity < n_loaded:
                        self.capacity = max(self.capacity, int(n_loaded * 1.5))

                    # Re-allocate arrays with capacity if needed (or just use current if empty/small)
                    # We always re-allocate here to ensure clean state matching capacity
                    self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
                    self.ids = np.empty(self.capacity, dtype=np.int64)
                    self.norms = np.empty(self.capacity, dtype=np.float32)

                    # Copy data
                    self.vectors[:n_loaded] = loaded_vectors
                    self.ids[:n_loaded] = loaded_ids
                    self.norms[:n_loaded] = np.sum(np.square(loaded_vectors), axis=1)

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self.size}, Capacity: {self.capacity}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.size = 0
                    self.vectors = np.empty((self.capacity, self.dimension), dtype=np.float32)
                    self.ids = np.empty(self.capacity, dtype=np.int64)
                    self.norms = np.empty(self.capacity, dtype=np.float32)
            else:
                self.size = 0
                # Arrays already initialized in __init__

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
                    # Save only valid data
                    np.save(handle, {
                        'vectors': self.vectors[:self.size],
                        'ids': self.ids[:self.size]
                    })
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self.size}")

    def _resize(self, new_capacity: int):
        """Resize internal storage arrays."""
        if new_capacity <= self.capacity:
            return

        logger.info(f"Resizing Vector Store from {self.capacity} to {new_capacity}")

        new_vectors = np.empty((new_capacity, self.dimension), dtype=np.float32)
        new_ids = np.empty(new_capacity, dtype=np.int64)
        new_norms = np.empty(new_capacity, dtype=np.float32)

        # Copy existing data
        new_vectors[:self.size] = self.vectors[:self.size]
        new_ids[:self.size] = self.ids[:self.size]
        new_norms[:self.size] = self.norms[:self.size]

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
            # Check capacity
            if self.size + n_new > self.capacity:
                new_capacity = max(self.capacity * 2, self.size + n_new)
                self._resize(new_capacity)

            # Add to storage
            start_idx = self.size
            end_idx = start_idx + n_new

            self.vectors[start_idx:end_idx] = vectors
            self.ids[start_idx:end_idx] = id_array
            self.norms[start_idx:end_idx] = new_norms

            self.size += n_new
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

            # Use valid slices
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
            # We need to operate on the valid slice
            valid_ids = self.ids[:self.size]
            mask = ~np.isin(valid_ids, ids)

            # Check if any changes needed
            if np.all(mask):
                return

            new_size = np.sum(mask)

            # Compact arrays
            # We can do this in-place
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
