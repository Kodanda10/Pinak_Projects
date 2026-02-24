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
        
        # Initial capacity
        self.capacity = 1000
        self.size = 0

        # Pre-allocate arrays
        self.vectors = np.zeros((self.capacity, dimension), dtype=np.float32)
        self.ids = np.zeros(self.capacity, dtype=np.int64)
        self.norms = np.zeros(self.capacity, dtype=np.float32)
        
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

                    loaded_size = len(loaded_ids)
                    if loaded_size > 0:
                        # Ensure capacity is sufficient (at least double the loaded size or min 1000)
                        self.capacity = max(loaded_size * 2, 1000)
                        self.size = loaded_size

                        # Re-allocate arrays
                        self.vectors = np.zeros((self.capacity, self.dimension), dtype=np.float32)
                        self.ids = np.zeros(self.capacity, dtype=np.int64)

                        # Copy data
                        self.vectors[:self.size] = loaded_vectors
                        self.ids[:self.size] = loaded_ids
                        self.norms = np.zeros(self.capacity, dtype=np.float32)
                        self.norms[:self.size] = np.sum(np.square(self.vectors[:self.size]), axis=1)
                    else:
                        # Reset to initial state if loaded file is empty
                        self.size = 0
                        self.capacity = 1000
                        self.vectors = np.zeros((self.capacity, self.dimension), dtype=np.float32)
                        self.ids = np.zeros(self.capacity, dtype=np.int64)
                        self.norms = np.zeros(self.capacity, dtype=np.float32)

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self.size}, Capacity: {self.capacity}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.size = 0
                    self.capacity = 1000
                    self.vectors = np.zeros((self.capacity, self.dimension), dtype=np.float32)
                    self.ids = np.zeros(self.capacity, dtype=np.int64)
                    self.norms = np.zeros(self.capacity, dtype=np.float32)

    def _schedule_save(self):
        """Schedule a throttled save."""
        if self._save_timer is not None and self._save_timer.is_alive():
            return

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

                # Save only the active part of the arrays
                vectors_to_save = self.vectors[:self.size]
                ids_to_save = self.ids[:self.size]

                with open(self.index_path, "wb") as handle:
                    np.save(handle, {'vectors': vectors_to_save, 'ids': ids_to_save})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self.size}")

    def _ensure_capacity(self, n_new):
        """Ensure there is enough capacity for n_new additional vectors."""
        required_capacity = self.size + n_new
        if required_capacity > self.capacity:
            new_capacity = max(self.capacity * 2, required_capacity + 1000) # Ensure at least some growth

            # Resize vectors
            new_vectors = np.zeros((new_capacity, self.dimension), dtype=np.float32)
            new_vectors[:self.size] = self.vectors[:self.size]
            self.vectors = new_vectors

            # Resize ids
            new_ids = np.zeros(new_capacity, dtype=np.int64)
            new_ids[:self.size] = self.ids[:self.size]
            self.ids = new_ids

            # Resize norms
            new_norms = np.zeros(new_capacity, dtype=np.float32)
            new_norms[:self.size] = self.norms[:self.size]
            self.norms = new_norms

            self.capacity = new_capacity
            logger.debug(f"Resized VectorStore to capacity {self.capacity}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n_new = len(ids)
        if vectors.shape[0] != n_new:
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            self._ensure_capacity(n_new)

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

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            # Use slicing to only consider valid vectors
            active_vectors = self.vectors[:self.size]
            active_norms = self.norms[:self.size]

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
                [int(i) for i in self.ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            # Create mask for elements to KEEP
            # Only consider active ids
            current_ids = self.ids[:self.size]
            mask = ~np.isin(current_ids, ids)

            new_size = np.sum(mask)

            if new_size < self.size:
                # Compact arrays
                self.vectors[:new_size] = self.vectors[:self.size][mask]
                self.ids[:new_size] = self.ids[:self.size][mask]
                self.norms[:new_size] = self.norms[:self.size][mask]

                # Zero out the rest (optional, but good for debugging/gc)
                # self.vectors[new_size:self.size] = 0
                # self.ids[new_size:self.size] = 0
                # self.norms[new_size:self.size] = 0

                self.size = new_size
                self.needs_save = True

        self._schedule_save()

    @property
    def total(self):
        return self.size

    def reset(self):
        with self.lock:
            # Just reset size, keep capacity
            self.size = 0
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            # Search only in active ids
            active_ids = self.ids[:self.size]
            matches = np.where(active_ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
