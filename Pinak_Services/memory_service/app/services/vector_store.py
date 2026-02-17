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
        
        # In-memory storage with capacity management
        self._capacity = 0
        self._count = 0
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
        return self._count

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
                    loaded_vecs = data['vectors'].astype(np.float32)
                    loaded_ids = data['ids'].astype(np.int64)

                    self._count = len(loaded_ids)
                    self._capacity = self._count # Start with exact fit

                    self.vectors = loaded_vecs
                    self.ids = loaded_ids
                    self.norms = np.sum(np.square(self.vectors), axis=1)

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self._count}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self._reset_state()
            else:
                 self._reset_state()

    def _reset_state(self):
        self._capacity = 0
        self._count = 0
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)

    def _schedule_save(self):
        """Schedule a save if not already scheduled."""
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
                with open(self.index_path, "wb") as handle:
                    # Only save the valid portion
                    valid_vectors = self.vectors[:self._count]
                    valid_ids = self.ids[:self._count]
                    np.save(handle, {'vectors': valid_vectors, 'ids': valid_ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self._count}")

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
            target_size = self._count + n_new

            # Dynamic resizing
            if target_size > self._capacity:
                new_capacity = max(target_size, self._capacity * 2 if self._capacity > 0 else 1024)

                # Resize vectors
                new_vecs = np.zeros((new_capacity, self.dimension), dtype=np.float32)
                if self._count > 0:
                    new_vecs[:self._count] = self.vectors[:self._count]
                self.vectors = new_vecs

                # Resize ids
                new_ids = np.zeros((new_capacity,), dtype=np.int64)
                if self._count > 0:
                    new_ids[:self._count] = self.ids[:self._count]
                self.ids = new_ids

                # Resize norms
                new_norms_arr = np.zeros((new_capacity,), dtype=np.float32)
                if self._count > 0:
                    new_norms_arr[:self._count] = self.norms[:self._count]
                self.norms = new_norms_arr

                self._capacity = new_capacity

            # Copy new data
            self.vectors[self._count:target_size] = vectors
            self.ids[self._count:target_size] = id_array
            self.norms[self._count:target_size] = new_norms

            self._count += n_new
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if self._count == 0:
                return [], []

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            # Use view of valid vectors
            active_vectors = self.vectors[:self._count]
            active_norms = self.norms[:self._count]
            active_ids = self.ids[:self._count]

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            dot_product = np.dot(active_vectors, query_vector.T).flatten()
            query_norm_sq = float(np.sum(np.square(query_vector)))
            sq_dists = active_norms + query_norm_sq - (2.0 * dot_product)
            sq_dists = np.maximum(sq_dists, 0.0)

            # Get top K indices
            actual_k = min(k, self._count)
            if actual_k < self._count:
                top_k_partition = np.argpartition(sq_dists, actual_k - 1)[:actual_k]
                sorted_idx_in_top_k = np.argsort(sq_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(sq_dists)

            return (
                [float(d) for d in sq_dists[top_k_idx].tolist()],
                [int(i) for i in active_ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            active_ids = self.ids[:self._count]
            mask = ~np.isin(active_ids, ids)

            # If nothing to remove, return
            if np.all(mask):
                return

            # Compact the arrays
            # We can't easily do in-place compaction without copying, but numpy masking creates copies anyway.
            # So we create new exact-sized arrays (resetting capacity to count)
            # This is simpler than implementing compaction within capacity

            new_ids = active_ids[mask]
            new_vecs = self.vectors[:self._count][mask]
            new_norms = self.norms[:self._count][mask]

            self._count = len(new_ids)
            self._capacity = self._count

            self.ids = new_ids
            self.vectors = new_vecs
            self.norms = new_norms

            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self._count

    def reset(self):
        with self.lock:
            self._reset_state()
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            # Search in active ids
            active_ids = self.ids[:self._count]
            matches = np.where(active_ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
