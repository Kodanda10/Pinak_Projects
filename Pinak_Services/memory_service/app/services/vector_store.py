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
        
        # In-memory storage with pre-allocation strategy
        self._size = 0
        self._capacity = 0

        # Initialize empty arrays, will be resized on demand
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

    @property
    def total(self):
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

                    self._size = len(loaded_ids)
                    self._capacity = self._size

                    self.vectors = loaded_vectors
                    self.ids = loaded_ids
                    self.norms = np.sum(np.square(self.vectors), axis=1)

                    logger.info(f"Loaded Vector Store from {load_path}. Size: {self._size}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self._reset_internal()
            else:
                self._reset_internal()

    def _reset_internal(self):
        """Internal reset without locking."""
        self._size = 0
        self._capacity = 0
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)

    def _ensure_capacity(self, min_capacity: int):
        """Ensure arrays have at least min_capacity, resizing geometrically if needed."""
        if min_capacity <= self._capacity:
            return

        # Double capacity or at least 1024
        new_capacity = max(min_capacity, self._capacity * 2, 1024)

        # Resize vectors
        new_vectors = np.zeros((new_capacity, self.dimension), dtype=np.float32)
        if self._size > 0:
            new_vectors[:self._size] = self.vectors[:self._size]
        self.vectors = new_vectors

        # Resize IDs
        new_ids = np.zeros(new_capacity, dtype=np.int64)
        if self._size > 0:
            new_ids[:self._size] = self.ids[:self._size]
        self.ids = new_ids

        # Resize norms
        new_norms = np.zeros(new_capacity, dtype=np.float32)
        if self._size > 0:
            new_norms[:self._size] = self.norms[:self._size]
        self.norms = new_norms

        self._capacity = new_capacity
        # logger.debug(f"Resized Vector Store to capacity {new_capacity}")

    def _schedule_save(self):
        """Schedule a debounced save."""
        # Avoid thread spam if timer is already running
        if self._save_timer is not None and self._save_timer.is_alive():
            return

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

                # Save only the used portion
                save_vectors = self.vectors[:self._size]
                save_ids = self.ids[:self._size]

                with open(self.index_path, "wb") as handle:
                    np.save(handle, {'vectors': save_vectors, 'ids': save_ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {self._size}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        n_new = len(ids)
        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            self._ensure_capacity(self._size + n_new)

            # Direct assignment to pre-allocated arrays
            self.vectors[self._size : self._size + n_new] = vectors
            self.ids[self._size : self._size + n_new] = id_array
            self.norms[self._size : self._size + n_new] = new_norms

            self._size += n_new
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

            # Use active slice of vectors
            active_vectors = self.vectors[:self._size]
            active_norms = self.norms[:self._size]
            active_ids = self.ids[:self._size]

            # Compute L2 distance using dot product: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
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
            return (
                [float(d) for d in sq_dists[top_k_idx].tolist()],
                [int(i) for i in active_ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            # Mask against active IDs
            mask = ~np.isin(self.ids[:self._size], ids)

            # Compact the arrays
            # We create new compacted arrays and reset capacity to match new size
            # This is simpler than in-place compaction for now
            new_vectors = self.vectors[:self._size][mask]
            new_ids = self.ids[:self._size][mask]
            new_norms = self.norms[:self._size][mask]

            self._size = len(new_ids)
            self._capacity = self._size

            self.vectors = new_vectors
            self.ids = new_ids
            self.norms = new_norms

            self.needs_save = True
        self._schedule_save()

    def reset(self):
        with self.lock:
            self._reset_internal()
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            matches = np.where(self.ids[:self._size] == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
