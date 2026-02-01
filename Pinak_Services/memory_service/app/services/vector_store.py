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
        self.lock = threading.Lock()
        
        # In-memory storage
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        
        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    def _load_index(self):
        """Loads vectors and IDs from a numpy file."""
        with self.lock:
            if os.path.exists(self.index_path):
                try:
                    data = np.load(self.index_path, allow_pickle=True).item()
                    self.vectors = data['vectors'].astype(np.float32)
                    self.ids = data['ids'].astype(np.int64)
                    # Recompute norms on load
                    self.norms = np.sum(np.square(self.vectors), axis=1)
                    logger.info(f"Loaded Vector Store from {self.index_path}. Size: {len(self.ids)}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)
                    self.ids = np.array([], dtype=np.int64)
                    self.norms = np.array([], dtype=np.float32)

    def _schedule_save(self):
        """Schedule a debounced save. Assumes lock is held."""
        if self._save_timer is not None:
            self._save_timer.cancel()

        self._save_timer = threading.Timer(self._save_interval, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def save(self):
        """Synchronously save to disk."""
        with self.lock:
            if self.needs_save:
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                np.save(self.index_path, {'vectors': self.vectors, 'ids': self.ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {len(self.ids)}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")

        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)

        # Precompute norms for new vectors
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, id_array])
            self.norms = np.concatenate([self.norms, new_norms])
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

            # Optimization: Use dot product for L2 distance
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>

            # 1. Dot product: (N, D) @ (D, 1) -> (N, 1) -> (N,)
            dot_product = np.dot(self.vectors, query_vector.T).flatten()

            # 2. Query norm squared
            query_norm_sq = np.sum(np.square(query_vector))

            # 3. Distance squared
            # self.norms is (N,)
            sq_dists = self.norms + query_norm_sq - 2 * dot_product

            # Fix potential precision issues causing negative distances close to 0
            sq_dists = np.maximum(sq_dists, 0.0)

            # Get top K indices
            actual_k = min(k, len(self.ids))

            if actual_k < len(self.ids):
                # Optimization: Use argpartition for O(N) selection instead of O(N log N) sort
                # Select the K smallest elements (unsorted)
                top_k_partition = np.argpartition(sq_dists, actual_k - 1)[:actual_k]
                # Sort only the top K
                sorted_idx_in_top_k = np.argsort(sq_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(sq_dists)

            # Return in FAISS compatibility format (2D arrays)
            return np.array([sq_dists[top_k_idx]]), np.array([self.ids[top_k_idx]])

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            mask = ~np.isin(self.ids, ids)
            self.vectors = self.vectors[mask]
            self.ids = self.ids[mask]
            self.norms = self.norms[mask]
            self.needs_save = True
            self._schedule_save()

    @property
    def total(self):
        return len(self.ids)

    @contextmanager
    def batch_add(self):
        yield
        self.save()

    # Compatibility properties/methods for testing
    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        return self.total

    def reset(self):
        with self.lock:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.ids = np.array([], dtype=np.int64)
            self.norms = np.array([], dtype=np.float32)
            self.needs_save = True
            self._schedule_save()

    def reconstruct(self, id: int) -> Optional[np.ndarray]:
        """Reconstruct vector by ID (Compatibility method)."""
        idx = np.where(self.ids == id)[0]
        if len(idx) > 0:
            return self.vectors[idx[0]]
        return None
