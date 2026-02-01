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
                    logger.info(f"Loaded Vector Store from {self.index_path}. Size: {len(self.ids)}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")

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

        with self.lock:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, id_array])
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

            # Compute L2 distances
            # We use a stable approach to avoid potential overflows with large values
            # sq_dists = sum((a-b)^2)
            diff = self.vectors - query_vector
            sq_dists = np.sum(np.square(diff), axis=1)

            # Get top K indices
            actual_k = min(k, len(self.ids))
            top_k_idx = np.argsort(sq_dists)[:actual_k]

            # Return in FAISS compatibility format (2D arrays)
            return np.array([sq_dists[top_k_idx]]), np.array([self.ids[top_k_idx]])

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            mask = ~np.isin(self.ids, ids)
            self.vectors = self.vectors[mask]
            self.ids = self.ids[mask]
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return len(self.ids)

    @contextmanager
    def batch_add(self):
        yield
        self.save()
