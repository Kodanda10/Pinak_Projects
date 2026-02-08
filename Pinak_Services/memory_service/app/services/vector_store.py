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
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        
        # Write buffer (list of arrays)
        self._buffer_vectors: List[np.ndarray] = []
        self._buffer_ids: List[int] = []

        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        with self.lock:
            return len(self.ids) + len(self._buffer_ids)

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
                    # Clear buffer on load to ensure clean state
                    self._buffer_vectors = []
                    self._buffer_ids = []
                    logger.info(f"Loaded Vector Store from {load_path}. Size: {len(self.ids)}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)
                    self.ids = np.array([], dtype=np.int64)
                    self.norms = np.array([], dtype=np.float32)
                    self._buffer_vectors = []
                    self._buffer_ids = []

    def _schedule_save(self):
        """Schedule a debounced save."""
        if self._save_timer is not None:
            self._save_timer.cancel()

        self._save_timer = threading.Timer(self._save_interval, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _flush_buffer(self):
        """Merges buffered vectors into the main storage. Assumes lock is held by caller."""
        if not self._buffer_vectors:
            return

        # Consolidate buffer
        new_vectors = np.vstack(self._buffer_vectors)
        new_ids = np.array(self._buffer_ids, dtype=np.int64)
        new_norms = np.sum(np.square(new_vectors), axis=1)

        # Merge into main storage
        self.vectors = np.vstack([self.vectors, new_vectors])
        self.ids = np.concatenate([self.ids, new_ids])
        self.norms = np.concatenate([self.norms, new_norms])

        # Clear buffer
        self._buffer_vectors = []
        self._buffer_ids = []

    def save(self):
        """Synchronously save to disk."""
        with self.lock:
            self._flush_buffer()
            if self.needs_save:
                dirpath = os.path.dirname(self.index_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(self.index_path, "wb") as handle:
                    np.save(handle, {'vectors': self.vectors, 'ids': self.ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {len(self.ids)}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors to the write buffer."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} does not match index dimension {self.dimension}")

        vectors = vectors.astype(np.float32)

        with self.lock:
            # Optimized: Append to buffer instead of immediate vstack (O(1) vs O(N))
            self._buffer_vectors.append(vectors)
            self._buffer_ids.extend(ids)
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            self._flush_buffer()

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
            self._flush_buffer()
            mask = ~np.isin(self.ids, ids)
            self.vectors = self.vectors[mask]
            self.ids = self.ids[mask]
            self.norms = self.norms[mask]
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        with self.lock:
            return len(self.ids) + len(self._buffer_ids)

    def reset(self):
        with self.lock:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.ids = np.array([], dtype=np.int64)
            self.norms = np.array([], dtype=np.float32)
            self._buffer_vectors = []
            self._buffer_ids = []
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            self._flush_buffer()
            matches = np.where(self.ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
