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
        
        # Buffer for pending additions (List of arrays)
        self._buffer_vectors = []
        self._buffer_ids = []
        self._buffer_norms = []

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
            buffer_count = sum(len(ids) for ids in self._buffer_ids)
            return len(self.ids) + buffer_count

    def _load_index(self):
        """Loads vectors and IDs from a numpy file."""
        with self.lock:
            self._buffer_vectors = []
            self._buffer_ids = []
            self._buffer_norms = []

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
                    logger.info(f"Loaded Vector Store from {load_path}. Size: {len(self.ids)}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.vectors = np.empty((0, self.dimension), dtype=np.float32)
                    self.ids = np.array([], dtype=np.int64)
                    self.norms = np.array([], dtype=np.float32)

    def _schedule_save(self):
        """Schedule a debounced save."""
        if self._save_timer is not None:
            self._save_timer.cancel()

        self._save_timer = threading.Timer(self._save_interval, self.save)
        self._save_timer.daemon = True
        self._save_timer.start()

    def _merge_buffer(self):
        """Merges pending buffer into main arrays."""
        if not self._buffer_vectors:
            return

        new_vectors = np.vstack(self._buffer_vectors)
        new_ids = np.concatenate(self._buffer_ids)
        new_norms = np.concatenate(self._buffer_norms)

        if self.vectors.size == 0:
            self.vectors = new_vectors
            self.ids = new_ids
            self.norms = new_norms
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])
            self.ids = np.concatenate([self.ids, new_ids])
            self.norms = np.concatenate([self.norms, new_norms])

        self._buffer_vectors = []
        self._buffer_ids = []
        self._buffer_norms = []

    def save(self):
        """Synchronously save to disk."""
        with self.lock:
            if self.needs_save:
                self._merge_buffer()
                dirpath = os.path.dirname(self.index_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)
                with open(self.index_path, "wb") as handle:
                    np.save(handle, {'vectors': self.vectors, 'ids': self.ids})
                self.needs_save = False
                logger.info(f"Saved Vector Store to {self.index_path}. Size: {len(self.ids)}")

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

        with self.lock:
            self._buffer_vectors.append(vectors)
            self._buffer_ids.append(id_array)
            self._buffer_norms.append(new_norms)
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            # Prepare search space (Main + Buffer)
            # We don't merge here to keep read latency low during frequent updates,
            # but we search both and combine.

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            query_norm_sq = float(np.sum(np.square(query_vector)))

            candidates_dists = []
            candidates_ids = []

            # 1. Search Main Index
            if len(self.ids) > 0:
                dot_product = np.dot(self.vectors, query_vector.T).flatten()
                sq_dists = self.norms + query_norm_sq - (2.0 * dot_product)
                sq_dists = np.maximum(sq_dists, 0.0)
                candidates_dists.append(sq_dists)
                candidates_ids.append(self.ids)

            # 2. Search Buffer (if any)
            if self._buffer_vectors:
                # Combine buffer chunks temporarily for efficient dot product
                # (This is cheaper than merging into the huge main array)
                buf_vecs = np.vstack(self._buffer_vectors)
                buf_ids = np.concatenate(self._buffer_ids)
                buf_norms = np.concatenate(self._buffer_norms)

                dot_product_buf = np.dot(buf_vecs, query_vector.T).flatten()
                sq_dists_buf = buf_norms + query_norm_sq - (2.0 * dot_product_buf)
                sq_dists_buf = np.maximum(sq_dists_buf, 0.0)

                candidates_dists.append(sq_dists_buf)
                candidates_ids.append(buf_ids)

            if not candidates_dists:
                return [], []

            # Combine results
            all_dists = np.concatenate(candidates_dists)
            all_ids = np.concatenate(candidates_ids)

            # Get top K
            actual_k = min(k, len(all_ids))
            if actual_k < len(all_ids):
                top_k_partition = np.argpartition(all_dists, actual_k - 1)[:actual_k]
                sorted_idx_in_top_k = np.argsort(all_dists[top_k_partition])
                top_k_idx = top_k_partition[sorted_idx_in_top_k]
            else:
                top_k_idx = np.argsort(all_dists)

            return (
                [float(d) for d in all_dists[top_k_idx].tolist()],
                [int(i) for i in all_ids[top_k_idx].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            self._merge_buffer()
            mask = ~np.isin(self.ids, ids)
            self.vectors = self.vectors[mask]
            self.ids = self.ids[mask]
            self.norms = self.norms[mask]
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return self.ntotal

    def reset(self):
        with self.lock:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.ids = np.array([], dtype=np.int64)
            self.norms = np.array([], dtype=np.float32)
            self._buffer_vectors = []
            self._buffer_ids = []
            self._buffer_norms = []
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            self._merge_buffer()
            matches = np.where(self.ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
