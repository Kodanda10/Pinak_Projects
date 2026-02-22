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
    Refactored to use safer .npz format without pickle.
    """
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        # In-memory storage
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
        return len(self.ids)

    def _get_actual_path(self) -> Optional[str]:
        """Resolves the actual file path, prioritizing .npz over .npy."""
        base, ext = os.path.splitext(self.index_path)

        # 1. Check if configured path is explicitly .npz and exists
        if ext == ".npz" and os.path.exists(self.index_path):
            return self.index_path

        # 2. Check for .npz variant (preferred)
        # If index_path is "foo.npy", look for "foo.npz"
        # If index_path is "foo", look for "foo.npz"
        npz_candidate = (base + ".npz") if ext == ".npy" else (self.index_path + ".npz")
        if os.path.exists(npz_candidate):
            return npz_candidate

        # 3. Fallback to legacy .npy paths
        if os.path.exists(self.index_path):
            return self.index_path

        # 4. Fallback to implicit .npy extension
        # If index_path is "foo", look for "foo.npy"
        if os.path.exists(self.index_path + ".npy"):
             return self.index_path + ".npy"

        return None

    def _load_index(self):
        """Loads vectors and IDs from disk, handling both .npz (safe) and .npy (legacy)."""
        with self.lock:
            load_path = self._get_actual_path()
            if load_path:
                try:
                    loaded = False
                    # Attempt safe load first (works for .npz)
                    try:
                        data = np.load(load_path, allow_pickle=False)
                        if isinstance(data, np.lib.npyio.NpzFile):
                             self.vectors = data['vectors'].astype(np.float32)
                             self.ids = data['ids'].astype(np.int64)
                             loaded = True
                             logger.info(f"Loaded Vector Store from {load_path} (format: npz). Size: {len(self.ids)}")
                    except (ValueError, OSError, KeyError):
                         # Not a valid npz or disallowed pickle (legacy npy)
                         pass

                    if not loaded:
                         # Fallback to legacy pickle load for .npy files
                         # Only verify logic if filename suggests legacy
                         if load_path.endswith(".npy") or not load_path.endswith(".npz"):
                             try:
                                 data = np.load(load_path, allow_pickle=True).item()
                                 self.vectors = data['vectors'].astype(np.float32)
                                 self.ids = data['ids'].astype(np.int64)
                                 loaded = True
                                 logger.warning(f"Loaded legacy Vector Store from {load_path}. This will be converted to .npz on next save.")
                             except Exception:
                                 pass

                    if not loaded:
                        raise ValueError(f"Could not load valid index from {load_path}")

                    self.norms = np.sum(np.square(self.vectors), axis=1)

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

    def save(self):
        """Synchronously save to disk in secure .npz format."""
        with self.lock:
            if self.needs_save:
                # Determine save path. Prefer .npz
                base, ext = os.path.splitext(self.index_path)
                if ext == ".npy":
                    save_path = base + ".npz"
                elif ext == ".npz":
                    save_path = self.index_path
                else:
                    save_path = self.index_path + ".npz"

                dirpath = os.path.dirname(save_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)

                # Save using npz format (compressed is usually better for vectors)
                np.savez_compressed(save_path, vectors=self.vectors, ids=self.ids)

                self.needs_save = False
                logger.info(f"Saved Vector Store to {save_path}. Size: {len(self.ids)}")

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
            mask = ~np.isin(self.ids, ids)
            self.vectors = self.vectors[mask]
            self.ids = self.ids[mask]
            self.norms = self.norms[mask]
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        return len(self.ids)

    def reset(self):
        with self.lock:
            self.vectors = np.empty((0, self.dimension), dtype=np.float32)
            self.ids = np.array([], dtype=np.int64)
            self.norms = np.array([], dtype=np.float32)
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            matches = np.where(self.ids == vector_id)[0]
            if len(matches) == 0:
                return None
            return self.vectors[matches[0]].copy()

    @contextmanager
    def batch_add(self):
        yield
        self.save()
