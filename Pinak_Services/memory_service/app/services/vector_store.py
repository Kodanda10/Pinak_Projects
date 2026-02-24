import os
import numpy as np
import threading
import logging
from typing import List, Tuple, Optional
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
        
        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

        # In-memory storage
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        
        self._load_index()

    @property
    def index(self):
        return self

    @property
    def ntotal(self):
        return len(self.ids)

    def _load_index(self):
        """Loads vectors and IDs."""
        with self.lock:
            # Try loading NPZ (Secure, New Format)
            if self.index_path.endswith('.npz'):
                npz_path = self.index_path
            elif self.index_path.endswith('.npy'):
                npz_path = self.index_path[:-4] + ".npz"
            else:
                npz_path = self.index_path + ".npz"

            if os.path.exists(npz_path):
                try:
                    with np.load(npz_path, allow_pickle=False) as data:
                        self.vectors = data['vectors'].astype(np.float32)
                        self.ids = data['ids'].astype(np.int64)
                        self.norms = np.sum(np.square(self.vectors), axis=1)
                    logger.info(f"Loaded Secure Vector Store from {npz_path}. Size: {len(self.ids)}")
                    return
                except Exception as e:
                    logger.error(f"Failed to load secure index from {npz_path}: {e}")

            # Fallback to Legacy NPY (Insecure Pickle)
            load_path = None
            if os.path.exists(self.index_path):
                load_path = self.index_path
            elif os.path.exists(f"{self.index_path}.npy"):
                load_path = f"{self.index_path}.npy"

            if load_path:
                try:
                    logger.warning(f"Loading legacy insecure index from {load_path}. Migration to .npz scheduled.")
                    data = np.load(load_path, allow_pickle=True).item()
                    self.vectors = data['vectors'].astype(np.float32)
                    self.ids = data['ids'].astype(np.int64)
                    self.norms = np.sum(np.square(self.vectors), axis=1)
                    # Trigger save to migrate
                    self.needs_save = True
                    self._schedule_save()
                except Exception as e:
                    logger.error(f"Failed to load legacy index: {e}. Creating new one.")
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
        """Synchronously save to disk using secure .npz format."""
        with self.lock:
            if self.needs_save:
                # Construct NPZ path
                if self.index_path.endswith('.npz'):
                    save_path = self.index_path
                elif self.index_path.endswith('.npy'):
                    save_path = self.index_path[:-4] + ".npz"
                else:
                    save_path = self.index_path + ".npz"

                dirpath = os.path.dirname(save_path)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)

                # Secure save without pickle
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
