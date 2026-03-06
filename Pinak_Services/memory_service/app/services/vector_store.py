import os
import numpy as np
import threading
import logging
import time
import json
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

import faiss

class VectorStore:
    """
    Thread-safe Vector Store using FAISS for robust similarity search.
    Provides scalable Approximate Nearest Neighbors (ANN) and exact search capabilities.
    """
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

        self._load_index()

    @property
    def ntotal(self):
        with self.lock:
            return self.index.ntotal

    def _load_index(self):
        """Loads FAISS index from disk or creates a new one."""
        with self.lock:
            load_path = None
            if os.path.exists(self.index_path):
                load_path = self.index_path
            elif os.path.exists(f"{self.index_path}.index"):
                load_path = f"{self.index_path}.index"

            if load_path:
                try:
                    self.index = faiss.read_index(load_path)
                    logger.info(f"Loaded FAISS Vector Store from {load_path}. Size: {self.index.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new IndexIDMap2(IndexFlatL2).")
                    base_index = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIDMap2(base_index)
            else:
                base_index = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIDMap2(base_index)

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

                # Make sure to save with standard faiss extension if saving new file
                save_path = self.index_path
                if not save_path.endswith('.index'):
                    save_path += '.index'

                faiss.write_index(self.index, save_path)
                self.needs_save = False
                logger.info(f"Saved FAISS Vector Store to {save_path}. Size: {self.index.ntotal}")

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

        with self.lock:
            self.index.add_with_ids(vectors, id_array)
            self.needs_save = True

        self._schedule_save()

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """Find top K nearest neighbors using L2 distance."""
        with self.lock:
            if self.index.ntotal == 0:
                return [], []

            # Ensure vectors and query are float32 for consistency
            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.shape[1] != self.dimension:
                return [], []

            actual_k = min(k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, actual_k)

            if len(distances) == 0:
                return [], []

            # Remove -1 indices (FAISS returns -1 for not found)
            valid_mask = indices[0] != -1

            return (
                [float(d) for d in distances[0][valid_mask].tolist()],
                [int(i) for i in indices[0][valid_mask].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        with self.lock:
            id_selector = faiss.IDSelectorBatch(ids)
            self.index.remove_ids(id_selector)
            self.needs_save = True
        self._schedule_save()

    @property
    def total(self):
        with self.lock:
            return self.index.ntotal

    def reset(self):
        with self.lock:
            self.index.reset()
            self.needs_save = True

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        with self.lock:
            try:
                # FAISS reconstruct returns the vector
                return self.index.reconstruct(vector_id)
            except RuntimeError:
                # Vector not found
                return None

    @contextmanager
    def batch_add(self):
        yield
        self.save()
