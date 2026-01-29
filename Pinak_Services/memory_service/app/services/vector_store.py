import os
import faiss
import numpy as np
import threading
import pickle
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Thread-safe wrapper around FAISS index.
    """
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.Lock()
        self.index = None
        self._load_index()

    def _load_index(self):
        with self.lock:
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded FAISS index from {self.index_path}. Size: {self.index.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to load index: {e}. Creating new one.")
                    self.index = self._create_index()
            else:
                self.index = self._create_index()

    def _create_index(self):
        # Using IDMap to allow add_with_ids (essential for mapping to DB)
        # However, IndexIDMap requires an index that supports add_with_ids?
        # Actually IndexIDMap wraps any index to support IDs.
        base_index = faiss.IndexFlatL2(self.dimension)
        return faiss.IndexIDMap(base_index)

    def save(self):
        with self.lock:
            if self.index:
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                faiss.write_index(self.index, self.index_path)
                logger.info(f"Saved FAISS index to {self.index_path}. Size: {self.index.ntotal}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """
        Add vectors with specific IDs.
        ids must be a list of integers (e.g. unique DB row IDs or generated seq IDs).
        """
        if vectors.shape[0] != len(ids):
            raise ValueError("Number of vectors and IDs must match")

        # Ensure float32
        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)

        with self.lock:
            self.index.add_with_ids(vectors, id_array)
            # Auto-save on modification? Or let caller handle it?
            # For safety, let's auto-save for now (performance hit but safe)
            # Or maybe just periodically. Let's do explicit save calls from Service.

    def remove_ids(self, ids: List[int]):
        """Remove vectors with specific IDs."""
        id_array = np.array(ids, dtype=np.int64)
        with self.lock:
            self.index.remove_ids(id_array)

    def reconstruct(self, id: int) -> Optional[np.ndarray]:
        """Get vector by ID (if supported by index type)."""
        with self.lock:
            try:
                # Direct access might fail depending on index type (IVF vs Flat)
                # But we use IDMap around FlatL2, so it should work if ID exists.
                # However, faiss.IndexIDMap.reconstruct is not always available.
                # Usually we rely on DB for content. Reconstruct is rarely needed unless update.
                return self.index.reconstruct(id)
            except Exception:
                return None

    def search(self, query_vector: np.ndarray, k: int = 10) -> Tuple[List[float], List[int]]:
        """
        Returns (distances, ids)
        """
        with self.lock:
            if self.index.ntotal == 0:
                return [], []

            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            distances, ids = self.index.search(query_vector, k)

            # FAISS returns -1 for not found
            valid_mask = ids[0] != -1
            return distances[0][valid_mask].tolist(), ids[0][valid_mask].tolist()

    @property
    def total(self):
        return self.index.ntotal if self.index else 0
