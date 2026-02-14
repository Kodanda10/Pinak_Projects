import os
import numpy as np
import threading
import logging
import time
from typing import List, Tuple, Optional, Dict, Any
from contextlib import contextmanager

try:
    import faiss
except ImportError:
    raise ImportError("faiss-cpu is required for Enterprise Vector Store. Please install it.")

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Enterprise-grade Thread-safe Vector Store using FAISS (IndexIDMap2 + IndexFlatL2).
    Provides efficient similarity search and vector reconstruction.
    """
    def __init__(self, index_path: str, dimension: int):
        self.index_path = index_path
        self.dimension = dimension
        self.lock = threading.RLock()
        
        # Initialize FAISS index
        # We use IndexIDMap2 to support arbitrary 64-bit IDs and reconstruction
        self.quantizer = faiss.IndexFlatL2(dimension)
        self.index = faiss.IndexIDMap2(self.quantizer)
        
        self._load_index()

        self._save_timer = None
        self._save_interval = 5.0  # seconds
        self.needs_save = False

    @property
    def ntotal(self):
        return self.index.ntotal

    @property
    def total(self):
        return self.ntotal

    def _load_index(self):
        """Loads index from disk if available."""
        with self.lock:
            load_path = None
            if os.path.exists(self.index_path):
                load_path = self.index_path
            # Check for legacy .npy path just in case, though we don't migrate automatically here
            # Ideally we would migrate, but let's stick to clean start or load valid faiss index

            if load_path:
                try:
                    self.index = faiss.read_index(load_path)
                    logger.info(f"Loaded FAISS Index from {load_path}. Size: {self.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to load index from {load_path}: {e}. Creating new one.")
                    self.quantizer = faiss.IndexFlatL2(self.dimension)
                    self.index = faiss.IndexIDMap2(self.quantizer)

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
                try:
                    faiss.write_index(self.index, self.index_path)
                    self.needs_save = False
                    logger.info(f"Saved FAISS Index to {self.index_path}. Size: {self.ntotal}")
                except Exception as e:
                    logger.error(f"Failed to save index: {e}")

    def add_vectors(self, vectors: np.ndarray, ids: List[int]):
        """Add vectors with specific IDs."""
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Validation
        if vectors.shape[0] != len(ids):
            raise ValueError(f"Number of vectors ({vectors.shape[0]}) and IDs ({len(ids)}) must match")
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
            if self.ntotal == 0:
                return [], []

            query_vector = query_vector.astype(np.float32)
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)

            if query_vector.shape[1] != self.dimension:
                # Fallback or error? Return empty for safety as per old implementation behavior
                return [], []

            distances, ids = self.index.search(query_vector, k)

            # distances and ids are 2D arrays (n_queries, k)
            # Flatten for single query compatibility
            return (
                [float(d) for d in distances[0].tolist()],
                [int(i) for i in ids[0].tolist()],
            )

    def remove_ids(self, ids: List[int]):
        """Remove specific vectors by ID."""
        if not ids:
            return

        with self.lock:
            id_array = np.array(ids, dtype=np.int64)
            self.index.remove_ids(id_array)
            self.needs_save = True
        self._schedule_save()

    def reset(self):
        with self.lock:
            self.index.reset()
            self.needs_save = True
        self._schedule_save()

    def reconstruct(self, vector_id: int) -> Optional[np.ndarray]:
        """Retrieve the vector for a given ID."""
        with self.lock:
            try:
                # faiss raises RuntimeError if ID not found in some versions,
                # or returns all zeros/garbage if IDMap not used correctly.
                # IndexIDMap2 supports reconstruct.
                vec = self.index.reconstruct(vector_id)
                # Check if it actually exists (FAISS might not throw but return empty/garbage?)
                # Actually IndexIDMap2 throws RuntimeError: "key not found" usually.
                return np.array(vec, dtype=np.float32)
            except RuntimeError:
                return None
            except Exception as e:
                logger.warning(f"Failed to reconstruct vector {vector_id}: {e}")
                return None

    @contextmanager
    def batch_add(self):
        """Context manager to batch operations (conceptually).
        FAISS handles batch adds internally via add_with_ids,
        so this mainly controls the save timing."""
        try:
            yield
        finally:
            self.save()
