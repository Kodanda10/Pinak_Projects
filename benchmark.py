import time
import numpy as np
import threading
from contextlib import contextmanager

class OldVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.lock = threading.RLock()
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)

    def add_vectors(self, vectors: np.ndarray, ids: list):
        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        with self.lock:
            self.vectors = np.vstack([self.vectors, vectors])
            self.ids = np.concatenate([self.ids, id_array])
            self.norms = np.concatenate([self.norms, new_norms])

    @contextmanager
    def batch_add(self):
        yield

class NewVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.lock = threading.RLock()
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        self._local = threading.local()

    def add_vectors(self, vectors: np.ndarray, ids: list):
        vectors = vectors.astype(np.float32)
        id_array = np.array(ids, dtype=np.int64)
        new_norms = np.sum(np.square(vectors), axis=1)

        if getattr(self._local, 'in_batch', False):
            self._local.batch_vectors.append(vectors)
            self._local.batch_ids.append(id_array)
            self._local.batch_norms.append(new_norms)
        else:
            with self.lock:
                self.vectors = np.vstack([self.vectors, vectors])
                self.ids = np.concatenate([self.ids, id_array])
                self.norms = np.concatenate([self.norms, new_norms])

    @contextmanager
    def batch_add(self):
        self._local.batch_vectors = []
        self._local.batch_ids = []
        self._local.batch_norms = []
        self._local.in_batch = True
        try:
            yield
        finally:
            self._local.in_batch = False
            if self._local.batch_vectors:
                with self.lock:
                    new_vectors = np.vstack(self._local.batch_vectors)
                    new_ids = np.concatenate(self._local.batch_ids)
                    new_norms = np.concatenate(self._local.batch_norms)

                    self.vectors = np.vstack([self.vectors, new_vectors])
                    self.ids = np.concatenate([self.ids, new_ids])
                    self.norms = np.concatenate([self.norms, new_norms])

def run_bench(store_cls, name, count):
    store = store_cls(128)
    start = time.time()
    with store.batch_add():
        for i in range(count):
            vec = np.random.rand(1, 128).astype(np.float32)
            store.add_vectors(vec, [i])
    duration = time.time() - start
    print(f"{name} ({count} items): {duration:.4f} seconds")

run_bench(OldVectorStore, "Old", 5000)
run_bench(NewVectorStore, "New", 5000)
run_bench(OldVectorStore, "Old", 10000)
run_bench(NewVectorStore, "New", 10000)
