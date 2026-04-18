import time
import numpy as np
import threading
from contextlib import contextmanager

class VectorStoreOpt:
    def __init__(self, dimension):
        self.dimension = dimension
        self.lock = threading.RLock()
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.ids = np.array([], dtype=np.int64)
        self.norms = np.array([], dtype=np.float32)
        self._local = threading.local()

    @contextmanager
    def batch_add(self):
        self._local.batching = True
        self._local.vectors = []
        self._local.ids = []
        self._local.norms = []
        try:
            yield
        finally:
            self._local.batching = False
            if self._local.vectors:
                # Flush
                with self.lock:
                    self.vectors = np.vstack([self.vectors, *self._local.vectors])
                    self.ids = np.concatenate([self.ids, *self._local.ids])
                    self.norms = np.concatenate([self.norms, *self._local.norms])

    def add_vectors(self, vectors, ids):
        if getattr(self._local, 'batching', False):
            self._local.vectors.append(vectors)
            self._local.ids.append(np.array(ids, dtype=np.int64))
            self._local.norms.append(np.sum(np.square(vectors), axis=1))
        else:
            with self.lock:
                self.vectors = np.vstack([self.vectors, vectors])
                self.ids = np.concatenate([self.ids, np.array(ids, dtype=np.int64)])
                self.norms = np.concatenate([self.norms, np.sum(np.square(vectors), axis=1)])

vs = VectorStoreOpt(384)
start = time.time()
with vs.batch_add():
    for i in range(1000):
        vs.add_vectors(np.random.rand(1, 384).astype(np.float32), [i])
print(f"With optimized batch_add: {time.time() - start:.4f}s")
