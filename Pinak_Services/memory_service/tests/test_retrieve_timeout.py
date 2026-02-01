import time
from unittest.mock import patch

import numpy as np

from app.services.memory_service import MemoryService


class SlowModel:
    def __init__(self, dimension: int = 8):
        self.embedding_dimension = dimension

    def encode(self, sentences):
        time.sleep(0.2)
        return np.ones((len(sentences), self.embedding_dimension), dtype=np.float32)


def test_search_hybrid_timeout_skips_vector(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_EMBEDDING_TIMEOUT_MS", "10")
    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService(model=SlowModel())
        svc.db.add_semantic("alpha beta", [], "t1", "p1", 123)

        start = time.monotonic()
        results = svc.search_hybrid("alpha", "t1", "p1", limit=5)
        duration = time.monotonic() - start

    assert duration < 0.5
    assert any("alpha" in (r.get("content") or "") for r in results)
