
import time
import os
import shutil
import numpy as np
import pytest
from app.services.vector_store import VectorStore
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_vector_store(tmp_path):
    index_path = str(tmp_path / "persistence.index")
    vs = VectorStore(index_path, dimension=128)
    yield vs

def test_vector_store_background_save(temp_vector_store):
    vs = temp_vector_store
    vs._save_interval = 0.5  # Fast interval

    # Mock save method to track calls
    with patch.object(vs, 'save', wraps=vs.save) as mock_save:
        vectors = np.random.random((1, 128))
        ids = [100]

        vs.add_vectors(vectors, ids)

        # At this point, save() should not be called immediately (debouncing)
        mock_save.assert_not_called()

        # Wait for the timer
        time.sleep(1.0)

        # Now save() should have been called
        mock_save.assert_called_once()

        # Verify file exists (sanity check)
        assert os.path.exists(vs.index_path)

def test_vector_store_debounce_save(temp_vector_store):
    vs = temp_vector_store
    vs._save_interval = 0.5

    with patch.object(vs, 'save', wraps=vs.save) as mock_save:
        vectors = np.random.random((1, 128))
        ids = [200]

        # Call add_vectors twice quickly
        vs.add_vectors(vectors, ids)
        vs.add_vectors(vectors, [201])

        # Only one save should be scheduled (the last one)
        time.sleep(1.0)

        # Expect save() called once because of debounce reset
        # Wait, if the first timer fires before the second add? No, we called quickly.
        # But if the second add happens, it cancels the first timer.
        # So only one save call.
        assert mock_save.call_count == 1
