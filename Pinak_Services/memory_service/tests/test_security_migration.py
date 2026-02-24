import os
import numpy as np
import pytest
from app.services.vector_store import VectorStore

def test_migration_from_pickle_to_npz(tmp_path):
    # 1. Setup Legacy State (Simulate old VectorStore behavior)
    legacy_path = tmp_path / "legacy.index"
    npy_path = str(legacy_path) + ".npy"

    vectors = np.random.rand(10, 384).astype(np.float32)
    ids = np.arange(10, dtype=np.int64)

    # Manually save as pickle (mimicking old behavior)
    with open(npy_path, "wb") as f:
        np.save(f, {'vectors': vectors, 'ids': ids})

    # 2. Verify it requires pickle
    # np.load raises ValueError if allow_pickle=False and pickle is needed
    try:
        np.load(npy_path, allow_pickle=False)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    except Exception:
        # Some numpy versions might raise UnpicklingError or other things, but usually ValueError
        pass

    # 3. Initialize VectorStore with this path
    # It should load the legacy file (currently uses allow_pickle=True)
    store = VectorStore(str(legacy_path), 384)
    assert store.ntotal == 10

    # 4. Trigger Save (which SHOULD BE secure after fix)
    store.needs_save = True
    store.save()

    # 5. Verify NPZ exists and is loadable WITHOUT pickle
    # After fix, save() should produce a .npz file
    # We need to handle how the path is constructed.
    # If input was "legacy.index", code currently appends .npy.
    # My proposed fix will check for .npz.

    # Let's assume the fix saves to .npz
    # If legacy path was used, maybe it saves to legacy.index.npz?

    found_npz = False
    possible_paths = [
        str(legacy_path) + ".npz",
        str(legacy_path) + ".npy.npz"
    ]

    final_path = None
    for p in possible_paths:
        if os.path.exists(p):
            found_npz = True
            final_path = p
            break

    # This assertion is expected to FAIL before the fix
    if not found_npz:
        pytest.fail("Secure .npz file not found")

    # Try loading with allow_pickle=False
    with np.load(final_path, allow_pickle=False) as data:
        assert 'vectors' in data
        assert 'ids' in data
        assert len(data['ids']) == 10

def test_secure_by_default(tmp_path):
    # New store should use secure format
    path = tmp_path / "secure.index"
    store = VectorStore(str(path), 384)

    vectors = np.random.rand(5, 384).astype(np.float32)
    ids = [1, 2, 3, 4, 5]
    store.add_vectors(vectors, ids)

    # Force save
    store.save()

    # Check file
    expected_npz = str(path) + ".npz"

    # This assertion is expected to FAIL before the fix (it will be .npy)
    if not os.path.exists(expected_npz):
        pytest.fail(f"Expected {expected_npz} to exist")

    # Verify secure load
    with np.load(expected_npz, allow_pickle=False) as data:
        assert len(data['ids']) == 5
