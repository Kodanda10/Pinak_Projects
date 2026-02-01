import pytest
import os
from unittest.mock import patch

@pytest.fixture(autouse=True, scope="session")
def setup_test_env():
    # Set default environment variables for all tests
    with patch.dict(os.environ, {
        "PINAK_JWT_SECRET": "test-secret",
        "PINAK_JWT_ALGORITHM": "HS256",
        "PINAK_EMBEDDING_BACKEND": "dummy",
        # Backward-compat keys for any legacy test usage
        "JWT_SECRET": "test-secret",
        "JWT_ALGORITHM": "HS256",
        "EMBEDDING_BACKEND": "dummy",
    }):
        yield
