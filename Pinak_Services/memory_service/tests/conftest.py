import pytest
from fastapi.testclient import TestClient
import json
import os
import tempfile
import shutil
from app.main import app
from app.services.memory_service import MemoryService
from app.dependencies import get_memory_service

@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary directory for test data for the whole test session."""
    temp_dir = tempfile.mkdtemp()
    # Create the data directory within the temp directory
    os.makedirs(os.path.join(temp_dir, 'data'), exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_config(temp_data_dir):
    """Create a temporary config file for testing for the whole test session."""
    config = {
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_db_path": os.path.join(temp_data_dir, "data/vectors.faiss"),
        "metadata_db_path": os.path.join(temp_data_dir, "data/metadata.json"),
        "redis_host": "localhost",
        "redis_port": 6379
    }
    config_path = os.path.join(temp_data_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f)
    return config_path

@pytest.fixture(scope="function")
def memory_service(test_config, temp_data_dir):
    """Fixture to provide a fresh MemoryService instance for each test function."""
    # Ensure the data directory is clean for each test
    data_dir = os.path.join(temp_data_dir, 'data')
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir)

    service = MemoryService(config_path=test_config)
    return service

@pytest.fixture(scope="function")
def client(memory_service):
    """Fixture to provide a TestClient with dependency override for each test function."""
    def get_test_memory_service():
        return memory_service

    app.dependency_overrides[get_memory_service] = get_test_memory_service

    with TestClient(app) as c:
        yield c

    app.dependency_overrides = {}