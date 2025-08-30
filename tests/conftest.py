"""
Comprehensive test configuration and fixtures for Pinak TDD framework.

This module provides shared fixtures, utilities, and configuration for all tests.
Following TDD principles: tests first, then implementation.
"""

import os
import sys
import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Generator, AsyncGenerator
from contextlib import contextmanager
import httpx
import uvicorn
from multiprocessing import Process

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pinak.memory.manager import MemoryManager


# ===== ENVIRONMENT SETUP =====

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables and configuration."""
    # Set test-specific environment variables
    os.environ.setdefault("PINAK_TESTING", "true")
    os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
    os.environ.setdefault("SECRET_KEY", "test-secret-key-change-in-prod")
    os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent.parent / "src"))

    # Ensure we're in test mode
    assert os.getenv("PINAK_TESTING") == "true"


# ===== TEST DATA AND FIXTURES =====

@pytest.fixture
def test_data() -> Dict[str, Any]:
    """Provide comprehensive test data for all test types."""
    return {
        "episodic": {
            "content": "Test episodic memory content",
            "salience": 0.8,
            "tags": ["test", "episodic"]
        },
        "procedural": {
            "skill_id": "test_skill",
            "steps": ["Step 1", "Step 2", "Step 3"]
        },
        "rag": {
            "query": "Test RAG content",
            "external_source": "https://example.com/test"
        },
        "session": {
            "session_id": "test_session",
            "content": "Session test content",
            "ttl": 300
        },
        "working": {
            "content": "Working memory test",
            "ttl": 300
        },
        "search_query": "test search query",
        "project_id": "test_project_123",
        "user_clearance": "confidential"
    }


@pytest.fixture
def sample_memories(test_data) -> Dict[str, Any]:
    """Provide sample memories for testing."""
    return {
        "episodic_1": {
            "content": "First episodic memory for testing",
            "salience": 0.9
        },
        "episodic_2": {
            "content": "Second episodic memory about Python development",
            "salience": 0.7
        },
        "procedural_1": {
            "skill_id": "git_workflow",
            "steps": ["git add .", "git commit -m 'test'", "git push"]
        },
        "rag_1": {
            "query": "Python async programming guide",
            "external_source": "https://docs.python.org/3/library/asyncio.html"
        }
    }


# ===== MEMORY MANAGER FIXTURES =====

@pytest.fixture
def memory_manager() -> MemoryManager:
    """Provide a configured MemoryManager instance for testing."""
    return MemoryManager(
        service_base_url="http://127.0.0.1:8000",
        token="TEST_TOKEN",
        project_id="test_project",
        timeout=30.0
    )


@pytest.fixture
def memory_manager_with_auth(memory_manager) -> MemoryManager:
    """Provide MemoryManager with authentication configured."""
    # In test environment, auth is handled via environment variables
    return memory_manager


# ===== MOCK SERVER FIXTURES =====

@pytest.fixture(scope="session")
def mock_server_url():
    """Provide the mock server URL for testing."""
    return "http://127.0.0.1:8000"


@pytest.fixture(scope="session")
def mock_server_process(mock_server_url) -> Generator[Process, None, None]:
    """Start and manage the mock server process for integration tests."""
    # Import here to avoid circular imports
    try:
        from mock_server import app
    except ImportError:
        pytest.skip("Mock server not available")

    # Start server in background process
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

    process = Process(target=run_server, daemon=True)
    process.start()

    # Wait for server to be ready
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            with httpx.Client(timeout=1.0) as client:
                response = client.get(f"{mock_server_url}/api/v1/memory/health")
                if response.status_code == 200:
                    break
        except Exception:
            pass
        import time
        time.sleep(0.5)
    else:
        process.terminate()
        pytest.fail("Mock server failed to start")

    yield process

    # Cleanup
    process.terminate()
    process.join(timeout=5)


@pytest.fixture
def http_client(mock_server_url) -> httpx.Client:
    """Provide HTTP client configured for mock server."""
    return httpx.Client(
        base_url=mock_server_url,
        headers={"Authorization": "Bearer TEST_TOKEN"},
        timeout=10.0
    )


# ===== ASYNC FIXTURES =====

@pytest.fixture
async def async_http_client(mock_server_url) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Provide async HTTP client for testing."""
    async with httpx.AsyncClient(
        base_url=mock_server_url,
        headers={"Authorization": "Bearer TEST_TOKEN"},
        timeout=10.0
    ) as client:
        yield client


# ===== TEMPORARY DIRECTORY FIXTURES =====

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="pinak_test_"))
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_file(temp_dir) -> Generator[Path, None, None]:
    """Provide temporary file for tests."""
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("test content")
    yield temp_file


# ===== CONFIGURATION FIXTURES =====

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Provide test configuration."""
    return {
        "memory_service": {
            "base_url": "http://127.0.0.1:8000",
            "timeout": 30.0,
            "retries": 3
        },
        "context_broker": {
            "cache_ttl": 300,
            "max_parallel": 5
        },
        "security": {
            "secret_key": "test-secret-key",
            "token_expiry": 3600
        }
    }


# ===== CONTEXT MANAGERS =====

@contextmanager
def mock_environment(**env_vars):
    """Context manager for temporarily setting environment variables."""
    original_values = {}
    for key, value in env_vars.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@pytest.fixture
def mock_env():
    """Provide mock environment context manager."""
    return mock_environment


# ===== TEST UTILITIES =====

@pytest.fixture
def assert_memory_operation():
    """Provide utility for asserting memory operations."""
    def _assert_operation(result, expected_type=None, should_succeed=True):
        if should_succeed:
            assert result is not None, "Memory operation should succeed"
            if expected_type:
                assert isinstance(result, expected_type), f"Result should be {expected_type}"
        else:
            assert result is None, "Memory operation should fail"
        return result

    return _assert_operation


@pytest.fixture
def wait_for_condition():
    """Provide utility for waiting for async conditions."""
    async def _wait_for_condition(condition_func, timeout=10.0, interval=0.1):
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(interval)
        return False

    return _wait_for_condition


# ===== MARKERS AND SKIPS =====

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "smoke: mark test as smoke test")
    config.addinivalue_line("markers", "tdd: mark test as written in TDD style")
    config.addinivalue_line("markers", "world_beating: mark test for world-beating features")
    config.addinivalue_line("markers", "governance: mark test for governance features")
    config.addinivalue_line("markers", "security: mark test for security features")
    config.addinivalue_line("markers", "memory: mark test for memory service")
    config.addinivalue_line("markers", "context: mark test for context broker")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on file path
        if "memory" in str(item.fspath):
            item.add_marker(pytest.mark.memory)
        elif "context" in str(item.fspath):
            item.add_marker(pytest.mark.context)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "world_beating" in str(item.fspath):
            item.add_marker(pytest.mark.world_beating)

        # Add TDD marker to all tests (we follow TDD principles)
        item.add_marker(pytest.mark.tdd)


# ===== PERFORMANCE MONITORING =====

@pytest.fixture
def performance_monitor():
    """Provide performance monitoring utility."""
    import time
    from contextlib import contextmanager

    @contextmanager
    def _monitor(operation_name: str):
        start_time = time.time()
        yield
        duration = time.time() - start_time
        print(f"Performance: {operation_name} took {duration:.3f}s")

    return _monitor


# ===== CLEANUP FIXTURES =====

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Add any cleanup logic here
    # For example, clear caches, reset state, etc.


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """Clean up after test session."""
    yield
    # Add session-level cleanup logic here
    # For example, clean up temp files, reset databases, etc.

