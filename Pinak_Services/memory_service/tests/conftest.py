# tests/conftest.py
import asyncio
import json  # Import json
import os
from typing import Awaitable, Callable, Optional

import pytest
import pytest_asyncio
from app.api.v1.endpoints import get_memory_service  # Import here for dependency override
from app.db.models import Base  # Import Base for metadata

# Adjust these imports to your codebase:
from app.main import app as fastapi_app
from app.services.memory_service import MemoryService
from httpx import ASGITransport, AsyncClient  # Import ASGITransport

# --------
# Asyncio
# --------


@pytest.fixture(scope="session")
def event_loop():
    """
    Ensure a single event loop for the entire test session.
    pytest-asyncio can also do this automatically if you use mode=auto in config,
    but this explicit loop is battle-tested across plugins.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------
# Factories to create fresh test objects
# ---------------------------------------


@pytest_asyncio.fixture(scope="function")
async def fresh_memory_service_factory():
    """
    Returns an async factory that builds a brand-new MemoryService pointing to the given db_url.
    The service MUST expose:
      - async def create_all()
      - async def drop_all()
      - async def aclose()
      - .engine (optional but typical)
    If your MemoryService doesn’t have these yet, add thin wrappers to do:
      - drop_all/create_all via your SQLAlchemy Base.metadata and engine
    """
    instances = []

    async def _factory(db_url: Optional[str] = None):
        # Provide sane defaults; on-disk file recommended for FAISS + SQLite
        # Adjust constructor to pass db_url via config_path
        # Create a temporary config file for each service instance
        temp_config_path = f"temp_config_{os.urandom(8).hex()}.json"

        # Create a unique temporary data directory for this test instance
        temp_data_dir = f"temp_data_{os.urandom(8).hex()}"
        os.makedirs(temp_data_dir, exist_ok=True)

        # Read original config (assuming it exists and is accessible)
        original_config = {}
        config_file_path = os.path.join(
            os.path.dirname(__file__), "..", "app", "core", "config.json"
        )
        if os.path.exists(config_file_path):
            with open(config_file_path, "r") as f:
                original_config = json.load(f)

        # Update config with test-specific db_url and vector_db_path
        updated_config = original_config.copy()
        updated_config["metadata_db_url"] = db_url or "sqlite:///./data/test_memory.db"
        # Ensure vector_db_path points to a directory within tmp_path
        # The tmp_path fixture is available in setup_teardown_db, so we need to pass it here.
        # For now, let's make it relative to the temp_config_path, and ensure it's a directory.
        vector_data_dir = os.path.join(
            os.path.dirname(temp_config_path), f"test_data_{os.urandom(8).hex()}"
        )
        os.makedirs(vector_data_dir, exist_ok=True)
        updated_config["vector_db_path"] = os.path.join(vector_data_dir, "memory.faiss")

        # Write to temporary config file
        with open(temp_config_path, "w") as f:
            json.dump(updated_config, f, indent=2)

        # Pass the database_url directly to MemoryService constructor
        service = MemoryService(
            config_path=temp_config_path, database_url=db_url, data_dir=temp_data_dir
        )
        instances.append(service)

        # Add create_all, drop_all, aclose methods to MemoryService if they don't exist
        # These are wrappers around SQLAlchemy's Base.metadata operations
        # (Already added to MemoryService directly in previous step)

        return service

    try:
        yield _factory
    finally:
        # safety: close any leaked instances
        for svc in instances:
            try:
                # Check if aclose exists before calling
                if hasattr(svc, "aclose") and callable(svc.aclose):
                    await svc.aclose()
                elif hasattr(svc, "engine") and hasattr(svc.engine, "dispose"):
                    svc.engine.dispose()
            except Exception:
                pass
        # Clean up temporary config files
        for f in os.listdir("."):
            if f.startswith("temp_config_") and f.endswith(".json"):
                os.remove(f)


@pytest_asyncio.fixture(scope="function")
async def setup_teardown_db(tmp_path, fresh_memory_service_factory):
    """
    Creates an isolated on-disk SQLite DB for each test, then drops & recreates all tables.
    Yields a ready-to-use MemoryService instance.
    """
    db_path = tmp_path / "test.db"
    service = await fresh_memory_service_factory(db_url=f"sqlite:///{db_path}")
    await service.drop_all()
    await service.create_all()

    try:
        yield service
    finally:
        # Clear the global FAISS index to ensure test isolation
        from app.services.memory_service import _FAISS

        if _FAISS._index is not None:
            with _FAISS.lock:
                _FAISS._index.reset()
                _FAISS._index = None
                _FAISS._dim = None

        try:
            await service.drop_all()
        finally:
            await service.aclose()


@pytest_asyncio.fixture(scope="function")
async def client_and_memory_service(setup_teardown_db, fresh_test_client_factory):
    """
    Provides (httpx.AsyncClient, MemoryService) wired together so API uses THIS service instance.
    """
    service = setup_teardown_db
    client = await fresh_test_client_factory(service)
    try:
        yield client, service
    finally:
        await client.aclose()


@pytest_asyncio.fixture(scope="function")
async def fresh_test_client_factory():
    """
    Returns an async factory that wires a FastAPI/ASGI client to a specific MemoryService instance
    via dependency override. If you don’t use FastAPI, replace this with a no-op or remove it.
    """

    async def _factory(service) -> AsyncClient:
        # If using FastAPI:
        from fastapi import Depends

        # get_memory_service is imported at the top of this file

        def _override():
            return service

        fastapi_app.dependency_overrides[get_memory_service] = _override

        client = AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test")
        return client

    yield _factory
