import pytest
import os
import asyncio
from unittest.mock import patch
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.async_db import AsyncSessionLocal, engine

@pytest.fixture(autouse=True, scope="session")
async def setup_test_env():
    # Set default environment variables for all tests
    test_db_path = "./test_memory.db"
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create new test engine
    test_engine = create_async_engine(test_db_url, echo=False)

    # Reconfigure the global sessionmaker to use the test engine
    # We must dispose the old engine first to release locks if any
    await engine.dispose()
    AsyncSessionLocal.configure(bind=test_engine)

    # Patch env vars for other components (like MemoryService config loading)
    with patch.dict(os.environ, {
        "PINAK_JWT_SECRET": "test-secret",
        "PINAK_JWT_ALGORITHM": "HS256",
        "PINAK_EMBEDDING_BACKEND": "dummy",
        "PINAK_DATABASE_URL": test_db_url,
        "JWT_SECRET": "test-secret",
        "JWT_ALGORITHM": "HS256",
        "EMBEDDING_BACKEND": "dummy",
    }):
        yield

    # Cleanup
    await test_engine.dispose()
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
