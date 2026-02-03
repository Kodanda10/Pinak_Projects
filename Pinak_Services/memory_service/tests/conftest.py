import pytest
import pytest_asyncio
import os
import asyncio
from unittest.mock import patch
from sqlalchemy.ext.asyncio import create_async_engine
import app.core.async_db
import app.core.database

# Use pytest_asyncio.fixture for async fixtures
@pytest_asyncio.fixture(autouse=True, scope="session")
async def setup_test_env():
    # Set default environment variables for all tests
    test_db_path = "./test_memory.db"
    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass

    # Create new test engine
    test_engine = create_async_engine(test_db_url, echo=False)

    # 1. Dispose old engine to release locks
    await app.core.async_db.engine.dispose()

    # 2. Patch global engine variable in modules that import it
    with patch("app.core.async_db.engine", test_engine), \
         patch("app.core.database.engine", test_engine):

        # 3. Configure Session to use test engine
        app.core.async_db.AsyncSessionLocal.configure(bind=test_engine)

        # 4. Patch env vars
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
        try:
            os.remove(test_db_path)
        except OSError:
            pass

@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
