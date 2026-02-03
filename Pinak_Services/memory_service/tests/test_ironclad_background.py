import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.background import cleanup_expired_memories

# Mark async tests
pytestmark = pytest.mark.asyncio

async def test_cleanup_expired_memories_execution():
    db = MagicMock()
    # Mock delete_expired_memories (async)
    db.delete_expired_memories = AsyncMock(return_value={"session": 5, "working": 3})
    
    # We want to run the loop but break after one iteration
    with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
        with patch("app.services.background.logger") as mock_logger:
            try:
                await cleanup_expired_memories(db, interval_seconds=1)
            except asyncio.CancelledError:
                pass
            
            # Verify Execution
            db.delete_expired_memories.assert_called()
            mock_logger.info.assert_any_call("Expired 8 memories (session: 5, working: 3)")
            mock_logger.info.assert_any_call("Cleanup task cancelled")

async def test_cleanup_expired_memories_exception_handling():
    db = MagicMock()
    db.delete_expired_memories = AsyncMock(side_effect=Exception("DB error"))
    
    # Run loop, fail on DB access, then cancel
    with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
        with patch("app.services.background.logger") as mock_logger:
            try:
                await cleanup_expired_memories(db, interval_seconds=1)
            except asyncio.CancelledError:
                pass
            
            mock_logger.error.assert_called_with("Cleanup task failed: DB error", exc_info=True)
