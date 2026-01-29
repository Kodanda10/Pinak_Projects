import asyncio
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from app.services.background import cleanup_expired_memories

@pytest.mark.asyncio
async def test_cleanup_expired_memories_execution():
    db = MagicMock()
    cursor = MagicMock()
    db.get_cursor.return_value.__enter__.return_value = cursor
    
    # Mock behavior for one iteration then cancel
    cursor.rowcount = 5
    
    # We want to run the loop but break after one iteration
    # Since it's a 'while True' loop, we trigger CancelledError via a timeout or similar
    
    with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
        with patch("app.services.background.logger") as mock_logger:
            await cleanup_expired_memories(db, interval_seconds=1)
            
            # Verify SQL execution
            assert cursor.execute.call_count >= 2
            mock_logger.info.assert_any_call("Cleanup task cancelled")

@pytest.mark.asyncio
async def test_cleanup_expired_memories_exception_handling():
    db = MagicMock()
    db.get_cursor.side_effect = Exception("DB error")
    
    # Run loop, fail on DB access, then cancel
    with patch("asyncio.sleep", side_effect=[None, asyncio.CancelledError()]):
        with patch("app.services.background.logger") as mock_logger:
            await cleanup_expired_memories(db, interval_seconds=1)
            
            mock_logger.error.assert_called_with("Cleanup task failed: DB error", exc_info=True)
