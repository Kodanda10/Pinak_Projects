import asyncio
import logging
from datetime import datetime, timezone
from app.core.database import DatabaseManager

logger = logging.getLogger(__name__)

async def cleanup_expired_memories(db: DatabaseManager, interval_seconds: int = 3600):
    """
    Background task: Delete expired session/working memories.

    Args:
        db: DatabaseManager instance
        interval_seconds: How often to run (default: 1 hour)
    """
    logger.info(f"Expiration cleanup task started (interval: {interval_seconds}s)")

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            now = datetime.now(timezone.utc).isoformat()

            # Delete expired session memories
            with db.get_cursor() as cur:
                cur.execute("DELETE FROM logs_session WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
                session_deleted = cur.rowcount

                # Delete expired working memories
                cur.execute("DELETE FROM working_memory WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
                working_deleted = cur.rowcount

            total = session_deleted + working_deleted

            if total > 0:
                logger.info(f"Expired {total} memories (session: {session_deleted}, working: {working_deleted})")

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}", exc_info=True)
            # Continue even if failed
