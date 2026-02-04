import asyncio
import logging
import datetime
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

            now = datetime.datetime.now(datetime.timezone.utc).isoformat()

            deleted = await db.delete_expired_memories(now)
            session_deleted = deleted.get("session", 0)
            working_deleted = deleted.get("working", 0)
            total = session_deleted + working_deleted

            if total > 0:
                logger.info(f"Expired {total} memories (session: {session_deleted}, working: {working_deleted})")

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup task failed: {e}", exc_info=True)
            # Continue even if failed
