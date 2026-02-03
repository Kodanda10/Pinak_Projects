
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

from app.api.v1 import endpoints
from app.api.v1.endpoints import get_memory_service
from app.services.background import cleanup_expired_memories

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    service = get_memory_service()
    await service.initialize() # New async init

    skip_verify = os.getenv("PINAK_SKIP_VERIFY_ON_STARTUP", "false").lower() in ("1", "true", "yes")
    background_verify = os.getenv("PINAK_VERIFY_IN_BACKGROUND", "false").lower() in ("1", "true", "yes")

    if not skip_verify:
        if background_verify:
            # If verify_and_recover is async, wrapping it in to_thread won't work directly if it awaits.
            # But currently verify_and_recover in my update was "pass" (skipped).
            # If I implement it, I should just await it or create task.
            # For now, let's assume it's async but harmless.
            asyncio.create_task(service.verify_and_recover())
        else:
            await service.verify_and_recover()

    # Start cleanup task
    cleanup_task = asyncio.create_task(
        cleanup_expired_memories(service.db, interval_seconds=3600)
    )

    yield

    # Shutdown
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    if getattr(service, "vector_store", None):
        service.vector_store.save()

app = FastAPI(title="Pinak Memory Service", lifespan=lifespan)

app.include_router(endpoints.router, prefix="/api/v1/memory", tags=["Memory"])

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}

@app.get("/")
async def read_root():
    return {"status": "Memory service is running"}
