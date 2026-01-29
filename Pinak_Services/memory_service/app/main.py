
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from app.api.v1 import endpoints
from app.api.v1.endpoints import get_memory_service
from app.services.background import cleanup_expired_memories

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    service = get_memory_service()
    service.verify_and_recover()

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
    service.vector_store.save()

app = FastAPI(title="Pinak Memory Service", lifespan=lifespan)

app.include_router(endpoints.router, prefix="/api/v1/memory", tags=["Memory"])

@app.get("/")
def read_root():
    return {"status": "Memory service is running"}
