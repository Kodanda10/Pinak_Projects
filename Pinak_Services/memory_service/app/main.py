
from fastapi import FastAPI
from app.api.v1 import endpoints
from app.api import health

app = FastAPI(title="Pinak Memory Service", version="0.1.0")

app.include_router(endpoints.router, prefix="/api/v1/memory", tags=["Memory"])
app.include_router(health.router, prefix="/health", tags=["Health"])

@app.get("/")
def read_root():
    return {"status": "Memory service is running"}
