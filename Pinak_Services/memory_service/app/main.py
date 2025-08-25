
from fastapi import FastAPI
from app.api.v1 import endpoints

app = FastAPI(title="Pinak Memory Service")

app.include_router(endpoints.router, prefix="/api/v1/memory", tags=["Memory"])

@app.get("/")
def read_root():
    return {"status": "Memory service is running"}
