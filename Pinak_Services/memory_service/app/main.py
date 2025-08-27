
from fastapi import FastAPI, Response
from app.api.v1 import endpoints
import os

PROM_ENABLED = os.getenv('PINAK_METRICS', 'false').lower() in {'1','true','yes','on'}
try:
    from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
    REQ_COUNTER = Counter('pinak_requests_total', 'Requests per layer', ['layer','project_id'])
except Exception:
    PROM_ENABLED = False
    REQ_COUNTER = None  # type: ignore

app = FastAPI(title="Pinak Memory Service")

app.include_router(endpoints.router, prefix="/api/v1/memory", tags=["Memory"])

@app.get("/")
def read_root():
    return {"status": "Memory service is running"}

@app.get('/metrics')
def metrics():
    if not PROM_ENABLED or REQ_COUNTER is None:
        return Response(status_code=404)
    data = generate_latest()  # type: ignore
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)  # type: ignore
