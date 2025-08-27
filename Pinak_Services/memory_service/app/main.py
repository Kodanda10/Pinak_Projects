
from fastapi import FastAPI, Response, Request
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

# Optional OTEL tracing middleware
OTEL_ENABLED = os.getenv('PINAK_OTEL', 'false').lower() in {'1','true','yes','on'}
if OTEL_ENABLED:
    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

        tp = TracerProvider()
        tp.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(tp)
        tracer = trace.get_tracer("pinak.memory")

        class OtelMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                with tracer.start_as_current_span("request") as span:  # type: ignore
                    span.set_attribute("http.method", request.method)
                    span.set_attribute("http.route", request.url.path)
                    pid = request.headers.get('X-Pinak-Project')
                    if pid:
                        span.set_attribute("pinak.project_id", pid)
                    response = await call_next(request)
                    span.set_attribute("http.status_code", response.status_code)
                    return response

        app.add_middleware(OtelMiddleware)
    except Exception:
        pass
