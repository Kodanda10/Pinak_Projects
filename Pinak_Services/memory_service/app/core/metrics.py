"""
Pinak Memory Service - Enterprise Metrics & Monitoring
====================================================

SOTA Observability with:
- Prometheus metrics collection
- Custom business metrics
- Performance monitoring
- Health checks
- Alert thresholds
- Metrics export
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse
import psutil

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest,
        CONTENT_TYPE_LATEST, CollectorRegistry
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback implementations
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass

    def generate_latest(registry=None):
        return b"# Prometheus metrics not available"

    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

from .config import settings
from .logging import logger


class EnterpriseMetrics:
    """
    Enterprise-grade metrics collection and monitoring.

    Features:
    - HTTP request metrics
    - Business logic metrics
    - System resource monitoring
    - Custom performance indicators
    - Prometheus integration
    - Alert-ready thresholds
    """

    def __init__(self):
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None

        # HTTP Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )

        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )

        # Business Metrics
        self.memory_operations_total = Counter(
            'memory_operations_total',
            'Total memory operations',
            ['operation_type', 'layer'],
            registry=self.registry
        )

        self.vector_search_duration = Histogram(
            'vector_search_duration_seconds',
            'Vector search duration',
            ['search_type'],
            registry=self.registry
        )

        self.cache_hits_total = Counter(
            'cache_hits_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )

        self.cache_misses_total = Counter(
            'cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )

        # System Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )

        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )

        self.process_memory_usage = Gauge(
            'process_memory_usage_bytes',
            'Process memory usage in bytes',
            registry=self.registry
        )

        # Business KPIs
        self.active_users = Gauge(
            'active_users_total',
            'Number of active users',
            registry=self.registry
        )

        self.memory_items_total = Gauge(
            'memory_items_total',
            'Total number of memory items stored',
            registry=self.registry
        )

        self.search_accuracy = Gauge(
            'search_accuracy_ratio',
            'Search accuracy ratio (0.0 to 1.0)',
            registry=self.registry
        )

        # Error Metrics
        self.errors_total = Counter(
            'errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        logger.info("Enterprise metrics initialized", metrics_enabled=settings.METRICS_ENABLED)

    def record_http_request(
        self, method: str, endpoint: str,
        status_code: int, duration: float
    ):
        """Record HTTP request metrics."""
        if not settings.METRICS_ENABLED:
            return

        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()

        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)

    def record_memory_operation(
        self, operation_type: str, layer: str,
        duration: Optional[float] = None
    ):
        """Record memory operation metrics."""
        if not settings.METRICS_ENABLED:
            return

        self.memory_operations_total.labels(
            operation_type=operation_type,
            layer=layer
        ).inc()

    def record_vector_search(
        self, search_type: str, duration: float,
        results_count: int = 0
    ):
        """Record vector search metrics."""
        if not settings.METRICS_ENABLED:
            return

        self.vector_search_duration.labels(
            search_type=search_type
        ).observe(duration)

    def record_cache_operation(
        self, cache_type: str, hit: bool
    ):
        """Record cache operation metrics."""
        if not settings.METRICS_ENABLED:
            return

        if hit:
            self.cache_hits_total.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses_total.labels(cache_type=cache_type).inc()

    def record_error(
        self, error_type: str, component: str
    ):
        """Record error metrics."""
        if not settings.METRICS_ENABLED:
            return

        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()

    def update_system_metrics(self):
        """Update system resource metrics."""
        if not settings.METRICS_ENABLED:
            return

        try:
            # CPU usage
            self.system_cpu_usage.set(psutil.cpu_percent(interval=1))

            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.used)

            # Process memory
            process = psutil.Process()
            self.process_memory_usage.set(process.memory_info().rss)

        except Exception as e:
            logger.error("Failed to update system metrics", error=str(e))

    def update_business_metrics(
        self, active_users: int = 0,
        memory_items: int = 0,
        search_accuracy: float = 0.0
    ):
        """Update business KPI metrics."""
        if not settings.METRICS_ENABLED:
            return

        self.active_users.set(active_users)
        self.memory_items_total.set(memory_items)
        self.search_accuracy.set(search_accuracy)

    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format."""
        if not PROMETHEUS_AVAILABLE or not settings.METRICS_ENABLED:
            return b"# Metrics collection disabled"

        return generate_latest(self.registry)

    async def metrics_middleware(self, request: Request, call_next) -> Response:
        """FastAPI middleware for automatic metrics collection."""
        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Record error metrics
            self.record_error("middleware_exception", "http")
            logger.error("Middleware exception", error=str(e))
            # Return error response
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
            status_code = 500

        # Record metrics
        duration = time.time() - start_time
        self.record_http_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=status_code,
            duration=duration
        )

        return response


# Global metrics instance
metrics = EnterpriseMetrics()


@asynccontextmanager
async def lifespan_metrics():
    """Lifespan context manager for metrics updates."""
    if settings.METRICS_ENABLED:
        # Startup
        logger.info("Starting metrics collection")
        metrics.update_system_metrics()

        yield

        # Shutdown
        logger.info("Stopping metrics collection")
    else:
        yield


def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    health = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.VERSION,
        "services": {}
    }

    try:
        # System health
        memory = psutil.virtual_memory()
        health["system"] = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_available_mb": memory.available / 1024 / 1024,
        }

        # Service health
        health["services"]["metrics"] = {
            "enabled": settings.METRICS_ENABLED,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

        # Check critical thresholds
        if memory.percent > 90:
            health["status"] = "unhealthy"
            health["issues"] = ["High memory usage"]

        if psutil.cpu_percent() > 95:
            health["status"] = "unhealthy"
            if "issues" not in health:
                health["issues"] = []
            health["issues"].append("High CPU usage")

    except Exception as e:
        health["status"] = "unhealthy"
        health["error"] = str(e)
        logger.error("Health check failed", error=str(e))

    return health


def setup_metrics(app):
    """Setup metrics middleware for the FastAPI app."""
    if settings.METRICS_ENABLED:
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(BaseHTTPMiddleware, dispatch=metrics.metrics_middleware)


# Global metrics objects for backward compatibility
REQUEST_COUNT = metrics.http_requests_total
REQUEST_LATENCY = metrics.http_request_duration
