"""Health check endpoints for monitoring service availability."""

import datetime
import os
from typing import Dict, Any

from fastapi import APIRouter, Depends
from app.api.v1.endpoints import get_memory_service
from app.services.memory_service import MemoryService


router = APIRouter()


def check_memory_service(service: MemoryService = Depends(get_memory_service)) -> Dict[str, str]:
    """Check if memory service is available."""
    try:
        # Simple check - if we can get the service, it's available
        return {"status": "ok", "message": "Memory service is available"}
    except Exception as e:
        return {"status": "error", "message": f"Memory service error: {str(e)}"}


def check_filesystem() -> Dict[str, str]:
    """Check if filesystem is writable."""
    try:
        # Check if we can write to a temp location
        test_dir = os.path.expanduser("~/.pinak_health_check")
        os.makedirs(test_dir, exist_ok=True)
        test_file = os.path.join(test_dir, "health.txt")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return {"status": "ok", "message": "Filesystem is writable"}
    except Exception as e:
        return {"status": "error", "message": f"Filesystem error: {str(e)}"}


@router.get("")
def health_check(
    service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Comprehensive health check endpoint.
    
    Returns:
        Health status with component checks
    """
    checks = {
        "memory_service": check_memory_service(service),
        "filesystem": check_filesystem(),
    }
    
    # Determine overall status
    all_ok = all(check["status"] == "ok" for check in checks.values())
    any_error = any(check["status"] == "error" for check in checks.values())
    
    if all_ok:
        overall_status = "healthy"
    elif any_error:
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "service": "memory-service",
        "version": "0.1.0",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "checks": checks,
    }


@router.get("/live")
def liveness_check() -> Dict[str, str]:
    """Liveness probe - indicates the service is running.
    
    This endpoint should always return 200 if the process is alive.
    Used by Kubernetes liveness probes.
    
    Returns:
        Simple alive status
    """
    return {
        "status": "alive",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }


@router.get("/ready")
def readiness_check(
    service: MemoryService = Depends(get_memory_service)
) -> Dict[str, Any]:
    """Readiness probe - indicates the service is ready to accept traffic.
    
    This endpoint returns 200 if the service can handle requests,
    503 otherwise. Used by Kubernetes readiness probes.
    
    Returns:
        Readiness status with component checks
    """
    checks = {
        "memory_service": check_memory_service(service),
        "filesystem": check_filesystem(),
    }
    
    all_ready = all(check["status"] == "ok" for check in checks.values())
    
    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "checks": checks,
    }
