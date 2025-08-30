"""
Pinak Memory Service - Enterprise Circuit Breaker Module
=======================================================

SOTA Circuit Breaker Implementation:
- Failure detection
- Automatic recovery
- Configurable thresholds
- Service degradation
- Monitoring integration
"""

import asyncio
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .config import settings
from .logging import logger


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Individual circuit breaker for a service endpoint.

    Implements circuit breaker pattern with configurable thresholds
    and automatic recovery.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} entering half-open state")
                else:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker {self.name} is open"
                    )

            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise e

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout

    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                logger.info(
                    f"Circuit breaker {self.name} closed after successful recovery"
                )
        else:
            # Reset failure count on success in closed state
            self.failure_count = 0

    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
            logger.warning(
                f"Circuit breaker {self.name} opened due to failure in half-open state"
            )
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker {self.name} opened due to failure threshold exceeded"
            )

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "next_attempt_time": (
                self.last_failure_time + self.recovery_timeout
                if self.last_failure_time
                else None
            ),
        }


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides centralized management and monitoring of circuit breakers
    across different services and endpoints.
    """

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_config = {
            "failure_threshold": settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "recovery_timeout": settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            "success_threshold": settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
        }

    def get_or_create(
        self,
        name: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[int] = None,
        success_threshold: Optional[int] = None,
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold
                or self._global_config["failure_threshold"],
                recovery_timeout=recovery_timeout
                or self._global_config["recovery_timeout"],
                success_threshold=success_threshold
                or self._global_config["success_threshold"],
            )
            logger.info(f"Created circuit breaker: {name}")

        return self._breakers[name]

    def get_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            "total_breakers": len(self._breakers),
            "breakers": {
                name: breaker.get_status() for name, breaker in self._breakers.items()
            },
            "global_config": self._global_config,
        }

    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self._breakers.values():
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.success_count = 0
            breaker.last_failure_time = None
        logger.info("Reset all circuit breakers")

    def remove(self, name: str):
        """Remove circuit breaker."""
        if name in self._breakers:
            del self._breakers[name]
            logger.info(f"Removed circuit breaker: {name}")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""

    pass


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()
