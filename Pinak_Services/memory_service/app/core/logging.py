"""
Pinak Memory Service - Enterprise Logging System
===============================================

SOTA Structured Logging with:
- JSON structured logging
- Log correlation and tracing
- Performance monitoring
- Security event logging
- Configurable log levels
- Enterprise compliance
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    import structlog
else:
    try:
        import structlog
    except ImportError:
        # Fallback if structlog is not available
        structlog = None
import structlog

from .config import settings

# Context variables for request tracking
request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class EnterpriseLogger:
    """
    Enterprise-grade structured logging system.

    Features:
    - JSON structured logging
    - Request correlation
    - Performance monitoring
    - Security event tracking
    - Multiple output formats
    - Log rotation and retention
    """

    def __init__(self):
        self._logger: Optional[Any] = None
        self._setup_logging()

    def _setup_logging(self):
        """Configure structured logging with enterprise features."""

        # Configure structlog
        shared_processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            self._add_enterprise_fields,
            structlog.processors.JSONRenderer(),
        ]

        if settings.DEBUG:
            # Development: human-readable logs
            shared_processors = [
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                self._add_enterprise_fields,
                structlog.dev.ConsoleRenderer(colors=True),
            ]

        structlog.configure(
            processors=shared_processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(settings.LOG_LEVEL)
            ),
            context_class=dict,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure standard logging
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, settings.LOG_LEVEL))

        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create logs directory
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)

        # File handler for all logs
        file_handler = logging.FileHandler(
            log_dir / "pinak_memory.log", encoding="utf-8"
        )
        file_handler.setFormatter(self._get_json_formatter())

        # Error handler for errors only
        error_handler = logging.FileHandler(
            log_dir / "pinak_memory_error.log", encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(self._get_json_formatter())

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        if settings.DEBUG:
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
        else:
            console_handler.setFormatter(self._get_json_formatter())

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(console_handler)

        # Initialize the structlog logger after configuration
        self._logger = structlog.get_logger()

    def _get_json_formatter(self) -> logging.Formatter:
        """Get JSON formatter for structured logging."""
        return logging.Formatter(
            json.dumps(
                {
                    "timestamp": "%(asctime)s",
                    "level": "%(levelname)s",
                    "logger": "%(name)s",
                    "message": "%(message)s",
                    "module": "%(module)s",
                    "function": "%(funcName)s",
                    "line": "%(lineno)d",
                }
            )
        )

    def _add_enterprise_fields(
        self, logger: logging.Logger, method_name: str, event_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add enterprise-specific fields to log records."""
        event_dict.update(
            {
                "service": settings.APP_NAME,
                "version": settings.VERSION,
                "environment": "production" if settings.PRODUCTION else "development",
                "request_id": request_id.get(),
                "user_id": user_id.get(),
                "correlation_id": correlation_id.get(),
                "timestamp_ms": int(time.time() * 1000),
            }
        )
        return event_dict

    def info(self, event: str, **kwargs):
        """Log info level message."""
        self._logger.info(event, **kwargs)

    def error(self, event: str, **kwargs):
        """Log error level message."""
        self._logger.error(event, **kwargs)

    def warning(self, event: str, **kwargs):
        """Log warning level message."""
        self._logger.warning(event, **kwargs)

    def debug(self, event: str, **kwargs):
        """Log debug level message."""
        self._logger.debug(event, **kwargs)

    def critical(self, event: str, **kwargs):
        """Log critical level message."""
        self._logger.critical(event, **kwargs)

    def security_event(self, event: str, severity: str = "medium", **kwargs):
        """Log security-related events."""
        self._logger.warning(event, event_type="security", severity=severity, **kwargs)

    def performance_metric(self, operation: str, duration_ms: float, **kwargs):
        """Log performance metrics."""
        self._logger.info(
            f"Performance: {operation}",
            event_type="performance",
            operation=operation,
            duration_ms=duration_ms,
            **kwargs,
        )

    def audit_event(self, action: str, resource: str, **kwargs):
        """Log audit events for compliance."""
        self._logger.info(
            f"Audit: {action} on {resource}",
            event_type="audit",
            action=action,
            resource=resource,
            **kwargs,
        )


# Global logger instance
logger = EnterpriseLogger()


def set_request_context(request_id_val: str, user_id_val: Optional[str] = None):
    """Set request context for correlation."""
    request_id.set(request_id_val)
    if user_id_val:
        user_id.set(user_id_val)


def set_correlation_id(correlation_id_val: str):
    """Set correlation ID for distributed tracing."""
    correlation_id.set(correlation_id_val)


def setup_logging():
    """Setup logging for the application."""
    # Logging is initialized via EnterpriseLogger instance
    pass


def clear_request_context():
    """Clear request context."""
    request_id.set(None)
    user_id.set(None)
    correlation_id.set(None)
