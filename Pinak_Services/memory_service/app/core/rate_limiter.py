"""
Pinak Memory Service - Enterprise Rate Limiter Module
====================================================

SOTA Rate Limiting Implementation:
- Sliding window algorithm
- Configurable limits
- Burst handling
- Client-specific limits
- Monitoring integration
"""

import asyncio
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Optional, Tuple

from .config import settings
from .logging import logger


class RateLimiter:
    """
    Enterprise-grade rate limiter using sliding window algorithm.

    Features:
    - Sliding window rate limiting
    - Burst tolerance
    - Client-specific limits
    - Memory efficient implementation
    - Async support
    """

    def __init__(
        self,
        requests_per_minute: int = 1000,
        burst_limit: int = 100,
        window_size_seconds: int = 60,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Max requests per minute per client
            burst_limit: Burst limit above normal rate
            window_size_seconds: Window size for sliding window
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_size_seconds = window_size_seconds

        # Store timestamps of requests per client
        # client_ip -> deque of (timestamp, count) tuples
        self._client_requests: Dict[str, Deque[Tuple[float, int]]] = defaultdict(deque)
        self._lock = asyncio.Lock()

        logger.info(
            "Rate limiter initialized",
            requests_per_minute=requests_per_minute,
            burst_limit=burst_limit,
            window_size_seconds=window_size_seconds,
        )

    async def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if request is within rate limit.

        Args:
            client_ip: Client IP address

        Returns:
            True if request is allowed, False if rate limited
        """
        async with self._lock:
            current_time = time.time()
            client_queue = self._client_requests[client_ip]

            # Clean old entries outside the window
            self._clean_old_entries(client_queue, current_time)

            # Count requests in current window
            total_requests = sum(count for _, count in client_queue)

            if total_requests >= self.requests_per_minute:
                # Check burst limit
                if total_requests >= self.requests_per_minute + self.burst_limit:
                    logger.warning(
                        "Rate limit exceeded for client",
                        client_ip=client_ip,
                        total_requests=total_requests,
                    )
                    return False

            # Add current request
            self._add_request(client_queue, current_time)
            return True

    def _clean_old_entries(
        self, client_queue: Deque[Tuple[float, int]], current_time: float
    ):
        """Remove entries older than window size."""
        cutoff_time = current_time - self.window_size_seconds
        while client_queue and client_queue[0][0] < cutoff_time:
            client_queue.popleft()

    def _add_request(self, client_queue: Deque[Tuple[float, int]], current_time: float):
        """Add a request to the client's queue."""
        # Group requests by second for efficiency
        if client_queue and client_queue[-1][0] == int(current_time):
            # Increment count for current second
            last_time, last_count = client_queue[-1]
            client_queue[-1] = (last_time, last_count + 1)
        else:
            # New second
            client_queue.append((int(current_time), 1))

    def get_client_status(self, client_ip: str) -> Dict[str, int]:
        """
        Get rate limiting status for a client.

        Args:
            client_ip: Client IP address

        Returns:
            Dict with current requests, remaining quota, etc.
        """
        current_time = time.time()
        client_queue = self._client_requests[client_ip]

        # Clean old entries
        self._clean_old_entries(client_queue, current_time)

        total_requests = sum(count for _, count in client_queue)
        remaining = max(0, self.requests_per_minute - total_requests)
        remaining_burst = max(
            0, self.requests_per_minute + self.burst_limit - total_requests
        )

        return {
            "total_requests": total_requests,
            "remaining_quota": remaining,
            "remaining_burst": remaining_burst,
            "requests_per_minute": self.requests_per_minute,
            "burst_limit": self.burst_limit,
            "window_size_seconds": self.window_size_seconds,
        }

    def reset_client(self, client_ip: str):
        """Reset rate limiting for a specific client."""
        if client_ip in self._client_requests:
            del self._client_requests[client_ip]
            logger.info("Reset rate limiter for client", client_ip=client_ip)

    def get_all_clients_status(self) -> Dict[str, Dict[str, int]]:
        """Get rate limiting status for all clients."""
        return {
            client_ip: self.get_client_status(client_ip)
            for client_ip in self._client_requests
        }

    def cleanup_old_clients(self, max_age_seconds: int = 3600):
        """
        Cleanup clients that haven't made requests in max_age_seconds.

        Args:
            max_age_seconds: Maximum age of client data to keep
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds

        clients_to_remove = []
        for client_ip, client_queue in self._client_requests.items():
            if client_queue and client_queue[-1][0] < cutoff_time:
                clients_to_remove.append(client_ip)

        for client_ip in clients_to_remove:
            del self._client_requests[client_ip]

        if clients_to_remove:
            logger.info(
                "Cleaned up old client rate limit data",
                removed_clients=len(clients_to_remove),
            )


# Global rate limiter instance with default settings
rate_limiter = RateLimiter(
    requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    burst_limit=settings.RATE_LIMIT_BURST,
)
