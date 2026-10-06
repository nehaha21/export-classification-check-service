"""
Caching and per-client rate limiting for the API.
"""

from collections import defaultdict
from time import monotonic
from typing import Any


class ResponseCache:
    """Simple in-memory cache for completed responses."""

    def __init__(self, ttl_seconds: float = 60.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)

        if entry is None:
            return None

        created_at, value = entry

        if monotonic() - created_at >= self.ttl_seconds:
            del self._entries[key]
            return None

        return value

    def set(self, key: str, value: Any) -> None:
        self._entries[key] = (monotonic(), value)


class RateLimiter:
    """Fixed-window per-client rate limiter."""

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: float = 60.0,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def allow(self, client_id: str) -> bool:
        now = monotonic()
        timestamps = self._requests[client_id]

        cutoff = now - self.window_seconds
        self._requests[client_id] = [
            timestamp for timestamp in timestamps if timestamp > cutoff
        ]

        timestamps = self._requests[client_id]

        if len(timestamps) >= self.max_requests:
            return False

        timestamps.append(now)
        return True