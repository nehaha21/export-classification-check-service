"""
Circuit breaker for unreliable model or tool dependencies.

The circuit breaker prevents repeated calls to a dependency that is
consistently failing.
"""

from dataclasses import dataclass
from enum import Enum


class CircuitState(str, Enum):
    """Possible circuit states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half-open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit-breaker behavior."""

    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0


class CircuitOpenError(RuntimeError):
    """Raised when calls are blocked by an open circuit."""


class CircuitBreaker:
    """
    Track dependency failures and temporarily block calls after repeated
    failures.
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
    ) -> None:
        self.config = config or CircuitBreakerConfig()

        if self.config.failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1.")

        if self.config.recovery_timeout_seconds <= 0:
            raise ValueError(
                "recovery_timeout_seconds must be greater than zero."
            )

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._opened_at: float | None = None

    def allow_request(self, now: float) -> bool:
        """
        Determine whether a dependency call is currently allowed.

        After the recovery timeout, an open circuit moves to half-open and
        permits one trial request.
        """

        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._opened_at is None:
                return False

            elapsed = now - self._opened_at

            if elapsed >= self.config.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                return True

            return False

        return True

    def record_success(self) -> None:
        """Record a successful dependency call."""

        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self._opened_at = None

    def record_failure(self, now: float) -> None:
        """Record a failed dependency call."""

        self.failure_count += 1

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = now

    def call(
        self,
        operation,
        *,
        now: float,
    ):
        """
        Execute a dependency operation through the circuit breaker.

        The caller supplies the current time so this component remains easy
        to test without depending directly on system time.
        """

        if not self.allow_request(now):
            raise CircuitOpenError(
                "Circuit is open; dependency calls are temporarily blocked."
            )

        try:
            result = operation()
        except Exception:
            self.record_failure(now)
            raise

        self.record_success()
        return result