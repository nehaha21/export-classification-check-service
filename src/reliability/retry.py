"""
Bounded exponential retry utilities.

Retries are used for transient model or tool failures. Retry attempts are
strictly bounded so an agent run cannot continue indefinitely.
"""

import time
from dataclasses import dataclass
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for bounded retries."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 4.0
    backoff_multiplier: float = 2.0


class RetryError(RuntimeError):
    """Raised when all retry attempts fail."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class RetryExecutor:
    """
    Execute an operation with bounded exponential backoff.

    Only exceptions selected by ``retry_on`` are retried. Other exceptions
    are propagated immediately.
    """

    def __init__(
        self,
        config: RetryConfig | None = None,
        *,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or RetryConfig()
        self.retry_on = retry_on
        self._sleep = sleep

    def execute(
        self,
        operation: Callable[[], T],
    ) -> T:
        """
        Execute an operation until it succeeds or the attempt limit is reached.
        """

        if self.config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        delay = self.config.initial_delay_seconds
        last_error: Exception | None = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return operation()
            except self.retry_on as error:
                last_error = error

                if attempt == self.config.max_attempts:
                    break

                self._sleep(min(delay, self.config.max_delay_seconds))
                delay *= self.config.backoff_multiplier

        if last_error is None:
            raise RuntimeError("Retry execution failed without an exception.")

        raise RetryError(
            "Operation failed after all retry attempts.",
            attempts=self.config.max_attempts,
            last_error=last_error,
        ) from last_error