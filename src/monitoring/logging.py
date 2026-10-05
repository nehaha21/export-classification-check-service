"""
Structured logging utilities for the classification service.

Logs are emitted as structured dictionaries so a classification run can be
reconstructed using its correlation ID.
"""

import json
import logging
import uuid
from typing import Any


LOGGER_NAME = "export_classification"


class StructuredLogger:
    """Emit structured JSON log records."""

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(LOGGER_NAME)

    @staticmethod
    def new_correlation_id() -> str:
        """Create a unique correlation ID for a classification run."""

        return str(uuid.uuid4())

    def log(
        self,
        *,
        event: str,
        correlation_id: str,
        level: int = logging.INFO,
        **fields: Any,
    ) -> None:
        """Emit one structured log event."""

        record = {
            "event": event,
            "correlation_id": correlation_id,
            **fields,
        }

        self.logger.log(
            level,
            json.dumps(
                record,
                default=str,
                sort_keys=True,
            ),
        )

    def run_started(
        self,
        *,
        correlation_id: str,
        **fields: Any,
    ) -> None:
        """Record the beginning of a classification run."""

        self.log(
            event="classification_run_started",
            correlation_id=correlation_id,
            **fields,
        )

    def run_completed(
        self,
        *,
        correlation_id: str,
        status: str,
        **fields: Any,
    ) -> None:
        """Record completion of a classification run."""

        self.log(
            event="classification_run_completed",
            correlation_id=correlation_id,
            status=status,
            **fields,
        )

    def run_failed(
        self,
        *,
        correlation_id: str,
        reason: str,
        **fields: Any,
    ) -> None:
        """Record a failed classification run."""

        self.log(
            event="classification_run_failed",
            correlation_id=correlation_id,
            level=logging.ERROR,
            reason=reason,
            **fields,
        )