"""
Tracing utilities for classification runs.

A trace records the sequence of important operations performed during a run,
using the correlation ID to connect all events.
"""

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class TraceSpan:
    """One operation within a classification trace."""

    name: str
    started_at: float
    duration_seconds: float
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Trace containing all recorded spans for one run."""

    correlation_id: str
    spans: list[TraceSpan] = field(default_factory=list)


class Tracer:
    """Create and record lightweight classification traces."""

    def start_trace(self, correlation_id: str) -> Trace:
        """Create a new trace for a classification run."""

        return Trace(correlation_id=correlation_id)

    def record_span(
        self,
        trace: Trace,
        *,
        name: str,
        started_at: float,
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Record a completed operation in a trace."""

        duration_seconds = perf_counter() - started_at

        span = TraceSpan(
            name=name,
            started_at=started_at,
            duration_seconds=duration_seconds,
            attributes=attributes or {},
        )

        trace.spans.append(span)

        return span

    @staticmethod
    def start_span() -> float:
        """Return a timestamp that can be used to measure an operation."""

        return perf_counter()