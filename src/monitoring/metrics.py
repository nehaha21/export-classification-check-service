"""
Metrics for classification-agent runs.

The metrics track the operational measures required for the agent workflow,
including task success, attempts, tool calls, latency, and retries.
"""

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class AgentRunMetrics:
    """Metrics collected for one agent run."""

    status: str
    attempts: int
    tool_calls: int
    latency_seconds: float
    retry_count: int = 0
    success: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


class MetricsRecorder:
    """Record metrics for individual classification runs."""

    def __init__(self) -> None:
        self._active_runs: dict[str, float] = {}
        self.records: list[AgentRunMetrics] = []

    def start(self, correlation_id: str) -> None:
        """Start timing a classification run."""

        self._active_runs[correlation_id] = perf_counter()

    def finish(
        self,
        *,
        correlation_id: str,
        status: str,
        attempts: int,
        tool_calls: int,
        retry_count: int = 0,
        metadata: dict[str, object] | None = None,
    ) -> AgentRunMetrics:
        """Finish timing and store metrics for a classification run."""

        started_at = self._active_runs.pop(correlation_id, None)

        if started_at is None:
            raise ValueError(
                f"No active run found for correlation ID: {correlation_id}"
            )

        latency_seconds = perf_counter() - started_at
        success = status == "clear-for-filing"

        record = AgentRunMetrics(
            status=status,
            attempts=attempts,
            tool_calls=tool_calls,
            latency_seconds=latency_seconds,
            retry_count=retry_count,
            success=success,
            metadata=metadata or {},
        )

        self.records.append(record)

        return record

    def success_rate(self) -> float:
        """Return the proportion of recorded runs that succeeded."""

        if not self.records:
            return 0.0

        successful_runs = sum(
            1 for record in self.records if record.success
        )

        return successful_runs / len(self.records)

    def retry_rate(self) -> float:
        """Return the proportion of runs that required at least one retry."""

        if not self.records:
            return 0.0

        retried_runs = sum(
            1 for record in self.records if record.retry_count > 0
        )

        return retried_runs / len(self.records)

    def quality_summary(self) -> dict[str, float]:
        """Return the current live-run quality summary."""
        if not self.records:
            return {
                "success_rate": 0.0,
                "retry_rate": 0.0,
                "average_latency_seconds": 0.0,
            }

        average_latency = sum(
            record.latency_seconds for record in self.records
        ) / len(self.records)

        return {
            "success_rate": self.success_rate(),
            "retry_rate": self.retry_rate(),
            "average_latency_seconds": average_latency,
        }