"""
Per-agent and per-run metrics for the classification service.

The metrics track:
- task success;
- graph steps;
- tool calls;
- latency;
- retries;
- uptime.

Metrics are recorded separately for the classification and verification
agents so their performance can be evaluated independently.
"""

from dataclasses import dataclass, field
from time import monotonic
from typing import Any


@dataclass
class AgentRunMetrics:
    """Metrics collected for one agent execution."""

    agent: str
    status: str
    steps: int
    tool_calls: int
    latency_seconds: float
    retry_count: int = 0
    success: bool = False
    started_at: float = 0.0
    completed_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class MetricsRecorder:
    """
    Record operational metrics for individual agents and complete runs.

    Each agent has its own execution records, allowing task success,
    latency, retries, tool calls, and uptime to be evaluated independently.
    """

    def __init__(self) -> None:
        self._active_runs: dict[str, float] = {}
        self.records: list[AgentRunMetrics] = []

    def start(
        self,
        correlation_id: str,
    ) -> None:
        """Start timing a complete classification run."""

        self._active_runs[correlation_id] = monotonic()

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
        """
        Finish timing a complete classification run.

        This method is retained for compatibility with the existing monitoring
        interface.
        """

        started_at = self._active_runs.pop(
            correlation_id,
            None,
        )

        if started_at is None:
            raise ValueError(
                f"No active run found for correlation ID: {correlation_id}"
            )

        completed_at = monotonic()

        record = AgentRunMetrics(
            agent="classification-workflow",
            status=status,
            steps=attempts,
            tool_calls=tool_calls,
            latency_seconds=completed_at - started_at,
            retry_count=retry_count,
            success=status == "clear-for-filing",
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
        )

        self.records.append(record)

        return record

    def record_agent(
        self,
        *,
        agent: str,
        status: str,
        steps: int,
        tool_calls: int,
        latency_seconds: float,
        retry_count: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRunMetrics:
        """
        Record metrics for one specific agent.

        ``agent`` should identify the responsibility, for example:
        ``classification-agent`` or ``verification-agent``.
        """

        if latency_seconds < 0:
            raise ValueError("latency_seconds cannot be negative.")

        if steps < 0:
            raise ValueError("steps cannot be negative.")

        if tool_calls < 0:
            raise ValueError("tool_calls cannot be negative.")

        if retry_count < 0:
            raise ValueError("retry_count cannot be negative.")

        completed_at = monotonic()
        started_at = completed_at - latency_seconds

        record = AgentRunMetrics(
            agent=agent,
            status=status,
            steps=steps,
            tool_calls=tool_calls,
            latency_seconds=latency_seconds,
            retry_count=retry_count,
            success=status == "clear-for-filing",
            started_at=started_at,
            completed_at=completed_at,
            metadata=metadata or {},
        )

        self.records.append(record)

        return record

    def records_for_agent(
        self,
        agent: str,
    ) -> list[AgentRunMetrics]:
        """Return all recorded executions for one agent."""

        return [
            record
            for record in self.records
            if record.agent == agent
        ]

    def agent_success_rate(
        self,
        agent: str,
    ) -> float:
        """Return the success rate for one agent."""

        records = self.records_for_agent(agent)

        if not records:
            return 0.0

        successful_runs = sum(
            1
            for record in records
            if record.success
        )

        return successful_runs / len(records)

    def agent_retry_rate(
        self,
        agent: str,
    ) -> float:
        """Return the proportion of an agent's runs that required retries."""

        records = self.records_for_agent(agent)

        if not records:
            return 0.0

        retried_runs = sum(
            1
            for record in records
            if record.retry_count > 0
        )

        return retried_runs / len(records)

    def agent_average_latency(
        self,
        agent: str,
    ) -> float:
        """Return average latency for one agent."""

        records = self.records_for_agent(agent)

        if not records:
            return 0.0

        return sum(
            record.latency_seconds
            for record in records
        ) / len(records)

    def agent_total_tool_calls(
        self,
        agent: str,
    ) -> int:
        """Return total tool calls made by one agent."""

        return sum(
            record.tool_calls
            for record in self.records_for_agent(agent)
        )

    def agent_total_steps(
        self,
        agent: str,
    ) -> int:
        """Return total recorded execution steps for one agent."""

        return sum(
            record.steps
            for record in self.records_for_agent(agent)
        )

    def agent_uptime(
        self,
        agent: str,
    ) -> float:
        """
        Return the proportion of recorded agent executions that completed
        without an execution failure.

        A value of 1.0 means all recorded executions completed successfully;
        0.0 means none did.
        """

        records = self.records_for_agent(agent)

        if not records:
            return 0.0

        completed_runs = sum(
            1
            for record in records
            if record.completed_at > 0
        )

        return completed_runs / len(records)

    def success_rate(self) -> float:
        """Return the success rate across all recorded executions."""

        if not self.records:
            return 0.0

        successful_runs = sum(
            1
            for record in self.records
            if record.success
        )

        return successful_runs / len(self.records)

    def retry_rate(self) -> float:
        """Return the proportion of all runs requiring at least one retry."""

        if not self.records:
            return 0.0

        retried_runs = sum(
            1
            for record in self.records
            if record.retry_count > 0
        )

        return retried_runs / len(self.records)