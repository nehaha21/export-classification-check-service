"""
Token and cost tracking for agent runs.

The tracker records model usage per agent step so that token consumption and
estimated cost can be associated with a classification run.
"""

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Token usage for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Return total tokens used by the call."""

        return self.input_tokens + self.output_tokens


@dataclass
class CostRecord:
    """Cost information for one agent run."""

    correlation_id: str
    agent: str
    usage: TokenUsage
    estimated_cost: float
    metadata: dict[str, object] = field(default_factory=dict)


class CostTracker:
    """Track token usage and estimated model costs."""

    def __init__(
        self,
        *,
        input_cost_per_1k_tokens: float = 0.0,
        output_cost_per_1k_tokens: float = 0.0,
    ) -> None:
        self.input_cost_per_1k_tokens = input_cost_per_1k_tokens
        self.output_cost_per_1k_tokens = output_cost_per_1k_tokens
        self.records: list[CostRecord] = []

    def record(
        self,
        *,
        correlation_id: str,
        agent: str,
        usage: TokenUsage,
        metadata: dict[str, object] | None = None,
    ) -> CostRecord:
        """Record token usage and calculate the estimated cost."""

        input_cost = (
            usage.input_tokens / 1000
        ) * self.input_cost_per_1k_tokens

        output_cost = (
            usage.output_tokens / 1000
        ) * self.output_cost_per_1k_tokens

        estimated_cost = input_cost + output_cost

        record = CostRecord(
            correlation_id=correlation_id,
            agent=agent,
            usage=usage,
            estimated_cost=estimated_cost,
            metadata=metadata or {},
        )

        self.records.append(record)

        return record

    def total_tokens(self) -> int:
        """Return total tokens recorded across all model calls."""

        return sum(
            record.usage.total_tokens
            for record in self.records
        )

    def total_estimated_cost(self) -> float:
        """Return the total estimated model cost."""

        return sum(
            record.estimated_cost
            for record in self.records
        )