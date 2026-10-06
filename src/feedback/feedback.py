"""
Feedback capture for classification outcomes.

The feedback component stores structured human feedback so that classification
runs can later be evaluated against reviewer outcomes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FeedbackRecord:
    """Feedback associated with one classification run."""

    correlation_id: str
    outcome: str
    reviewer_comment: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)


class FeedbackStore:
    """In-memory feedback store for the initial implementation."""

    def __init__(self) -> None:
        self.records: list[FeedbackRecord] = []

    def add(
        self,
        *,
        correlation_id: str,
        outcome: str,
        reviewer_comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FeedbackRecord:
        """Store feedback for a classification run."""

        record = FeedbackRecord(
            correlation_id=correlation_id,
            outcome=outcome,
            reviewer_comment=reviewer_comment,
            metadata=metadata or {},
        )

        self.records.append(record)
        return record

    def get_for_run(
        self,
        correlation_id: str,
    ) -> list[FeedbackRecord]:
        """Return all feedback records for a classification run."""

        return [
            record
            for record in self.records
            if record.correlation_id == correlation_id
        ]
    
    def outcome_counts(self) -> dict[str, int]:
        """Return the number of feedback records for each outcome."""

        counts: dict[str, int] = {}

        for record in self.records:
            counts[record.outcome] = counts.get(record.outcome, 0) + 1

        return counts

    def total_feedback(self) -> int:
        """Return the total number of feedback records."""

        return len(self.records)