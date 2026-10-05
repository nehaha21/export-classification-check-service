"""
Graceful fallback behavior for incomplete or unavailable agent runs.

The service must not return a confident classification when required
dependencies fail or when verification cannot be completed.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class FallbackResult:
    """Result returned when the normal workflow cannot complete."""

    status: str
    message: str
    details: dict[str, Any]


class FallbackHandler:
    """
    Convert incomplete workflow states into safe service outcomes.

    The fallback never produces a classification on behalf of the failed
    workflow.
    """

    def unavailable_dependency(
        self,
        dependency: str,
        *,
        reason: str,
    ) -> FallbackResult:
        """Handle an unavailable model, retrieval service, or other dependency."""

        return FallbackResult(
            status="specialist-classification-review",
            message=(
                "The classification workflow could not be completed because "
                f"the {dependency} dependency was unavailable."
            ),
            details={
                "dependency": dependency,
                "reason": reason,
            },
        )

    def verification_failed(
        self,
        *,
        reason: str,
        attempts: int,
    ) -> FallbackResult:
        """Handle a classification that could not be verified."""

        return FallbackResult(
            status="specialist-classification-review",
            message=(
                "The proposed classification could not be verified against "
                "the retrieved evidence."
            ),
            details={
                "reason": reason,
                "attempts": attempts,
            },
        )

    def insufficient_information(
        self,
        *,
        missing_fields: list[str],
    ) -> FallbackResult:
        """Request clarification when required product information is missing."""

        return FallbackResult(
            status="seek-product-clarification",
            message=(
                "Additional product information is required before "
                "classification can proceed."
            ),
            details={
                "missing_fields": missing_fields,
            },
        )