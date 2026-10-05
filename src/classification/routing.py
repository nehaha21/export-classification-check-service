"""
Routing decisions for classification outcomes.

The service supports three routing outcomes:
- clear-for-filing
- seek-product-clarification
- specialist-classification-review
"""

from dataclasses import dataclass


CLEAR_FOR_FILING = "clear-for-filing"
SEEK_PRODUCT_CLARIFICATION = "seek-product-clarification"
SPECIALIST_CLASSIFICATION_REVIEW = "specialist-classification-review"


@dataclass
class RoutingDecision:
    """Routing result for a classification run."""

    status: str
    reason: str


class ClassificationRouter:
    """Determine the next workflow outcome from classification evidence."""

    def clear_for_filing(self, reason: str) -> RoutingDecision:
        """Route a verified classification for filing."""

        return RoutingDecision(
            status=CLEAR_FOR_FILING,
            reason=reason,
        )

    def seek_product_clarification(self, reason: str) -> RoutingDecision:
        """Request additional product information."""

        return RoutingDecision(
            status=SEEK_PRODUCT_CLARIFICATION,
            reason=reason,
        )

    def specialist_review(self, reason: str) -> RoutingDecision:
        """Escalate an unresolved classification to specialist review."""

        return RoutingDecision(
            status=SPECIALIST_CLASSIFICATION_REVIEW,
            reason=reason,
        )