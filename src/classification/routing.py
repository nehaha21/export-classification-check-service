"""
Routing decisions for classification outcomes.

The service supports three routing outcomes:
- clear-for-filing
- seek-product-clarification
- specialist-classification-review

Unresolved GRI 3 ties and products outside the supported chapters must be
escalated rather than classified by guesswork.
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

    def clear_for_filing(
        self,
        reason: str,
    ) -> RoutingDecision:
        """Route a verified classification for filing."""

        return RoutingDecision(
            status=CLEAR_FOR_FILING,
            reason=reason,
        )

    def seek_product_clarification(
        self,
        reason: str,
    ) -> RoutingDecision:
        """Request additional product information."""

        return RoutingDecision(
            status=SEEK_PRODUCT_CLARIFICATION,
            reason=reason,
        )

    def specialist_review(
        self,
        reason: str,
    ) -> RoutingDecision:
        """Escalate an unresolved classification to specialist review."""

        return RoutingDecision(
            status=SPECIALIST_CLASSIFICATION_REVIEW,
            reason=reason,
        )

    def unresolved_gri3_tie(
        self,
        headings: list[str],
    ) -> RoutingDecision:
        """
        Escalate when two or more headings remain equally supported after GRI 3.

        The system must not select a heading arbitrarily when the retrieved
        evidence does not establish a unique result.
        """

        heading_text = ", ".join(headings)

        return self.specialist_review(
            reason=(
                "Two or more headings remain equally supported after "
                f"application of GRI 3: {heading_text}"
            ),
        )

    def outside_supported_chapters(
        self,
        chapter: str,
    ) -> RoutingDecision:
        """
        Escalate when the product falls outside the supported corpus scope.
        """

        return self.specialist_review(
            reason=(
                "The product appears to fall outside the supported "
                f"chapters: {chapter}"
            ),
        )

    def route(
        self,
        *,
        verified: bool,
        unresolved_gri3_tie: bool = False,
        tied_headings: list[str] | None = None,
        outside_supported_chapters: bool = False,
        chapter: str | None = None,
        needs_product_clarification: bool = False,
    ) -> RoutingDecision:
        """
        Determine the final routing decision.

        Escalation conditions take precedence over clear-for-filing so that
        an unresolved or out-of-scope classification cannot be returned as
        a confident result.
        """

        if needs_product_clarification:
            return self.seek_product_clarification(
                "Additional product information is required."
            )

        if outside_supported_chapters:
            return self.outside_supported_chapters(
                chapter or "unspecified",
            )

        if unresolved_gri3_tie:
            return self.unresolved_gri3_tie(
                tied_headings or [],
            )

        if not verified:
            return self.specialist_review(
                "The proposed classification was not verified.",
            )

        return self.clear_for_filing(
            "The proposed classification was verified against retrieved evidence."
        )