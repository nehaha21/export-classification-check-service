"""
Classification result models and basic classification validation.

This module does not determine tariff classifications by itself. A proposed
classification must be produced from retrieved corpus evidence by the agent
workflow.
"""

from dataclasses import dataclass, field

from .rules import SUPPORTED_GRI_RULES


@dataclass
class ClassificationProposal:
    """Structured representation of a proposed classification."""

    article: str
    proposed_classification: str | None
    rule_applied: str | None
    reasoning: str
    citations: list[str] = field(default_factory=list)
    confidence: str = "unknown"
    unresolved_issues: list[str] = field(default_factory=list)

    def has_supported_rule(self) -> bool:
        """Return whether the proposal names a supported GRI rule."""

        if not self.rule_applied:
            return False

        return self.rule_applied in SUPPORTED_GRI_RULES

    def has_citations(self) -> bool:
        """Return whether the proposal contains at least one citation."""

        return bool(self.citations)

    def is_complete(self) -> bool:
        """
        Check whether the proposal contains the minimum required fields.

        This does not establish that the classification is correct. It only
        checks structural completeness.
        """

        return bool(
            self.article.strip()
            and self.proposed_classification
            and self.rule_applied
            and self.reasoning.strip()
            and self.has_citations()
        )


class ClassificationValidator:
    """Validate the structure of a classification proposal."""

    def validate(
        self,
        proposal: ClassificationProposal,
    ) -> list[str]:
        """
        Return validation errors.

        An empty list means the proposal is structurally valid.
        """

        errors: list[str] = []

        if not proposal.article.strip():
            errors.append("Article description is missing.")

        if not proposal.proposed_classification:
            errors.append("Proposed classification is missing.")

        if not proposal.rule_applied:
            errors.append("GRI rule is missing.")
        elif not proposal.has_supported_rule():
            errors.append(
                f"Unsupported GRI rule: {proposal.rule_applied}"
            )

        if not proposal.reasoning.strip():
            errors.append("Classification reasoning is missing.")

        if not proposal.has_citations():
            errors.append("At least one citation is required.")

        return errors