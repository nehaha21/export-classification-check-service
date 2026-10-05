"""
Basic toxicity detection for agent inputs and outputs.

This module provides a deterministic first-layer check. It is intended as a
guardrail before content is passed through the agent workflow.
"""

from dataclasses import dataclass


@dataclass
class ToxicityCheckResult:
    """Result of a toxicity check."""

    flagged: bool
    reason: str | None = None


class ToxicityGuard:
    """
    Lightweight toxicity detector.

    This is deliberately conservative and deterministic. It is not intended
    to replace a dedicated toxicity model.
    """

    _PATTERNS = (
        "kill yourself",
        "go kill yourself",
        "i will hurt you",
        "i will attack you",
        "racial slur",
        "violent threat",
    )

    def check(self, text: str) -> ToxicityCheckResult:
        """
        Check text for known abusive or threatening patterns.
        """

        if not text or not text.strip():
            return ToxicityCheckResult(flagged=False)

        normalized = " ".join(text.lower().split())

        for pattern in self._PATTERNS:
            if pattern in normalized:
                return ToxicityCheckResult(
                    flagged=True,
                    reason="Potentially toxic or threatening content detected.",
                )

        return ToxicityCheckResult(flagged=False)

    def check_fields(
        self,
        fields: dict[str, object],
    ) -> ToxicityCheckResult:
        """Check textual request fields for potentially toxic content."""

        for value in fields.values():
            if not isinstance(value, str):
                continue

            result = self.check(value)

            if result.flagged:
                return result

        return ToxicityCheckResult(flagged=False)