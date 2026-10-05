"""
Output filtering for classification-agent responses.

The filter prevents incomplete or unsafe model output from being treated as a
final classification result.
"""

from dataclasses import dataclass


@dataclass
class OutputFilterResult:
    """Result of output validation."""

    allowed: bool
    reason: str | None = None


class OutputFilter:
    """
    Deterministic validation layer for agent output.

    This filter checks for required classification fields and rejects outputs
    that explicitly claim unsupported certainty.
    """

    _REQUIRED_FIELDS = (
        "proposed_classification",
        "rule_applied",
        "reasoning",
        "citations",
    )

    def check(self, output: str) -> OutputFilterResult:
        """
        Check whether a model response contains the minimum required
        classification information.
        """

        if not output or not output.strip():
            return OutputFilterResult(
                allowed=False,
                reason="Model output is empty.",
            )

        normalized = output.lower()

        missing_fields = [
            field
            for field in self._REQUIRED_FIELDS
            if field not in normalized
        ]

        if missing_fields:
            return OutputFilterResult(
                allowed=False,
                reason=(
                    "Model output is missing required classification fields: "
                    + ", ".join(missing_fields)
                ),
            )

        unsafe_claims = (
            "guaranteed classification",
            "authoritative customs determination",
            "legally binding classification",
        )

        for claim in unsafe_claims:
            if claim in normalized:
                return OutputFilterResult(
                    allowed=False,
                    reason=(
                        "Model output contains an unsupported authoritative "
                        f"claim: {claim}"
                    ),
                )

        return OutputFilterResult(allowed=True)