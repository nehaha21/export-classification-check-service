"""
Prompt-injection detection for the classification service.

The classifier processes product descriptions and retrieved corpus content.
This module detects common attempts to manipulate the agent's instructions
through those inputs.
"""

from dataclasses import dataclass


@dataclass
class InjectionCheckResult:
    """Result of a prompt-injection check."""

    blocked: bool
    reason: str | None = None
    matched_pattern: str | None = None


class InjectionGuard:
    """
    Lightweight prompt-injection guard.

    This is a first-layer deterministic check. It does not replace model-level
    safety controls, but it prevents obvious instruction-manipulation attempts
    from being passed directly into the agent workflow.
    """

    _PATTERNS = (
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore the previous instructions",
        "disregard previous instructions",
        "disregard all previous instructions",
        "forget previous instructions",
        "forget all previous instructions",
        "system prompt",
        "reveal your instructions",
        "reveal the system prompt",
        "show your system prompt",
        "developer message",
        "developer instructions",
        "override your instructions",
        "bypass your instructions",
        "do not follow the instructions",
        "follow these instructions instead",
    )

    def check(self, text: str) -> InjectionCheckResult:
        """
        Check text for known prompt-injection patterns.

        Matching is case-insensitive and whitespace-normalized.
        """

        normalized = " ".join(text.lower().split())

        for pattern in self._PATTERNS:
            if pattern in normalized:
                return InjectionCheckResult(
                    blocked=True,
                    reason="Potential prompt injection detected.",
                    matched_pattern=pattern,
                )

        return InjectionCheckResult(blocked=False)

    def check_fields(
        self,
        fields: dict[str, object],
    ) -> InjectionCheckResult:
        """
        Check textual request fields for prompt-injection patterns.

        Non-text values are ignored because they cannot contain natural-language
        instructions.
        """

        for value in fields.values():
            if not isinstance(value, str):
                continue

            result = self.check(value)

            if result.blocked:
                return result

        return InjectionCheckResult(blocked=False)