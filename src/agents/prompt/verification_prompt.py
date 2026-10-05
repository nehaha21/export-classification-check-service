"""
Verification agent.

Responsible for checking that a proposed classification is supported by the
passages retrieved during the run.

An unsupported or unverifiable answer must not be treated as a valid result.
"""

from dataclasses import dataclass
from typing import Protocol

from .PROMPT.verification_prompt import (
    PROMPT_VERSION,
    VERIFICATION_SYSTEM_PROMPT,
)
from .tools import RetrievedPassage


class ModelClient(Protocol):
    """Minimal interface required from the configured language model."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        ...


@dataclass
class VerificationResult:
    """Outcome of verification."""

    verified: bool
    supported_claims: list[str]
    unsupported_claims: list[str]
    invalid_citations: list[str]
    reason: str
    prompt_version: str = PROMPT_VERSION


class VerificationAgent:
    """
    Agent responsible for self-verification of classification results.

    This agent receives the retrieved passages from the orchestration layer.
    It does not perform its own hidden retrieval.
    """

    def __init__(self, *, model: ModelClient) -> None:
        self.model = model

    def verify(
        self,
        *,
        proposed_answer: str,
        passages: list[RetrievedPassage],
    ) -> str:
        """
        Verify the proposed answer against the retrieved passages.

        The returned model output is interpreted by the orchestration layer.
        """

        evidence = "\n\n".join(
            (
                f"[Passage ID: {passage.passage_id}]\n"
                f"Source: {passage.source}\n"
                f"{passage.text}"
            )
            for passage in passages
        )

        user_prompt = f"""
Proposed classification:
{proposed_answer}

Retrieved passages:
{evidence}

Verify the proposed classification strictly against these passages.

Do not use outside knowledge.
Do not assume that an unsupported citation is valid.
If an important claim cannot be supported by the supplied passages,
verification must fail.
""".strip()

        return self.model.generate(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )