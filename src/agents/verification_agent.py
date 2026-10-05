"""
Verification agent.

Responsible for checking that a proposed classification is supported by the
passages retrieved during the run.

An unsupported or unverifiable answer must not be treated as a valid result.
"""

from dataclasses import dataclass
from typing import Any, Protocol

from .prompt.classification_prompt import PROMPT_VERSION
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


VERIFICATION_SYSTEM_PROMPT = """
You are the verification agent for the Export Classification Check Service.

Your responsibility is to verify a proposed classification against the
retrieved passages provided to you.

Check all of the following:

1. The proposed classification is supported by retrieved text.
2. The cited passages actually contain the evidence claimed by the proposal.
3. The stated GRI rule is supported by the retrieved evidence.
4. Any cited heading is present in the retrieved evidence.
5. Applicable section or chapter notes used by the proposal are supported.
6. The answer does not rely on facts or tariff provisions that were not
   retrieved.

A citation is valid only when it resolves to one of the supplied passages.

If the evidence does not support the proposed answer, mark verification as
failed. Do not make up missing evidence and do not approve an unsupported
classification.

Return:
- verified: true or false
- supported_claims
- unsupported_claims
- invalid_citations
- reason
""".strip()


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