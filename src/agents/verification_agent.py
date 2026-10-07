"""
Verification agent.

Responsible for checking that a proposed classification is supported by the
passages retrieved during the run.

An unsupported or unverifiable answer must not be treated as a valid result.
"""

from dataclasses import dataclass
from typing import Protocol

from .prompt.verification_prompt import (
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

        Citation identifiers are checked deterministically before the model
        performs semantic verification. This prevents an answer from being
        treated as verified when it cites evidence that was not retrieved.
        """

        invalid_citations = self._find_invalid_citations(
            proposed_answer=proposed_answer,
            passages=passages,
        )

        if invalid_citations:
            return (
                "verified: false\n"
                "supported_claims: []\n"
                "unsupported_claims: "
                f"['Citation(s) do not resolve to retrieved passages: "
                f"{', '.join(invalid_citations)}']\n"
                f"invalid_citations: {invalid_citations}\n"
                "reason: One or more citations do not resolve to retrieved "
                "evidence."
            )

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
Every citation must correspond to one of the supplied Passage IDs.

If an important claim cannot be supported by the supplied passages,
verification must fail.

Return:
- verified
- supported_claims
- unsupported_claims
- invalid_citations
- reason
""".strip()

        return self.model.generate(
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    @staticmethod
    def _find_invalid_citations(
        *,
        proposed_answer: str,
        passages: list[RetrievedPassage],
    ) -> list[str]:
        """
        Find citation identifiers in the proposed answer that were not
        present in the retrieved passages.

        The classification agent is expected to cite passages using the
        passage identifier, for example:

            passage_id: chapter-84-chunk-2

        or:

            [Passage ID: chapter-84-chunk-2]
        """

        import re

        valid_ids = {
            passage.passage_id
            for passage in passages
        }

        cited_ids = set(
            re.findall(
                r"(?:passage_id\s*:\s*|\[Passage ID:\s*)([^\]\n,]+)",
                proposed_answer,
                flags=re.IGNORECASE,
            )
        )

        cited_ids = {
            citation.strip()
            for citation in cited_ids
        }

        return sorted(
            citation
            for citation in cited_ids
            if citation not in valid_ids
        )