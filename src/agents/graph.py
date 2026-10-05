"""
Agent orchestration graph.

The graph coordinates the classification and verification agents.

Flow:
    request
      ↓
classification agent
      ↓
verification agent
      ↓
verified result OR retry/escalation

The graph uses bounded attempts so an incomplete or unverifiable run
cannot continue indefinitely.
"""

from dataclasses import dataclass, field
from typing import Any

from .classification_agent import ClassificationAgent
from .tools import RetrievedPassage
from .verification_agent import VerificationAgent


@dataclass
class AgentRunResult:
    """Final state produced by the agent graph."""

    status: str
    proposed_answer: str | None = None
    verification_result: str | None = None
    passages: list[RetrievedPassage] = field(default_factory=list)
    attempts: int = 0
    tool_calls: int = 0
    escalation_reason: str | None = None


class ClassificationGraph:
    """
    Coordinates the classification and verification agents.

    Responsibilities:
    - obtain the required request fields;
    - retrieve candidate tariff evidence;
    - ask the classification agent for a proposal;
    - pass the proposal to the verification agent;
    - retry with a different retrieval query when verification fails;
    - escalate instead of returning an unverified answer.

    The number of attempts and retrieval tool calls is bounded.
    """

    def __init__(
        self,
        *,
        classification_agent: ClassificationAgent,
        verification_agent: VerificationAgent,
        max_attempts: int = 2,
        max_tool_calls: int = 4,
    ) -> None:
        self.classification_agent = classification_agent
        self.verification_agent = verification_agent
        self.max_attempts = max_attempts
        self.max_tool_calls = max_tool_calls

    def run(
        self,
        request: dict[str, Any],
    ) -> AgentRunResult:
        """Run classification followed by verification."""

        tool_calls = 0

        request_fields = self.classification_agent.identify_request(request)
        tool_calls += 1

        article = request_fields.get("article") or request_fields.get(
            "product_description",
            "",
        )

        if not article:
            return AgentRunResult(
                status="seek-product-clarification",
                tool_calls=tool_calls,
                escalation_reason="No product description was provided.",
            )

        if tool_calls >= self.max_tool_calls:
            return AgentRunResult(
                status="specialist-classification-review",
                tool_calls=tool_calls,
                escalation_reason="Tool-call limit reached before retrieval.",
            )

        passages = self.classification_agent.retrieve_candidates(
            article,
            top_k=5,
        )
        tool_calls += 1

        if not passages:
            return AgentRunResult(
                status="specialist-classification-review",
                attempts=0,
                tool_calls=tool_calls,
                escalation_reason="No supporting corpus passages were retrieved.",
            )

        for attempt in range(1, self.max_attempts + 1):
            proposed_answer = self.classification_agent.propose(
                request,
                passages,
            )

            verification_result = self.verification_agent.verify(
                proposed_answer=proposed_answer,
                passages=passages,
            )

            if self._verification_passed(verification_result):
                return AgentRunResult(
                    status="clear-for-filing",
                    proposed_answer=proposed_answer,
                    verification_result=verification_result,
                    passages=passages,
                    attempts=attempt,
                    tool_calls=tool_calls,
                )

            if attempt == self.max_attempts:
                return AgentRunResult(
                    status="specialist-classification-review",
                    proposed_answer=None,
                    verification_result=verification_result,
                    passages=passages,
                    attempts=attempt,
                    tool_calls=tool_calls,
                    escalation_reason=(
                        "The proposed classification could not be verified "
                        "against the retrieved evidence."
                    ),
                )

            if tool_calls >= self.max_tool_calls:
                return AgentRunResult(
                    status="specialist-classification-review",
                    proposed_answer=None,
                    verification_result=verification_result,
                    passages=passages,
                    attempts=attempt,
                    tool_calls=tool_calls,
                    escalation_reason=(
                        "The maximum number of retrieval tool calls was reached "
                        "after verification failed."
                    ),
                )

            retry_query = self._build_retry_query(
                article=article,
                previous_answer=proposed_answer,
            )

            passages = self.classification_agent.retrieve_candidates(
                retry_query,
                top_k=5,
            )
            tool_calls += 1

            if not passages:
                return AgentRunResult(
                    status="specialist-classification-review",
                    verification_result=verification_result,
                    attempts=attempt,
                    tool_calls=tool_calls,
                    escalation_reason=(
                        "Verification failed and the retry search returned "
                        "no supporting passages."
                    ),
                )

        return AgentRunResult(
            status="specialist-classification-review",
            attempts=self.max_attempts,
            tool_calls=tool_calls,
            escalation_reason="Agent attempt limit reached.",
        )

    @staticmethod
    def _verification_passed(
        verification_result: str,
    ) -> bool:
        """
        Determine whether verification succeeded.

        The current implementation accepts an explicit 'verified: true'
        response. A later structured-output model interface can replace this
        parser without changing the graph's responsibilities.
        """

        normalized = verification_result.lower()

        return (
            "verified: true" in normalized
            or '"verified": true' in normalized
            or "'verified': true" in normalized
        )

    @staticmethod
    def _build_retry_query(
        *,
        article: str,
        previous_answer: str,
    ) -> str:
        """
        Build a different retrieval query after failed verification.

        The retry searches using classification evidence rather than simply
        repeating the original product description.
        """

        return (
            f"{article} tariff heading classification "
            f"GRI section chapter notes. "
            f"Previous proposal: {previous_answer}"
        )