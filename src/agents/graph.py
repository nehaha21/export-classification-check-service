"""
LangGraph orchestration for classification and verification.

Flow:
    request
      ↓
classification agent
      ↓
verification agent
      ↓
verified result OR bounded retry
      ↓
final result / specialist review
"""

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..classification.routing import (
    CLEAR_FOR_FILING,
    SEEK_PRODUCT_CLARIFICATION,
    SPECIALIST_CLASSIFICATION_REVIEW,
)
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


class GraphState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    request: dict[str, Any]
    request_fields: dict[str, Any]
    article: str
    passages: list[RetrievedPassage]
    proposed_answer: str | None
    verification_result: str | None
    status: str
    attempts: int
    tool_calls: int
    escalation_reason: str | None


class ClassificationGraph:
    """
    LangGraph-based classification workflow.

    The graph coordinates two distinct agents:
    - ClassificationAgent proposes a classification from retrieved evidence.
    - VerificationAgent verifies the proposal against that evidence.

    Attempts and retrieval calls are explicitly bounded.
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
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""

        workflow = StateGraph(GraphState)

        workflow.add_node("prepare_request", self._prepare_request)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("classify", self._classify)
        workflow.add_node("verify", self._verify)
        workflow.add_node("retry_retrieve", self._retry_retrieve)
        workflow.add_node("finalize", self._finalize)

        workflow.add_edge(START, "prepare_request")
        workflow.add_edge("prepare_request", "retrieve")
        workflow.add_conditional_edges(
            "retrieve",
            self._after_retrieval,
            {
                "classify": "classify",
                "finalize": "finalize",
            },
        )
        workflow.add_edge("classify", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._after_verification,
            {
                "finalize": "finalize",
                "retry": "retry_retrieve",
            },
        )
        workflow.add_edge("retry_retrieve", "classify")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    def run(
        self,
        request: dict[str, Any],
    ) -> AgentRunResult:
        """Run the compiled LangGraph workflow."""

        initial_state: GraphState = {
            "request": request,
            "attempts": 0,
            "tool_calls": 0,
            "passages": [],
            "proposed_answer": None,
            "verification_result": None,
            "status": SPECIALIST_CLASSIFICATION_REVIEW,
            "escalation_reason": None,
        }

        final_state = self._graph.invoke(initial_state)

        return AgentRunResult(
            status=final_state.get(
                "status",
                SPECIALIST_CLASSIFICATION_REVIEW,
            ),
            proposed_answer=final_state.get("proposed_answer"),
            verification_result=final_state.get("verification_result"),
            passages=final_state.get("passages", []),
            attempts=final_state.get("attempts", 0),
            tool_calls=final_state.get("tool_calls", 0),
            escalation_reason=final_state.get("escalation_reason"),
        )

    def _prepare_request(
        self,
        state: GraphState,
    ) -> GraphState:
        """Extract the fields required by the classification agent."""

        request_fields = self.classification_agent.identify_request(
            state["request"],
        )

        tool_calls = state.get("tool_calls", 0) + 1

        article = request_fields.get("article") or request_fields.get(
            "product_description",
            "",
        )

        if not article:
            return {
                **state,
                "request_fields": request_fields,
                "article": "",
                "tool_calls": tool_calls,
                "status": SEEK_PRODUCT_CLARIFICATION,
                "escalation_reason": (
                    "No product description was provided."
                ),
            }

        return {
            **state,
            "request_fields": request_fields,
            "article": article,
            "tool_calls": tool_calls,
        }

    def _retrieve(
        self,
        state: GraphState,
    ) -> GraphState:
        """Retrieve candidate tariff passages."""

        if state.get("status") == SEEK_PRODUCT_CLARIFICATION:
            return state

        if state.get("tool_calls", 0) >= self.max_tool_calls:
            return {
                **state,
                "status": SPECIALIST_CLASSIFICATION_REVIEW,
                "escalation_reason": (
                    "Tool-call limit reached before retrieval."
                ),
            }

        passages = self.classification_agent.retrieve_candidates(
            state["article"],
            top_k=5,
        )

        tool_calls = state.get("tool_calls", 0) + 1

        if not passages:
            return {
                **state,
                "passages": [],
                "tool_calls": tool_calls,
                "status": SPECIALIST_CLASSIFICATION_REVIEW,
                "escalation_reason": (
                    "No supporting corpus passages were retrieved."
                ),
            }

        return {
            **state,
            "passages": passages,
            "tool_calls": tool_calls,
        }

    def _classify(
        self,
        state: GraphState,
    ) -> GraphState:
        """Ask the classification agent for a proposal."""

        attempt = state.get("attempts", 0) + 1

        proposed_answer = self.classification_agent.propose(
            state["request"],
            state["passages"],
        )

        return {
            **state,
            "proposed_answer": proposed_answer,
            "attempts": attempt,
        }

    def _verify(
        self,
        state: GraphState,
    ) -> GraphState:
        """Ask the verification agent to verify the proposal."""

        verification_result = self.verification_agent.verify(
            proposed_answer=state["proposed_answer"] or "",
            passages=state["passages"],
        )

        return {
            **state,
            "verification_result": verification_result,
        }

    def _retry_retrieve(
        self,
        state: GraphState,
    ) -> GraphState:
        """Retrieve evidence using a different query after failed verification."""

        if state.get("tool_calls", 0) >= self.max_tool_calls:
            return {
                **state,
                "status": SPECIALIST_CLASSIFICATION_REVIEW,
                "escalation_reason": (
                    "Maximum retrieval tool calls reached after "
                    "verification failure."
                ),
            }

        retry_query = self._build_retry_query(
            article=state["article"],
            previous_answer=state.get("proposed_answer") or "",
        )

        passages = self.classification_agent.retrieve_candidates(
            retry_query,
            top_k=5,
        )

        tool_calls = state.get("tool_calls", 0) + 1

        if not passages:
            return {
                **state,
                "passages": [],
                "tool_calls": tool_calls,
                "status": SPECIALIST_CLASSIFICATION_REVIEW,
                "escalation_reason": (
                    "Verification failed and the retry search returned "
                    "no supporting passages."
                ),
            }

        return {
            **state,
            "passages": passages,
            "tool_calls": tool_calls,
        }

    def _finalize(
        self,
        state: GraphState,
    ) -> GraphState:
        """Produce the final safe workflow status."""

        if state.get("status") == SEEK_PRODUCT_CLARIFICATION:
            return state

        verification_result = state.get("verification_result", "")

        if self._verification_passed(verification_result):
            return {
                **state,
                "status": CLEAR_FOR_FILING,
                "escalation_reason": None,
            }

        return {
            **state,
            "status": SPECIALIST_CLASSIFICATION_REVIEW,
            "proposed_answer": None,
            "escalation_reason": (
                "The proposed classification could not be verified "
                "against the retrieved evidence."
            ),
        }

    def _after_retrieval(
        self,
        state: GraphState,
    ) -> str:
        """Choose classification or finalization after retrieval."""

        if state.get("status") in {
            SEEK_PRODUCT_CLARIFICATION,
            SPECIALIST_CLASSIFICATION_REVIEW,
        }:
            return "finalize"

        return "classify"

    def _after_verification(
        self,
        state: GraphState,
    ) -> str:
        """Choose finalization or bounded retry after verification."""

        if self._verification_passed(
            state.get("verification_result", ""),
        ):
            return "finalize"

        if state.get("attempts", 0) >= self.max_attempts:
            return "finalize"

        if state.get("tool_calls", 0) >= self.max_tool_calls:
            return "finalize"

        return "retry"

    @staticmethod
    def _verification_passed(
        verification_result: str,
    ) -> bool:
        """Check whether the verification agent explicitly passed the result."""

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
        """Build a different retrieval query after failed verification."""

        return (
            f"{article} tariff heading classification "
            f"GRI section chapter notes. "
            f"Previous proposal: {previous_answer}"
        )