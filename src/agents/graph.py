"""
LangGraph orchestration for classification and verification.

Flow:
    request
      ↓
prepare request
      ↓
agent chooses tool
      ↓
tool execution / retrieval
      ↓
classification agent
      ↓
verification agent
      ↓
verified result
    OR
bounded retry with a different retrieval query
    OR
specialist escalation

The workflow is bounded by:
- maximum classification attempts;
- maximum tool calls;
- maximum graph steps;
- maximum wall-clock execution time.
"""

from dataclasses import dataclass, field
from time import monotonic
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
    steps: int = 0
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
    steps: int

    tool_choice: str
    run_started_at: float

    escalation_reason: str | None


class ClassificationGraph:
    """
    LangGraph-based classification workflow.

    The graph coordinates two distinct agents:

    - ClassificationAgent proposes a classification from retrieved evidence.
    - VerificationAgent verifies the proposal against that evidence.

    The classification agent is given three explicit tools:

    - search_corpus
    - lookup_clause
    - get_request_fields

    The orchestration layer enforces execution bounds so the workflow cannot
    continue indefinitely.
    """

    def __init__(
        self,
        *,
        classification_agent: ClassificationAgent,
        verification_agent: VerificationAgent,
        max_attempts: int = 2,
        max_tool_calls: int = 4,
        max_steps: int = 10,
        max_wall_clock_seconds: float = 30.0,
    ) -> None:
        self.classification_agent = classification_agent
        self.verification_agent = verification_agent

        self.max_attempts = max_attempts
        self.max_tool_calls = max_tool_calls
        self.max_steps = max_steps
        self.max_wall_clock_seconds = max_wall_clock_seconds

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")

        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1.")

        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        if self.max_wall_clock_seconds <= 0:
            raise ValueError(
                "max_wall_clock_seconds must be greater than zero."
            )

        self._graph = self._build_graph()

    def _build_graph(self):
        """Build and compile the LangGraph workflow."""

        workflow = StateGraph(GraphState)

        workflow.add_node(
            "prepare_request",
            self._prepare_request,
        )

        workflow.add_node(
            "choose_tool",
            self._choose_tool,
        )

        workflow.add_node(
            "retrieve",
            self._retrieve,
        )

        workflow.add_node(
            "classify",
            self._classify,
        )

        workflow.add_node(
            "verify",
            self._verify,
        )

        workflow.add_node(
            "finalize",
            self._finalize,
        )

        workflow.add_edge(
            START,
            "prepare_request",
        )

        workflow.add_edge(
            "prepare_request",
            "choose_tool",
        )

        workflow.add_conditional_edges(
            "choose_tool",
            self._after_tool_choice,
            {
                "retrieve": "retrieve",
                "finalize": "finalize",
            },
        )

        workflow.add_conditional_edges(
            "retrieve",
            self._after_retrieval,
            {
                "classify": "classify",
                "finalize": "finalize",
            },
        )

        workflow.add_edge(
            "classify",
            "verify",
        )

        workflow.add_conditional_edges(
            "verify",
            self._after_verification,
            {
                "finalize": "finalize",
                "retry": "choose_tool",
            },
        )

        workflow.add_edge(
            "finalize",
            END,
        )

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
            "steps": 0,
            "tool_choice": "",
            "run_started_at": monotonic(),
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
            proposed_answer=final_state.get(
                "proposed_answer",
            ),
            verification_result=final_state.get(
                "verification_result",
            ),
            passages=final_state.get(
                "passages",
                [],
            ),
            attempts=final_state.get(
                "attempts",
                0,
            ),
            tool_calls=final_state.get(
                "tool_calls",
                0,
            ),
            steps=final_state.get(
                "steps",
                0,
            ),
            escalation_reason=final_state.get(
                "escalation_reason",
            ),
        )

    def _prepare_request(
        self,
        state: GraphState,
    ) -> GraphState:
        """Obtain the request fields required for classification."""

        state = self._advance_step(state)

        if self._bounds_exceeded(state):
            return self._bounded_escalation(
                state,
                reason=(
                    "Agent execution bounds were reached while "
                    "preparing the request."
                ),
            )

        request_fields = self.classification_agent.identify_request(
            state["request"],
        )

        tool_calls = state.get("tool_calls", 0) + 1

        article = request_fields.get(
            "article",
        ) or request_fields.get(
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

        if tool_calls > self.max_tool_calls:
            return self._bounded_escalation(
                {
                    **state,
                    "request_fields": request_fields,
                    "article": article,
                    "tool_calls": tool_calls,
                },
                reason="Maximum tool-call limit reached.",
            )

        return {
            **state,
            "request_fields": request_fields,
            "article": article,
            "tool_calls": tool_calls,
        }

    def _choose_tool(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Let the classification agent choose its next available tool.

        The available tool set is explicit and bounded. The graph then routes
        to the corresponding execution node.
        """

        state = self._advance_step(state)

        if self._bounds_exceeded(state):
            return self._bounded_escalation(
                state,
                reason=(
                    "Agent execution bounds were reached before "
                    "tool selection."
                ),
            )

        available_tools = [
            "search_corpus",
            "lookup_clause",
            "get_request_fields",
        ]

        tool_choice = self.classification_agent.choose_tool(
            request=state["request"],
            available_tools=available_tools,
        )

        if tool_choice not in available_tools:
            return self._bounded_escalation(
                {
                    **state,
                    "tool_choice": tool_choice,
                },
                reason=(
                    f"Agent selected an unsupported tool: {tool_choice}"
                ),
            )

        return {
            **state,
            "tool_choice": tool_choice,
        }

    def _after_tool_choice(
        self,
        state: GraphState,
    ) -> str:
        """Route according to the tool selected by the classification agent."""

        if state.get("status") in {
            SEEK_PRODUCT_CLARIFICATION,
            SPECIALIST_CLASSIFICATION_REVIEW,
        }:
            return "finalize"

        if state.get("tool_choice") == "search_corpus":
            return "retrieve"

        # The remaining tools are available to the agent interface, but their
        # execution requires additional structured inputs that are not present
        # in the current request state. Do not silently invent those inputs.
        return "finalize"

    def _retrieve(
        self,
        state: GraphState,
    ) -> GraphState:
        """
        Execute the corpus-search tool selected by the classification agent.

        On a retry, the query is changed so the workflow does not simply
        repeat the same retrieval request.
        """

        state = self._advance_step(state)

        if self._bounds_exceeded(state):
            return self._bounded_escalation(
                state,
                reason=(
                    "Agent execution bounds were reached during retrieval."
                ),
            )

        if state.get("tool_calls", 0) >= self.max_tool_calls:
            return self._bounded_escalation(
                state,
                reason="Maximum tool-call limit reached.",
            )

        article = state.get(
            "article",
            "",
        )

        if not article:
            return {
                **state,
                "status": SEEK_PRODUCT_CLARIFICATION,
                "escalation_reason": (
                    "No article description is available for retrieval."
                ),
            }

        if state.get("attempts", 0) == 0:
            query = (
                f"{article}. "
                "Find candidate tariff headings, especially headings that "
                "specifically name the article. Also retrieve relevant "
                "material-based headings, section notes, chapter notes, "
                "and General Rules for Interpretation (GRI) provisions "
                "needed to compare the candidates."
            )
        else:
            previous_answer = state.get(
                "proposed_answer",
                "",
            ) or ""

            query = (
                f"{article} tariff heading classification "
                f"GRI section chapter notes. "
                f"Previous proposal: {previous_answer}"
            )

        passages = self.classification_agent.retrieve_candidates(
            query,
            top_k=5,
        )

        tool_calls = state.get("tool_calls", 0) + 1

        if tool_calls > self.max_tool_calls:
            return self._bounded_escalation(
                {
                    **state,
                    "tool_calls": tool_calls,
                },
                reason="Maximum tool-call limit reached after retrieval.",
            )

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

        if not state.get("passages"):
            return "finalize"

        if self._bounds_exceeded(state):
            return "finalize"

        return "classify"

    def _classify(
        self,
        state: GraphState,
    ) -> GraphState:
        """Ask the classification agent for a proposal."""

        state = self._advance_step(state)

        if self._bounds_exceeded(state):
            return self._bounded_escalation(
                state,
                reason=(
                    "Agent execution bounds were reached before "
                    "classification."
                ),
            )

        attempt = state.get(
            "attempts",
            0,
        ) + 1

        if attempt > self.max_attempts:
            return self._bounded_escalation(
                {
                    **state,
                    "attempts": attempt,
                },
                reason="Maximum classification-attempt limit reached.",
            )

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

        state = self._advance_step(state)

        if self._bounds_exceeded(state):
            return self._bounded_escalation(
                state,
                reason=(
                    "Agent execution bounds were reached before "
                    "verification."
                ),
            )

        verification_result = self.verification_agent.verify(
            proposed_answer=state.get(
                "proposed_answer",
                "",
            ) or "",
            passages=state.get(
                "passages",
                [],
            ),
        )

        return {
            **state,
            "verification_result": verification_result,
        }

    def _after_verification(
        self,
        state: GraphState,
    ) -> str:
        """
        Decide whether to finalize or retry with a different search.

        A successful verification always proceeds to finalization.
        Failed verification may retry only while all execution bounds remain.
        """

        if self._verification_passed(
            state.get(
                "verification_result",
                "",
            ),
        ):
            return "finalize"

        if state.get("attempts", 0) >= self.max_attempts:
            return "finalize"

        if state.get("tool_calls", 0) >= self.max_tool_calls:
            return "finalize"

        if self._bounds_exceeded(state):
            return "finalize"

        return "retry"

    def _finalize(
        self,
        state: GraphState,
    ) -> GraphState:
        """Produce the final safe workflow status."""

        state = self._advance_step(state)

        if self._verification_passed(
            state.get(
                "verification_result",
                "",
            ),
        ):
            return {
                **state,
                "status": CLEAR_FOR_FILING,
                "escalation_reason": None,
            }

        if state.get("status") == SEEK_PRODUCT_CLARIFICATION:
            return state

        return {
            **state,
            "status": SPECIALIST_CLASSIFICATION_REVIEW,
            "proposed_answer": None,
            "escalation_reason": (
                state.get("escalation_reason")
                or (
                    "The proposed classification could not be verified "
                    "against the retrieved evidence."
                )
            ),
        }

    def _advance_step(
        self,
        state: GraphState,
    ) -> GraphState:
        """Increment the bounded graph-step counter."""

        return {
            **state,
            "steps": state.get(
                "steps",
                0,
            ) + 1,
        }

    def _bounds_exceeded(
        self,
        state: GraphState,
    ) -> bool:
        """Check graph-step and wall-clock execution limits."""

        steps_exceeded = (
            state.get(
                "steps",
                0,
            )
            > self.max_steps
        )

        started_at = state.get(
            "run_started_at",
        )

        if started_at is None:
            return True

        elapsed_seconds = monotonic() - started_at

        time_exceeded = (
            elapsed_seconds
            > self.max_wall_clock_seconds
        )

        return steps_exceeded or time_exceeded

    @staticmethod
    def _bounded_escalation(
        state: GraphState,
        *,
        reason: str,
    ) -> GraphState:
        """Convert an execution-bound failure into specialist review."""

        return {
            **state,
            "status": SPECIALIST_CLASSIFICATION_REVIEW,
            "proposed_answer": None,
            "escalation_reason": reason,
        }

    @staticmethod
    def _verification_passed(
        verification_result: str,
    ) -> bool:
        """
        Check whether the verification agent explicitly passed the result.

        The verification agent must explicitly return a true verification
        result before the graph can produce clear-for-filing.
        """

        normalized = verification_result.lower()

        return (
            "verified: true" in normalized
            or '"verified": true' in normalized
            or "'verified': true" in normalized
        )