"""
Classification agent.

Responsible for:
- identifying the article and materials from the request;
- using the retrieval tools to find relevant tariff evidence;
- proposing a classification from the retrieved evidence.

The agent does not implement retrieval itself.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from .prompt.classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    PROMPT_VERSION,
)
from .tools import ClassificationTools, RetrievedPassage
from src.guardrails.injection import InjectionGuard
from src.guardrails.output_filter import OutputFilter
from src.guardrails.toxicity import ToxicityGuard


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
class ClassificationResult:
    """Result produced by the classification agent."""

    article: str
    materials: list[str]
    proposed_classification: str | None
    rule_applied: str | None
    reasoning: str
    citations: list[str]
    confidence: str
    unresolved_issues: list[str] = field(default_factory=list)
    prompt_version: str = PROMPT_VERSION


class ClassificationAgent:
    """
    Agent responsible for proposing a tariff classification.

    Retrieval is exposed through explicit tools so that the orchestration
    layer can decide when retrieval should occur.
    """

    def __init__(
        self,
        *,
        tools: ClassificationTools,
        model: ModelClient,
    ) -> None:
        self.tools = tools
        self.model = model
        self.injection_guard = InjectionGuard()
        self.output_filter = OutputFilter()
        self.toxicity_guard = ToxicityGuard()

    def identify_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract the request fields required for classification.

        The request-field lookup is delegated to the retrieval interface
        rather than reading the retrieval implementation directly.
        """
        return self.tools.get_request_fields(request)

    def retrieve_candidates(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        """
        Retrieve candidate tariff evidence for classification.

        The retrieval query explicitly asks for:
        - headings that name the article;
        - material-based candidate headings;
        - applicable section/chapter notes;
        - GRI provisions relevant to resolving the candidates.

        The retrieval implementation remains responsible for searching the corpus.
        """

        classification_query = (
            f"{query}. "
            "Find candidate tariff headings, especially headings that specifically "
            "name the article. Also retrieve relevant material-based headings, "
            "section notes, chapter notes, and General Rules for Interpretation "
            "(GRI) provisions needed to compare the candidates."
        )

        return self.tools.search_corpus(
            classification_query,
            top_k=top_k,
        )

    def build_classification_prompt(
        self,
        request_fields: dict[str, Any],
        passages: list[RetrievedPassage],
    ) -> str:
        """Build the model input from the request and retrieved evidence."""

        evidence = "\n\n".join(
            (
                f"[Passage ID: {passage.passage_id}]\n"
                f"Source: {passage.source}\n"
                f"{passage.text}"
            )
            for passage in passages
        )

        return f"""
Product request:
{request_fields}

Retrieved evidence:
{evidence}

Using only the retrieved evidence, propose the classification.

Remember:
- Apply the GRI in numerical order.
- Consider headings that specifically name the article.
- Consider applicable section and chapter notes.
- Do not invent evidence.
- If the evidence does not support a unique classification, say so.

Return the requested structured classification information.
""".strip()

    def propose(
        self,
        request: dict[str, Any],
        passages: list[RetrievedPassage],
    ) -> str:
        """
        Ask the model for a proposed classification using retrieved evidence.

        Verification is handled separately by the verification agent.
        """
        request_fields = self.identify_request(request)

        injection_result = self.injection_guard.check_fields(request_fields)
        if injection_result.blocked:
            raise ValueError(injection_result.reason)

        toxicity_result = self.toxicity_guard.check_fields(request_fields)
        if toxicity_result.flagged:
            raise ValueError(toxicity_result.reason)

        user_prompt = self.build_classification_prompt(
            request_fields,
            passages,
        )

        model_output = self.model.generate(
            system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        output_result = self.output_filter.check(model_output)
        if not output_result.allowed:
            raise ValueError(output_result.reason)

        toxicity_result = self.toxicity_guard.check(model_output)
        if toxicity_result.flagged:
            raise ValueError(toxicity_result.reason)

        return model_output