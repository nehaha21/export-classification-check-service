"""
Tool interfaces used by the agent layer.

The agent depends on these interfaces rather than on the internal
implementation of the retrieval system.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RetrievedPassage:
    """A passage returned by the retrieval layer."""

    passage_id: str
    text: str
    source: str
    metadata: dict[str, Any]


class RetrievalInterface(Protocol):
    """
    Interface expected from the retrieval component.

    Person 3's retrieval implementation should provide these operations.
    The agent layer should not depend on how retrieval is implemented.
    """

    def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        """Search the approved corpus for relevant passages."""
        ...

    def lookup_clause(
        self,
        clause_id: str,
    ) -> RetrievedPassage | None:
        """Retrieve a specific clause or provision by identifier."""
        ...

    def get_request_fields(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the fields from the classification request needed by the agent."""
        ...


class ClassificationTools:
    """
    Agent-facing wrapper around the retrieval interface.

    This gives the reasoning layer three explicit tools:
    1. corpus search
    2. clause lookup
    3. request-field lookup
    """

    def __init__(self, retrieval: RetrievalInterface) -> None:
        self._retrieval = retrieval

    def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        return self._retrieval.search_corpus(query, top_k=top_k)

    def lookup_clause(
        self,
        clause_id: str,
    ) -> RetrievedPassage | None:
        return self._retrieval.lookup_clause(clause_id)

    def get_request_fields(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._retrieval.get_request_fields(request)