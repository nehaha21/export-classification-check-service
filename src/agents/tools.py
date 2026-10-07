"""
Tool interfaces used by the agent layer.

The agent depends on these interfaces rather than on the internal
implementation of the retrieval system.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class RetrievedPassage:
    """A normalized passage returned to an agent."""

    passage_id: str
    text: str
    source: str
    metadata: dict[str, Any]


class RetrievalInterface(Protocol):
    """
    Interface expected from the retrieval component.

    Person 3's retrieval implementation can use its own internal result
    objects. ClassificationTools normalizes those results before passing
    them to the agents.
    """

    def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[Any]:
        """Search the approved corpus for relevant passages."""
        ...

    def lookup_clause(
        self,
        clause_id: str,
    ) -> Any | None:
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

    The reasoning layer has three explicit tools:
    1. corpus search
    2. clause lookup
    3. request-field lookup

    Retrieval-specific result objects are normalized here so the agents do
    not depend on Person 3's internal retrieval implementation.
    """

    def __init__(self, retrieval: RetrievalInterface) -> None:
        self._retrieval = retrieval

    def search_corpus(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedPassage]:
        """Search the corpus and normalize the returned evidence."""

        results = self._retrieval.search_corpus(
            query,
            top_k=top_k,
        )

        return [
            self._normalize_result(result)
            for result in results
        ]

    def lookup_clause(
        self,
        clause_id: str,
    ) -> RetrievedPassage | None:
        """Look up one clause and normalize it as retrieved evidence."""

        result = self._retrieval.lookup_clause(clause_id)

        if result is None:
            return None

        return self._normalize_result(result)

    def get_request_fields(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the product fields required by the agent."""

        return self._retrieval.get_request_fields(request)

    @staticmethod
    def _normalize_result(
        result: Any,
    ) -> RetrievedPassage:
        """
        Convert retrieval-layer results into the agent evidence format.

        Supports Person 3's current VectorSearchResult shape, where the
        DocumentChunk is available through ``result.chunk``, as well as a
        direct DocumentChunk-like object.
        """

        chunk = getattr(result, "chunk", result)

        passage_id = getattr(
            chunk,
            "chunk_id",
            "",
        )

        text = getattr(
            chunk,
            "text",
            "",
        )

        source = getattr(
            chunk,
            "source",
            "",
        )

        metadata = getattr(
            chunk,
            "metadata",
            {},
        )

        if not passage_id:
            raise ValueError(
                "Retrieved evidence is missing a passage identifier."
            )

        if not text:
            raise ValueError(
                f"Retrieved passage {passage_id} contains no text."
            )

        return RetrievedPassage(
            passage_id=str(passage_id),
            text=str(text),
            source=str(source),
            metadata=dict(metadata),
        )