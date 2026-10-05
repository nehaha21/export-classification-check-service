from src.agents.classification_agent import ClassificationAgent
from src.agents.graph import ClassificationGraph
from src.agents.tools import ClassificationTools, RetrievedPassage
from src.agents.verification_agent import VerificationAgent


class FakeRetrieval:
    def search_corpus(self, query: str, *, top_k: int = 5):
        return [
            RetrievedPassage(
                passage_id="test-001",
                text="Vacuum flasks are classified under the relevant tariff heading.",
                source="test-corpus",
                metadata={},
            )
        ]

    def lookup_clause(self, clause_id: str):
        return None

    def get_request_fields(self, request):
        return {
            "product_description": request["product_description"],
            "article": request["product_description"],
        }


class FakeClassificationModel:
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "proposed classification"


class FakeVerificationModel:
    def __init__(self, response: str):
        self.response = response

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response


def make_classification_agent():
    tools = ClassificationTools(FakeRetrieval())

    return ClassificationAgent(
        tools=tools,
        model=FakeClassificationModel(),
    )


def make_graph(verification_response: str, max_attempts: int = 2):
    classification_agent = make_classification_agent()

    verification_agent = VerificationAgent(
        model=FakeVerificationModel(verification_response),
    )

    return ClassificationGraph(
        classification_agent=classification_agent,
        verification_agent=verification_agent,
        max_attempts=max_attempts,
    )


def test_graph_returns_verified_result():
    graph = make_graph(
        "verified: true\n"
        "reason: The proposal is supported by the retrieved passage."
    )

    result = graph.run(
        {
            "product_description": (
                "Stainless steel vacuum flask with a moulded plastic outer body."
            )
        }
    )

    assert result.status == "clear-for-filing"
    assert result.proposed_answer == "proposed classification"
    assert result.attempts == 1


def test_graph_escalates_when_verification_fails():
    graph = make_graph(
        "verified: false\n"
        "reason: The proposal is not supported by the retrieved evidence.",
        max_attempts=2,
    )

    result = graph.run(
        {
            "product_description": (
                "Stainless steel vacuum flask with a moulded plastic outer body."
            )
        }
    )

    assert result.status == "specialist-classification-review"
    assert result.proposed_answer is None
    assert result.attempts == 2
    assert result.escalation_reason is not None


def test_graph_escalates_when_no_passages_are_found():
    class EmptyRetrieval(FakeRetrieval):
        def search_corpus(self, query: str, *, top_k: int = 5):
            return []

    tools = ClassificationTools(EmptyRetrieval())

    classification_agent = ClassificationAgent(
        tools=tools,
        model=FakeClassificationModel(),
    )

    verification_agent = VerificationAgent(
        model=FakeVerificationModel("verified: true"),
    )

    graph = ClassificationGraph(
        classification_agent=classification_agent,
        verification_agent=verification_agent,
    )

    result = graph.run(
        {
            "product_description": "product outside the supported corpus",
        }
    )

    assert result.status == "specialist-classification-review"
    assert result.proposed_answer is None
    assert result.escalation_reason is not None


def test_graph_requests_clarification_when_description_is_missing():
    graph = make_graph("verified: true")

    result = graph.run({})

    assert result.status == "seek-product-clarification"
    assert result.proposed_answer is None