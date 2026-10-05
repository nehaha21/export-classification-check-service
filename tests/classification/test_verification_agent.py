from src.agents.tools import RetrievedPassage
from src.agents.verification_agent import VerificationAgent


class FakeModel:
    def __init__(self, response: str):
        self.response = response

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.response


def make_passage():
    return RetrievedPassage(
        passage_id="test-001",
        text="Vacuum flasks are classified under the relevant tariff heading.",
        source="test-corpus",
        metadata={},
    )


def test_verification_agent_accepts_verified_answer():
    agent = VerificationAgent(
        model=FakeModel(
            "verified: true\n"
            "supported_claims: vacuum flask heading\n"
            "unsupported_claims: none\n"
            "invalid_citations: none\n"
            "reason: The retrieved passage supports the proposal."
        )
    )

    result = agent.verify(
        proposed_answer="The product is a vacuum flask.",
        passages=[make_passage()],
    )

    assert "verified: true" in result


def test_verification_agent_rejects_unsupported_answer():
    agent = VerificationAgent(
        model=FakeModel(
            "verified: false\n"
            "supported_claims: none\n"
            "unsupported_claims: proposed classification\n"
            "invalid_citations: test-999\n"
            "reason: The supplied passages do not support the proposal."
        )
    )

    result = agent.verify(
        proposed_answer="The product belongs to an unsupported heading.",
        passages=[make_passage()],
    )

    assert "verified: false" in result


def test_verification_agent_receives_retrieved_passage():
    captured = {}

    class CapturingModel:
        def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            captured["prompt"] = user_prompt
            return "verified: false"

    agent = VerificationAgent(model=CapturingModel())

    agent.verify(
        proposed_answer="Test proposal",
        passages=[make_passage()],
    )

    assert "test-001" in captured["prompt"]
    assert "Vacuum flasks" in captured["prompt"]