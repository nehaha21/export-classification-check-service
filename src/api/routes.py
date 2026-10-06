"""
API routes for the Export Classification Check Service.
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from src.agents.classification_agent import ClassificationAgent
from src.agents.graph import ClassificationGraph
from src.agents.tools import ClassificationTools, RetrievedPassage
from src.agents.verification_agent import VerificationAgent

from .middleware import RateLimiter, ResponseCache
from .schemas import ClassificationRequest, ClassificationResponse


router = APIRouter()

response_cache = ResponseCache(ttl_seconds=60.0)
rate_limiter = RateLimiter(max_requests=10, window_seconds=60.0)


class LoadTestRetrieval:
    """Deterministic retrieval stub used only for load testing."""

    def search_corpus(self, query: str, *, top_k: int = 5):
        return [
            RetrievedPassage(
                passage_id="load-test-001",
                text="Synthetic evidence used only for load testing.",
                source="load-test-stub",
                metadata={"load_test": True},
            )
        ]

    def lookup_clause(self, clause_id: str):
        return None

    def get_request_fields(self, request):
        return {
            "product_description": request["product_description"],
            "article": request["product_description"],
        }


class LoadTestClassificationModel:
    """Deterministic LLM stub used only for load testing."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "proposed_classification: load-test-stub\n"
            "rule_applied: load-test-GRI\n"
            "reasoning: Synthetic response used only for load testing.\n"
            "citations: load-test-001"
        )


class LoadTestVerificationModel:
    """Deterministic verification-model stub used only for load testing."""

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return (
            "verified: true\n"
            "reason: Synthetic verification used only for load testing."
        )


def build_load_test_graph() -> ClassificationGraph:
    tools = ClassificationTools(LoadTestRetrieval())

    classification_agent = ClassificationAgent(
        tools=tools,
        model=LoadTestClassificationModel(),
    )

    verification_agent = VerificationAgent(
        model=LoadTestVerificationModel(),
    )

    return ClassificationGraph(
        classification_agent=classification_agent,
        verification_agent=verification_agent,
    )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
)
async def classify(
    request: ClassificationRequest,
    http_request: Request,
) -> JSONResponse:
    """Submit a product for classification."""

    client_id = (
        http_request.client.host
        if http_request.client
        else "unknown"
    )

    if not rate_limiter.allow(client_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )

    cache_key = str(request.model_dump())
    cached_response = response_cache.get(cache_key)

    if cached_response is not None:
        return JSONResponse(
            content=cached_response.model_dump(),
            headers={"X-Cache": "HIT"},
        )

    if request.additional_information.get("llm_stub") is True:
        graph = build_load_test_graph()

        result = graph.run(
            {
                "product_description": request.product_description,
                "materials": request.materials,
            }
        )

        response = ClassificationResponse(
            correlation_id="load-test",
            status=result.status,
            proposed_classification=result.proposed_answer,
            message="LLM stub load-test workflow.",
        )

        response_cache.set(cache_key, response)

        return JSONResponse(
            content=response.model_dump(),
            headers={"X-Cache": "MISS"},
        )

    response = ClassificationResponse(
        correlation_id="pending",
        status="specialist-classification-review",
        message="Classification workflow is not connected yet.",
    )

    response_cache.set(cache_key, response)

    return JSONResponse(
        content=response.model_dump(),
        headers={"X-Cache": "MISS"},
    )