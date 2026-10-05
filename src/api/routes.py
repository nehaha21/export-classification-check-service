"""
API routes for the Export Classification Check Service.
"""

from fastapi import APIRouter

from .schemas import ClassificationRequest, ClassificationResponse


router = APIRouter()


@router.post(
    "/classify",
    response_model=ClassificationResponse,
)
async def classify(
    request: ClassificationRequest,
) -> ClassificationResponse:
    """
    Submit a product for classification.

    The workflow integration will be connected to this route when the
    application orchestration layer is wired together.
    """

    return ClassificationResponse(
        correlation_id="pending",
        status="specialist-classification-review",
        message="Classification workflow is not connected yet.",
    )