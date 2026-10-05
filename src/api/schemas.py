"""
API schemas for the Export Classification Check Service.

These models define the structured contract between API clients and the
classification workflow.
"""

from typing import Any

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    """Product information submitted for classification."""

    product_description: str = Field(
        ...,
        min_length=1,
        description="Description of the product to classify.",
    )
    materials: list[str] = Field(
        default_factory=list,
        description="Known materials used in the product.",
    )
    country_of_origin: str | None = Field(
        default=None,
        description="Country of origin when provided.",
    )
    additional_information: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional product information supplied by the requester.",
    )


class ClassificationResponse(BaseModel):
    """Response returned by the classification workflow."""

    correlation_id: str
    status: str
    proposed_classification: str | None = None
    reasoning: str | None = None
    citations: list[str] = Field(default_factory=list)
    unresolved_issues: list[str] = Field(default_factory=list)
    message: str | None = None