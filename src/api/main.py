"""
FastAPI application entry point.
"""

from fastapi import FastAPI

from .routes import router


app = FastAPI(
    title="Export Classification Check Service",
    version="0.1.0",
    description=(
        "API for checking proposed export classifications against "
        "retrieved tariff evidence."
    ),
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return basic service health."""

    return {"status": "ok"}