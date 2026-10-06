"""
Registry of versioned prompts used by the classification service.
"""

from dataclasses import dataclass

from .classification_prompt import (
    CLASSIFICATION_SYSTEM_PROMPT,
    PROMPT_VERSION as CLASSIFICATION_PROMPT_VERSION,
)
from .verification_prompt import (
    VERIFICATION_SYSTEM_PROMPT,
    PROMPT_VERSION as VERIFICATION_PROMPT_VERSION,
)


@dataclass(frozen=True)
class PromptVersion:
    """A versioned prompt artifact."""

    name: str
    version: str
    template: str


CLASSIFICATION_PROMPT = PromptVersion(
    name="classification",
    version=CLASSIFICATION_PROMPT_VERSION,
    template=CLASSIFICATION_SYSTEM_PROMPT,
)

VERIFICATION_PROMPT = PromptVersion(
    name="verification",
    version=VERIFICATION_PROMPT_VERSION,
    template=VERIFICATION_SYSTEM_PROMPT,
)


PROMPT_REGISTRY = {
    CLASSIFICATION_PROMPT.version: CLASSIFICATION_PROMPT,
    VERIFICATION_PROMPT.version: VERIFICATION_PROMPT,
}


def get_prompt(version: str) -> PromptVersion:
    """Return a registered prompt by version."""

    if version not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt version: {version}")

    return PROMPT_REGISTRY[version]