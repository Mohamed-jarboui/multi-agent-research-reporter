# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (src/copilot/llm); package renamed.
"""Provider layer: one interface, Gemini and Groq backends, fallback and usage records."""

from reporter.llm.base import (
    AllProvidersFailedError,
    AuthError,
    BadRequestError,
    Embedding,
    Generation,
    NotFoundError,
    Provider,
    ProviderError,
    RateLimitError,
    ServerError,
    UnsupportedOperationError,
    Usage,
)
from reporter.llm.factory import build_default_llm
from reporter.llm.router import FallbackLLM

__all__ = [
    "AllProvidersFailedError",
    "AuthError",
    "BadRequestError",
    "Embedding",
    "FallbackLLM",
    "Generation",
    "NotFoundError",
    "Provider",
    "ProviderError",
    "RateLimitError",
    "ServerError",
    "UnsupportedOperationError",
    "Usage",
    "build_default_llm",
]
