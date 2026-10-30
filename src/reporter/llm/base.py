# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (src/copilot/llm); package renamed.
"""Provider interface, result types and the error taxonomy shared by all LLM providers.

Providers translate their SDK's exceptions into the errors below, so the router never has to
import an SDK and can be tested with plain fakes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

Tier = Literal["small", "large"]
"""Model tier: "small" for routing/grading, "large" for synthesis."""

EmbedTask = Literal["document", "query"]
"""What an embedding is for: asymmetric models embed a passage and a query differently."""


class ProviderError(Exception):
    """Base class for every error a provider is allowed to raise."""

    def __init__(self, message: str, *, provider: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status = status  # HTTP status when there is one

    def short(self) -> str:
        """Class and status only: safe for traces and reports (no message, ids or URLs)."""
        return type(self).__name__ + (f" {self.status}" if self.status else "")


class RateLimitError(ProviderError):
    """HTTP 429: too many requests or quota exhausted."""


class ServerError(ProviderError):
    """HTTP 5xx, timeouts and connection failures."""


class EmptyReplyError(ServerError):
    """The call succeeded but the reply had no text (a reasoning model used all its tokens)."""


class AuthError(ProviderError):
    """HTTP 401/403: missing, invalid or unauthorised key."""


class NotFoundError(ProviderError):
    """HTTP 404: the model or endpoint does not exist for this key (e.g. a retired model)."""


class BadRequestError(ProviderError):
    """Other 4xx: the request itself is wrong, so resending it anywhere will not help."""


class UnsupportedOperationError(ProviderError):
    """The provider does not offer this operation (e.g. Groq has no embeddings endpoint)."""


class AllProvidersFailedError(ProviderError):
    """Every provider was tried (or skipped) and none succeeded."""


def error_from_status(status: int | None, message: str, *, provider: str) -> ProviderError:
    """Map an HTTP status code to the matching ProviderError."""
    if status == 429:
        error: ProviderError = RateLimitError(message, provider=provider)
    elif status in (401, 403):
        error = AuthError(message, provider=provider)
    elif status == 404:
        error = NotFoundError(message, provider=provider)
    elif status is not None and 400 <= status < 500:
        error = BadRequestError(message, provider=provider)
    else:
        error = ServerError(message, provider=provider)
    error.status = status
    return error


@dataclass(frozen=True)
class Generation:
    """What a provider returns from generate()."""

    text: str
    model: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class StreamEvent:
    """One step of a streamed generation: a text delta, then a last event carrying the totals."""

    text: str = ""
    final: Generation | None = None  # set only on the last event (token counts may be None)


@dataclass(frozen=True)
class Embedding:
    """What a provider returns from embed(): one vector per input text, in order."""

    vectors: list[list[float]]
    model: str
    input_tokens: int | None = None


@dataclass(frozen=True)
class Usage:
    """One record per provider call, successful or not.

    Token counts are None when the provider does not report them (never guessed).
    """

    provider: str
    model: str
    operation: Literal["generate", "embed"]
    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    error: str | None = None


class Provider(Protocol):
    """Minimal interface every LLM backend implements."""

    name: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation: ...

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]: ...

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding: ...
