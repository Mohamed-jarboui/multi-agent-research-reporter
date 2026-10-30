# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (src/copilot/llm); package renamed.
"""Groq provider (groq SDK). Groq has no embeddings endpoint, so embed() is unsupported."""

from __future__ import annotations

import os
from collections.abc import Iterator

import groq as groq_sdk

from reporter.llm.base import (
    Embedding,
    EmbedTask,
    Generation,
    ProviderError,
    ServerError,
    StreamEvent,
    Tier,
    UnsupportedOperationError,
    error_from_status,
)

# Overridable via the constructor. Checked against `available_models()` on 2026-10-29: the
# llama-3.x names used before were not offered by this key, these two chat models were.
DEFAULT_MODELS: dict[Tier, str] = {
    "small": "openai/gpt-oss-20b",
    "large": "openai/gpt-oss-120b",
}


class GroqProvider:
    name = "groq"

    def available_models(self) -> list[str]:
        """Model ids this key can use (a live call), to check the configured names exist."""
        return [m.id for m in self._client.models.list().data]

    def __init__(
        self,
        api_key: str | None = None,
        *,
        models: dict[Tier, str] | None = None,
        reasoning_effort: str | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set")
        # retries are the router's job; the timeout stops a stalled call from hanging a request
        self._client = groq_sdk.Groq(api_key=key, max_retries=0, timeout=timeout_s)
        self._models = models or DEFAULT_MODELS
        # Reasoning models (gpt-oss) spend output tokens thinking; "low" keeps that small.
        self._extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation:
        model = self._models[tier]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                **self._extra,
            )
        except Exception as exc:
            raise _translate(exc) from exc
        usage = response.usage
        return Generation(
            text=response.choices[0].message.content or "",
            model=model,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        model = self._models[tier]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        parts: list[str] = []
        usage = None
        try:
            stream = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                stream=True,
                **self._extra,
            )
            for chunk in stream:
                piece = chunk.choices[0].delta.content if chunk.choices else None
                if piece:
                    parts.append(piece)
                    yield StreamEvent(text=piece)
                # Groq reports usage on the last chunk, either as `usage` or under `x_groq`
                usage = (
                    getattr(chunk, "usage", None)
                    or getattr(getattr(chunk, "x_groq", None), "usage", None)
                    or usage
                )
        except Exception as exc:
            raise _translate(exc) from exc
        yield StreamEvent(
            final=Generation(
                text="".join(parts),
                model=model,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
            )
        )

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding:
        raise UnsupportedOperationError("Groq has no embeddings endpoint", provider=self.name)


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, groq_sdk.APIStatusError):
        return error_from_status(exc.status_code, str(exc), provider=GroqProvider.name)
    if isinstance(exc, groq_sdk.APIConnectionError):  # includes APITimeoutError
        return ServerError(str(exc), provider=GroqProvider.name)
    raise exc
