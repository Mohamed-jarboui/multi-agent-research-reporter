# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (src/copilot/llm); package renamed.
"""Gemini provider (google-genai SDK)."""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from reporter.llm.base import (
    Embedding,
    EmbedTask,
    Generation,
    ProviderError,
    ServerError,
    StreamEvent,
    Tier,
    error_from_status,
)
from reporter.llm.limits import classify_gemini

# Overridable via the constructor. gemini-2.5-* answers 404 "no longer available to new users" on
# this key though the model list shows it; these two were tested with a real call on 2026-10-29.
DEFAULT_MODELS: dict[Tier, str] = {"small": "gemini-3.1-flash-lite", "large": "gemini-3.6-flash"}
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
_TASK_TYPES: dict[EmbedTask, str] = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


class GeminiProvider:
    name = "gemini"

    def available_models(self) -> list[str]:
        """Model names this key can use (a live call), to check the configured names exist."""
        return [m.name.removeprefix("models/") for m in self._client.models.list() if m.name]

    def __init__(
        self,
        api_key: str | None = None,
        *,
        models: dict[Tier, str] | None = None,
        embed_model: str = DEFAULT_EMBED_MODEL,
        timeout_s: int = 60,
        thinking_level: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not set")
        # An explicit timeout (ms): without one a stalled model call can hang a request for minutes.
        self._client = genai.Client(
            api_key=key, http_options=types.HttpOptions(timeout=timeout_s * 1000)
        )
        self._models = models or DEFAULT_MODELS
        self._embed_model = embed_model
        # "MINIMAL" skips most of the model's own reasoning: for answers extracted from supplied
        # sources it gave the same answer in 2.7 s where the default took 7-22 s (one test).
        self._thinking = (
            types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None
        )

    def _config(self, system, temperature, max_output_tokens) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=self._thinking,
        )

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
        config = self._config(system, temperature, max_output_tokens)
        try:
            response = self._client.models.generate_content(
                model=model, contents=prompt, config=config
            )
        except Exception as exc:
            raise _translate(exc) from exc
        meta = response.usage_metadata
        return Generation(
            text=response.text or "",
            model=model,
            provider=self.name,
            input_tokens=meta.prompt_token_count if meta else None,
            output_tokens=meta.candidates_token_count if meta else None,
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
        config = self._config(system, temperature, max_output_tokens)
        parts: list[str] = []
        meta = None
        try:
            for chunk in self._client.models.generate_content_stream(
                model=model, contents=prompt, config=config
            ):
                piece = chunk.text or ""
                if piece:
                    parts.append(piece)
                    yield StreamEvent(text=piece)
                if chunk.usage_metadata:
                    meta = chunk.usage_metadata  # the last chunk carries the totals
        except Exception as exc:
            raise _translate(exc) from exc
        yield StreamEvent(
            final=Generation(
                text="".join(parts),
                model=model,
                provider=self.name,
                input_tokens=meta.prompt_token_count if meta else None,
                output_tokens=meta.candidates_token_count if meta else None,
            )
        )

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding:
        config = types.EmbedContentConfig(
            task_type=_TASK_TYPES[task], output_dimensionality=dimensions
        )
        try:
            response = self._client.models.embed_content(
                model=self._embed_model, contents=texts, config=config
            )
        except Exception as exc:
            raise _translate(exc) from exc
        return Embedding(
            vectors=[list(e.values or []) for e in response.embeddings or []],
            model=self._embed_model,
        )


def _translate(exc: Exception) -> ProviderError:
    if isinstance(exc, genai_errors.APIError):
        limit, retry = (
            classify_gemini(getattr(exc, "details", None), str(exc))
            if exc.code == 429
            else (None, None)
        )
        return error_from_status(
            exc.code, str(exc), provider=GeminiProvider.name, limit=limit, retry_after_s=retry
        )
    if isinstance(exc, httpx.TransportError | TimeoutError | ConnectionError):
        return ServerError(str(exc), provider=GeminiProvider.name)
    raise exc
