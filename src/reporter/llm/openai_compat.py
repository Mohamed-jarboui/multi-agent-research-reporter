"""OpenAI-compatible chat-completions provider, used for the opt-in "agentrouter" gateway.

A third-party proxy: it is never part of the default chain (see factory.py). Errors carry a fixed
plain message plus the status; the key, the request body and the response body never reach an
exception, a log line or a trace.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import httpx

from reporter.llm.base import (
    AuthError,
    Embedding,
    EmbedTask,
    Generation,
    NotFoundError,
    ProviderError,
    ServerError,
    StreamEvent,
    Tier,
    UnsupportedOperationError,
    error_from_status,
)
from reporter.llm.limits import classify_groq, parse_duration_s

DEFAULT_BASE_URL = "https://agentrouter.org/v1"
KEY_VARIABLE = "AGENTROUTER_API_KEY"
MODEL_VARIABLE = "AGENTROUTER_MODEL"
BASE_URL_VARIABLE = "AGENTROUTER_BASE_URL"

_MODEL_HINTS = ("no available channel", "model not found", "model_not_found", "does not exist")


class ProviderConfigError(ProviderError):
    """A required setting is missing. The message names the variable, never a value."""


class ModelUnavailableError(NotFoundError):
    """The gateway cannot serve the configured model (404, or 503 'no available channel')."""


class OpenAICompatProvider:
    name = "agentrouter"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        key = api_key or os.environ.get(KEY_VARIABLE)
        if not key:
            raise ProviderConfigError(f"{KEY_VARIABLE} is not set", provider=self.name)
        chosen = model or os.environ.get(MODEL_VARIABLE)
        if not chosen:
            raise ProviderConfigError(
                f"{MODEL_VARIABLE} is not set (it has no default; name a model the gateway serves)",
                provider=self.name,
            )
        url = base_url or os.environ.get(BASE_URL_VARIABLE) or DEFAULT_BASE_URL
        self._url = url.rstrip("/") + "/chat/completions"
        self._key = key
        self._model = chosen
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            body["max_tokens"] = max_output_tokens
        try:
            response = self._client.post(
                self._url, json=body, headers={"Authorization": f"Bearer {self._key}"}
            )
        except httpx.HTTPError as exc:  # timeouts and connection failures: class name only
            raise ServerError(
                f"gateway unreachable ({type(exc).__name__})", provider=self.name
            ) from None
        if response.status_code >= 400:
            raise self._error(response)
        return self._parse(response)

    def _parse(self, response: httpx.Response) -> Generation:
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            tokens_in, tokens_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ServerError(
                "gateway reply was not a chat completion", provider=self.name, status=200
            ) from None
        return Generation(
            text=text if isinstance(text, str) else "",
            model=self._model,
            provider=self.name,
            input_tokens=tokens_in if isinstance(tokens_in, int) else None,
            output_tokens=tokens_out if isinstance(tokens_out, int) else None,
        )

    def _error(self, response: httpx.Response) -> ProviderError:
        status = response.status_code
        message = _error_text(response)  # classification only; never stored or re-raised
        if status == 429:
            limit, retry = classify_groq(message)
            header = response.headers.get("retry-after", "")
            try:
                retry = float(header)
            except ValueError:
                retry = retry or (parse_duration_s(header) if header else None)
            return error_from_status(
                429,
                "rate limited by gateway",
                provider=self.name,
                limit=limit,
                retry_after_s=retry,
            )
        if status in (401, 403):
            error: ProviderError = AuthError(
                f"gateway rejected the key ({status}); check {KEY_VARIABLE}", provider=self.name
            )
        elif status == 404 or (status == 503 and any(h in message.lower() for h in _MODEL_HINTS)):
            error = ModelUnavailableError(
                f"model not available on gateway; check {MODEL_VARIABLE}", provider=self.name
            )
        else:
            return error_from_status(status, f"gateway error ({status})", provider=self.name)
        error.status = status
        return error

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Iterator[StreamEvent]:
        """Not streamed on the wire: one call, replayed as a single delta plus the final event."""
        gen = self.generate(
            prompt,
            system=system,
            tier=tier,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if gen.text:
            yield StreamEvent(text=gen.text)
        yield StreamEvent(final=gen)

    def embed(
        self, texts: list[str], *, task: EmbedTask = "document", dimensions: int | None = None
    ) -> Embedding:
        raise UnsupportedOperationError("gateway provider has no embeddings", provider=self.name)


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        err = data.get("error", data)
        text = err.get("message") if isinstance(err, dict) else err
        return text if isinstance(text, str) else ""
    except (ValueError, AttributeError):
        return ""
