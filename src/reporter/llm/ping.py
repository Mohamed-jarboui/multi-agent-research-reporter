"""Quota check: one tiny call to each provider on its own (no fallback, no retries).

Output is one line per provider with the failure class, status and limit type only: never a key
or a raw error message.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from reporter.llm.base import Provider, ProviderError, RateLimitError

PING_PROMPT = "Reply with: ok"
PING_MAX_TOKENS = 8
# provider name -> the environment variable that holds its key, in the order they are tried
PROVIDER_KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


@dataclass(frozen=True)
class PingResult:
    name: str
    status: str  # "ok", "failed" or "not configured"
    reason: str = ""  # class and HTTP status, e.g. "RateLimitError 429"
    limit: str | None = None
    retry_after_s: float | None = None
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def line(self) -> str:
        if self.status == "not configured":
            return f"{self.name}: not configured"
        if self.ok:
            return f"{self.name}: ok ({self.latency_s:.1f}s)"
        parts = [f"{self.name}: failed", self.reason]
        if self.limit:
            parts.append(f"limit={self.limit}")
        if self.retry_after_s is not None:
            parts.append(f"retry_after={self.retry_after_s:g}s")
        parts.append(f"({self.latency_s:.1f}s)")
        return " ".join(parts)


def ping_provider(
    name: str, provider: Provider | None, clock: Callable[[], float] = time.perf_counter
) -> PingResult:
    """One call, no retry. `provider` None means no key is configured."""
    if provider is None:
        return PingResult(name, "not configured")
    start = clock()
    try:
        provider.generate(
            PING_PROMPT, tier="small", temperature=0.0, max_output_tokens=PING_MAX_TOKENS
        )
    except ProviderError as exc:
        return PingResult(
            name,
            "failed",
            exc.short(),
            exc.limit if isinstance(exc, RateLimitError) else None,
            exc.retry_after_s if isinstance(exc, RateLimitError) else None,
            clock() - start,
        )
    except Exception as exc:  # an unexpected SDK error: the class name only, never its text
        return PingResult(name, "failed", type(exc).__name__, latency_s=clock() - start)
    return PingResult(name, "ok", latency_s=clock() - start)


def ping_all(
    providers: Mapping[str, Provider | None], clock: Callable[[], float] = time.perf_counter
) -> list[PingResult]:
    return [ping_provider(name, provider, clock) for name, provider in providers.items()]


def build_providers() -> dict[str, Provider | None]:
    """Each provider on its own; None when its key is not set."""
    found: dict[str, Provider | None] = {}
    for name, variable in PROVIDER_KEYS.items():
        if not os.environ.get(variable):
            found[name] = None
        elif name == "gemini":
            from reporter.llm.gemini import GeminiProvider

            found[name] = GeminiProvider()
        else:
            from reporter.llm.groq import GroqProvider

            found[name] = GroqProvider()
    return found
