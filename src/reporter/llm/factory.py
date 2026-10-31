"""Build the default Gemini -> Groq chain from environment variables.

The third-party gateway ("agentrouter") joins the chain, last, only by explicit opt-in.
"""

from __future__ import annotations

import logging
import os

from reporter.llm.router import FallbackLLM

log = logging.getLogger(__name__)
EXTRA_PROVIDERS_VARIABLE = "REPORTER_EXTRA_PROVIDERS"


def extra_providers() -> set[str]:
    """Names listed in REPORTER_EXTRA_PROVIDERS (comma list); empty by default."""
    raw = os.environ.get(EXTRA_PROVIDERS_VARIABLE, "")
    return {part.strip().lower() for part in raw.split(",") if part.strip()}


def gateway_enabled() -> bool:
    """The gateway is in the chain only when it is listed AND its key and model are set."""
    from reporter.llm.openai_compat import KEY_VARIABLE, MODEL_VARIABLE

    if "agentrouter" not in extra_providers() or not os.environ.get(KEY_VARIABLE):
        return False
    if not os.environ.get(MODEL_VARIABLE):
        log.warning("agentrouter is requested but %s is not set; leaving it out", MODEL_VARIABLE)
        return False
    return True


def build_default_llm(**router_kwargs) -> FallbackLLM:
    """Gemini first, Groq second; a provider is included only if its key is set."""
    providers = []
    if os.environ.get("GEMINI_API_KEY"):
        from reporter.llm.gemini import GeminiProvider

        providers.append(GeminiProvider())
    if os.environ.get("GROQ_API_KEY"):
        from reporter.llm.groq import GroqProvider

        providers.append(GroqProvider())
    if gateway_enabled():  # last, and only by opt-in: it is a third-party proxy
        from reporter.llm.openai_compat import OpenAICompatProvider

        providers.append(OpenAICompatProvider())
    if not providers:
        raise ValueError("Set GEMINI_API_KEY and/or GROQ_API_KEY (see .env.example)")
    return FallbackLLM(providers, **router_kwargs)
