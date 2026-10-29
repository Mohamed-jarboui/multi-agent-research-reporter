# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (src/copilot/llm); package renamed.
"""Build the default Gemini -> Groq chain from environment variables."""

from __future__ import annotations

import os

from reporter.llm.router import FallbackLLM


def build_default_llm(**router_kwargs) -> FallbackLLM:
    """Gemini first, Groq second; a provider is included only if its key is set."""
    providers = []
    if os.environ.get("GEMINI_API_KEY"):
        from reporter.llm.gemini import GeminiProvider

        providers.append(GeminiProvider())
    if os.environ.get("GROQ_API_KEY"):
        from reporter.llm.groq import GroqProvider

        providers.append(GroqProvider())
    if not providers:
        raise ValueError("Set GEMINI_API_KEY and/or GROQ_API_KEY (see .env.example)")
    return FallbackLLM(providers, **router_kwargs)
