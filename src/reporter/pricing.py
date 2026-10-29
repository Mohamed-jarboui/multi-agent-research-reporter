"""Cost of a model call in US dollars.

Prices are USD per million tokens as (input, output). Both providers are used on their free tier,
which bills nothing, so the default models are priced at 0.0. A model that is not in the table
costs `None` (unknown): the trace counts those calls instead of inventing a price. Edit
PRICES_PER_MTOK when you move to a paid tier so the cost ceiling means something.
"""

from __future__ import annotations

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.1-flash-lite": (0.0, 0.0),  # free tier
    "gemini-3.6-flash": (0.0, 0.0),  # free tier
    "openai/gpt-oss-20b": (0.0, 0.0),  # Groq free tier
    "openai/gpt-oss-120b": (0.0, 0.0),  # Groq free tier
}


def call_cost_usd(model: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """Dollar cost of one call, or None when the model has no price or tokens were not reported."""
    price = PRICES_PER_MTOK.get(model)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
