"""Parse model output: pull one JSON object out of text that may have fences or chatter."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any]:
    """Return the first JSON object in `text`. Raises ValueError when there is none."""
    cleaned = _FENCE.sub("", text.strip())
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            start = cleaned.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = cleaned.find("{", start + 1)
    raise ValueError("no JSON object found in the reply")


_SOURCES_HEADING = re.compile(r"\n#{1,6}\s*(sources|references)\b.*\Z", re.IGNORECASE | re.DOTALL)
_CITATION = re.compile(r"\[(\d+)\]")


def strip_sources_section(markdown: str) -> str:
    """Drop a trailing Sources/References section the model wrote itself (we append our own)."""
    return _SOURCES_HEADING.sub("", "\n" + markdown).strip()


def cited_numbers(markdown: str) -> list[int]:
    """Every citation number used in the text, in order of first use."""
    seen: list[int] = []
    for match in _CITATION.finditer(markdown):
        n = int(match.group(1))
        if n not in seen:
            seen.append(n)
    return seen
