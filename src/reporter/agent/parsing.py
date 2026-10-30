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
_CITATION_GROUP = re.compile(r"\[\s*(\d+(?:\s*[,;\-–—]\s*\d+)*)\s*\]")
_MAX_RANGE = 100


def citation_numbers(markdown: str) -> list[int]:
    """Every citation number in the text, in order of first use.

    Understands [5], [5, 6], [5][6], [5-7] and [5,7-9]. A range that runs backwards or is longer
    than 100 is not expanded: its two ends are kept (so the checker can flag them).
    """
    seen: list[int] = []
    for match in _CITATION_GROUP.finditer(markdown):
        for part in re.split(r"\s*[,;]\s*", match.group(1)):
            ends = [int(n) for n in re.findall(r"\d+", part)]
            if len(ends) == 2 and 0 <= ends[1] - ends[0] <= _MAX_RANGE:
                numbers = list(range(ends[0], ends[1] + 1))
            else:
                numbers = ends
            seen += [n for n in numbers if n not in seen]
    return seen


def strip_sources_section(markdown: str) -> str:
    """Drop a trailing Sources/References section the model wrote itself (we append our own)."""
    return _SOURCES_HEADING.sub("", "\n" + markdown).strip()


def cited_numbers(markdown: str) -> list[int]:
    """Alias of `citation_numbers`, kept for the R0 baseline."""
    return citation_numbers(markdown)
