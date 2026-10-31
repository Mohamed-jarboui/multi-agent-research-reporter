"""Which 429 was it? Classify a rate-limit error into a fixed set of limit types.

Only the enum value and an optional retry delay leave this module: never the raw message,
organisation or project ids, or URLs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, get_args

LimitType = Literal[
    "per_minute_tokens",
    "per_minute_requests",
    "per_day_tokens",
    "per_day_requests",
    "quota_exhausted",
    "unknown",
]
LIMIT_TYPES: tuple[str, ...] = get_args(LimitType)

_GROQ_CODES: dict[str, LimitType] = {
    "TPM": "per_minute_tokens",
    "RPM": "per_minute_requests",
    "TPD": "per_day_tokens",
    "RPD": "per_day_requests",
}
_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)", re.IGNORECASE)
_UNIT_S = {"d": 86400.0, "h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
_MAX_RETRY_S = 7 * 86400.0


def parse_duration_s(text: str) -> float | None:
    """'6.5s', '1m30.5s', '12h3m', '34s' -> seconds (None when nothing matches)."""
    parts = _DURATION.findall(text.replace(" ", ""))
    if not parts:
        return None
    total = sum(float(n) * _UNIT_S[u.lower()] for n, u in parts)
    return round(total, 2) if 0 <= total <= _MAX_RETRY_S else None


def _limit_from_words(text: str) -> LimitType | None:
    """'per day' / 'perday' and 'token' / 'request' anywhere in `text` -> the limit type."""
    lowered = re.sub(r"[\s_]+", "", text.lower())
    period = "day" if "perday" in lowered else "minute" if "perminute" in lowered else ""
    kind = "tokens" if "token" in lowered else "requests" if "request" in lowered else ""
    if period and kind:
        return f"per_{period}_{kind}"  # type: ignore[return-value]
    return None


def classify_groq(message: str) -> tuple[LimitType, float | None]:
    """Groq names the limit in its message, e.g. 'on tokens per minute (TPM): Limit 8000 ...
    Please try again in 6.5s'."""
    code = re.search(r"\((TPM|RPM|TPD|RPD)\)", message)
    limit: LimitType = (
        _GROQ_CODES[code.group(1)] if code else _limit_from_words(message) or "unknown"
    )
    retry = re.search(r"try again in\s+([0-9a-z.]+)", message, re.IGNORECASE)
    return limit, parse_duration_s(retry.group(1)) if retry else None


def classify_gemini(details: Any, message: str = "") -> tuple[LimitType, float | None]:
    """Gemini's error body carries `error.details`: a QuotaFailure with violations
    (`quotaMetric`, `quotaId`, e.g. 'GenerateRequestsPerDayPerProjectPerModel-FreeTier') and a
    RetryInfo with `retryDelay` ('34s'). Without a period in them the quota is reported as
    'quota_exhausted' (it names a quota) or 'unknown'."""
    body = details.get("error", details) if isinstance(details, Mapping) else {}
    raw = body.get("details") if isinstance(body, Mapping) else None
    items = [d for d in raw if isinstance(d, Mapping)] if isinstance(raw, list) else []
    retry: float | None = None
    ids: list[str] = []
    for item in items:
        delay = item.get("retryDelay")
        if isinstance(delay, str):
            retry = parse_duration_s(delay) or retry
        for violation in item.get("violations") or []:
            if isinstance(violation, Mapping):
                ids += [str(violation.get("quotaId", "")), str(violation.get("quotaMetric", ""))]
    limit = _limit_from_words(" ".join(ids))
    if limit:
        return limit, retry
    if ids or "quota" in message.lower():
        return "quota_exhausted", retry
    return "unknown", retry
