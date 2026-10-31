"""Per-IP token buckets, the client address, and the provider-quota check."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

DEFAULT_LATCH_S = 3600.0  # when a provider gives no retry delay: look again in an hour
MAX_TRACKED_IPS = 10_000


class RateLimiter:
    """A token bucket per client, kept in an LRU of at most `max_clients` entries.

    `burst` requests may arrive at once; one more is allowed every `refill_s` seconds.
    """

    def __init__(
        self,
        burst: int,
        refill_s: float,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = MAX_TRACKED_IPS,
    ) -> None:
        self.burst = float(burst)
        self.refill_s = refill_s
        self._clock = clock
        self._max = max_clients
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # ip -> (tokens, at)
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        now = self._clock()
        with self._lock:
            tokens, at = self._buckets.pop(client, (self.burst, now))
            tokens = min(self.burst, tokens + (now - at) / self.refill_s)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[client] = (tokens, now)
            while len(self._buckets) > self._max:
                self._buckets.popitem(last=False)
            return allowed

    def retry_after_s(self, client: str) -> int:
        """Seconds until this client has a token again (0 when it has one)."""
        now = self._clock()
        with self._lock:
            tokens, at = self._buckets.get(client, (self.burst, now))
        tokens = min(self.burst, tokens + (now - at) / self.refill_s)
        return 0 if tokens >= 1.0 else int((1.0 - tokens) * self.refill_s) + 1


def client_ip(headers: Mapping[str, str], peer: str | None, hops: int) -> str:
    """The client address: the `hops`-th entry from the right of X-Forwarded-For (what our own
    proxy appended; anything further left is client-controlled), else the socket peer."""
    forwarded = headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if hops > 0 and len(parts) >= hops:
        return parts[-hops]
    return peer or "unknown"


def provider_quota_exhausted(summary: Mapping[str, Any]) -> bool:
    """True when every configured provider answered with a per-day or exhausted-quota 429 (a
    per-minute limit clears by itself, so it never counts). `summary` is the router's."""
    providers = list(summary.get("providers", []))
    limits = summary.get("limits", {})
    if not providers:
        return False
    return all(
        any(kind.startswith("per_day") or kind == "quota_exhausted" for kind in limits.get(p, {}))
        for p in providers
    )


def retry_delay_s(failures: list[dict[str, Any]]) -> float:
    """The longest retry delay a provider gave for a per-day limit, else one hour."""
    delays = [
        float(f["retry_after_s"])
        for f in failures
        if "retry_after_s" in f
        and (str(f.get("limit", "")).startswith("per_day") or f.get("limit") == "quota_exhausted")
    ]
    return max(delays) if delays else DEFAULT_LATCH_S
