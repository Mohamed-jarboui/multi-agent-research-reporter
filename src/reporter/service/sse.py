"""Server-sent events from the persisted events of a job.

The stream reads SQLite (poll every second), so it gives the full history to a late subscriber,
resumes from `Last-Event-ID`, and works the same after a restart. A client that disconnects only
ends this generator; the job keeps running.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from reporter.service.store import TERMINAL, TERMINAL_EVENTS, JobStore

RETRY_MS = 3000


def frame(event: dict[str, Any]) -> str:
    return (
        f"id: {event['seq']}\nevent: {event['type']}\n"
        f"data: {json.dumps(event['data'], separators=(',', ':'))}\n\n"
    )


async def event_stream(
    store: JobStore,
    job_id: str,
    after: int,
    *,
    poll_s: float,
    heartbeat_s: float,
    sleep: Callable[[float], Awaitable[None]],
    monotonic: Callable[[], float],
) -> AsyncIterator[str]:
    """Frames with `seq > after`, then new ones as they appear; ends after a terminal event."""
    yield f"retry: {RETRY_MS}\n\n"
    last = after
    last_sent = monotonic()
    while True:
        job = store.get_job(job_id)  # read first: its terminal events are then already stored
        events = store.events_after(job_id, last)
        for event in events:
            last = event["seq"]
            yield frame(event)
            last_sent = monotonic()
            if event["type"] in TERMINAL_EVENTS:
                return
        if events:
            continue
        if job is None or job["state"] in TERMINAL:  # nothing more can come
            return
        if monotonic() - last_sent >= heartbeat_s:
            yield ": hb\n\n"
            last_sent = monotonic()
        await sleep(poll_s)
