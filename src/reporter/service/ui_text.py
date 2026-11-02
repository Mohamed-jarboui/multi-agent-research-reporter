"""Plain-text pieces of the Gradio demo page (no Gradio import, so they are easy to test).

The Space UI in `deploy/space_app.py` only wires these to widgets: it turns the service's
public JSON (a job, an event, the quota) into short strings and one Markdown document.
"""

from __future__ import annotations

import json
from typing import Any


def quota_line(quota: dict[str, Any]) -> str:
    """One line about the daily cap, from the `GET /quota` answer."""
    jobs, queue = quota["jobs"], quota["queue"]
    line = (
        f"Today (UTC): {jobs['left']} of {jobs['limit']} runs left; "
        f"{queue['waiting']} waiting. Resets at {quota['resets_at']}."
    )
    if quota.get("latch_until"):
        line += f" The model quota is used up until {quota['latch_until']}."
    return line


def event_line(event: dict[str, Any]) -> str:
    """One line per progress event: `plan {...}`."""
    return f"{event['type']} {json.dumps(event['data'], ensure_ascii=False)}"


def status_line(job: dict[str, Any]) -> str:
    """The job state, with the plain reason when it is not a clean success."""
    text = str(job["state"])
    if job.get("state") == "queued" and job.get("queue_position") is not None:
        text += f" (position {job['queue_position']})"
    progress = job.get("progress")
    if progress and progress.get("sections_total"):
        text += f" ({progress['sections_done']}/{progress['sections_total']} sections done)"
    if job.get("message"):
        text += f": {job['message']}"
    return text


def report_markdown(job: dict[str, Any], quota: dict[str, Any] | None) -> str:
    """The final report plus its coverage summary and the quota left, as one Markdown text."""
    parts: list[str] = []
    if job["state"] == "degraded" and job.get("message"):
        parts.append(f"> **Degraded run:** {job['message']}\n")
    result = job.get("result")
    if result:
        parts.append(result.get("report_markdown") or "_No report text._")
        cov = result["coverage"]
        parts.append(
            "\n---\n**Coverage:** "
            f"{cov['sections_ok']} sections ok, {cov['sections_weak']} weak, "
            f"{cov['sections_missing']} missing."
        )
    if quota:
        parts.append(f"\n_{quota_line(quota)}_")
    return "\n".join(parts)
