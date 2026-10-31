"""Checkpoints of a multi-agent run: what the agent needs from a store, and how node outputs are
saved and replayed.

Every graph node is a memo: its output (the dict it returns) is saved when the node finishes. A
resumed run starts the graph again from the top; a node whose key is saved returns the saved output
at once, so only the unfinished nodes run. The store itself is in `reporter.service.store`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from reporter.agent.schemas import MergedResearch, Review, SectionBrief, WorkerResult


class JobCancelled(Exception):
    """The job was cancelled; seen at the next node boundary."""


@dataclass(frozen=True)
class Snapshot:
    """Everything outside the graph state that a resume needs (latest checkpoint only)."""

    trace_json: str
    meter: dict[str, float]  # steps, tokens, cost_usd, elapsed_s
    llm: dict[str, Any]  # router summary so far, summed over the earlier run segments
    last_key: str


class CheckpointStore(Protocol):
    """What `MultiAgent.run` needs from a store (one job)."""

    def load(self, key: str) -> str | None: ...

    def load_snapshot(self) -> Snapshot | None: ...

    def cancel_requested(self) -> bool: ...

    def commit(
        self,
        key: str,
        output_json: str,
        snapshot: Callable[[], Snapshot],
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Save the output, the snapshot (built under the store's lock, so snapshots never go
        back in time) and the events in one transaction."""


class RunHooks(Protocol):
    def after_checkpoint(self, key: str) -> None: ...


class NodeOutput(BaseModel):
    """The dict a node returns, as JSON. Fields match `State`; only the ones set are replayed."""

    sections: list[SectionBrief] | None = None
    results: list[WorkerResult] | None = None
    retry_results: list[WorkerResult] | None = None
    merged: MergedResearch | None = None
    digest: str | None = None
    review: Review | None = None
    retry_ids: list[str] | None = None
    report: str | None = None


def encode_output(output: dict[str, Any]) -> str:
    return NodeOutput(**output).model_dump_json(exclude_unset=True)


def decode_output(text: str) -> dict[str, Any]:
    node = NodeOutput.model_validate_json(text)
    return {name: getattr(node, name) for name in node.model_fields_set}


def merge_counts(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Sum two router summaries: integers add, dicts merge, anything else takes the newer value."""
    out: dict[str, Any] = dict(old)
    for key, value in new.items():
        before = out.get(key)
        if isinstance(value, dict) and isinstance(before, dict):
            out[key] = merge_counts(before, value)
        elif isinstance(value, int | float) and isinstance(before, int | float):
            out[key] = before + value
        else:
            out[key] = value
    return out


def events_for(
    key: str, output: dict[str, Any], *, writer_status: str = ""
) -> list[tuple[str, dict[str, Any]]]:
    """The progress events of a finished node (the fixed vocabulary of the SSE stream)."""
    if key == "supervisor":
        sections = [{"id": b.id, "title": b.title} for b in output["sections"]]
        return [("plan", {"sections": sections})]
    if key.startswith(("worker:", "retry:")):
        result = output["results" if key.startswith("worker:") else "retry_results"][0]
        data = {
            "section_id": result.section_id,
            "status": result.status,
            "pages": len(result.pages),
            "findings": len(result.findings),
        }
        return [("worker_done" if key.startswith("worker:") else "retry", data)]
    if key in ("critic", "regrade"):
        review = output["review"]
        first = key == "critic"
        data = {
            "stage": "first" if first else "after_retry",
            "verdicts": {sid: r.verdict for sid, r in review.sections.items()},
            "retrying": list(output.get("retry_ids", [])) if first else [],
        }
        return [("critic_verdict", data)]
    if key == "writer":
        return [
            ("written", {"writer_status": writer_status, "words": len(output["report"].split())})
        ]
    return []
