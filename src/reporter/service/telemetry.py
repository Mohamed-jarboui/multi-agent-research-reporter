"""OpenTelemetry for the service: the SDK provider, our own file exporter, and the trace export.

Spans go to `runs/<run id>.spans.jsonl`, one JSON object per line, appended as each span ends, so
they survive a crash. No collector and no Langfuse SDK: the attribute names are the ones Langfuse
reads, so the file can be sent to it later.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from reporter import otel

SYNTHETIC_LABEL = "fake LLM and tools; numbers are not measurements"


class SessionProcessor(SpanProcessor):
    """Copies `langfuse.session.id` from the parent to each new span, so every span of a run
    carries the run id (the job span is the only one that sets it)."""

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        parent = trace.get_current_span(parent_context)
        attributes = getattr(parent, "attributes", None) or {}
        session = attributes.get(otel.SESSION_ID)
        if session:
            span.set_attribute(otel.SESSION_ID, session)


class JsonFileSpanExporter(SpanExporter):
    """Append each finished span to `<runs_dir>/<session id>.spans.jsonl`."""

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            for span in spans:
                session = (span.attributes or {}).get(otel.SESSION_ID)
                if not session:
                    continue
                line = span.to_json(indent=None)
                with open(self.runs_dir / f"{session}.spans.jsonl", "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:  # nothing is held open
        return None


def setup_tracing(
    runs_dir: Path | None, *, extra_exporters: Sequence[SpanExporter] = ()
) -> TracerProvider:
    """Build a provider (file exporter when `runs_dir` is given) and make the agents use it."""
    provider = TracerProvider(resource=Resource.create({"service.name": "research-reporter"}))
    provider.add_span_processor(SessionProcessor())
    if runs_dir is not None:
        provider.add_span_processor(SimpleSpanProcessor(JsonFileSpanExporter(runs_dir)))
    for exporter in extra_exporters:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel.set_provider(provider)
    return provider


def read_spans(runs_dir: Path, run_id: str) -> list[dict[str, Any]]:
    path = Path(runs_dir) / f"{run_id}.spans.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def add_placeholder_parents(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`spans` plus one synthetic parent for each parent id that is not in the list.

    A segment that crashed never ended its `job` span (or its in-flight worker spans), so those
    spans were never written and their children point to nothing. Each missing id gets a
    placeholder spanning its children (min start to max end), marked `reporter.placeholder=true`.
    It is named `job (crashed segment)` when its children are nodes or workers, else
    `unfinished span (crashed segment)` (an in-flight worker); the latter is attached to the job
    placeholder when there is exactly one.
    """
    known = {s["context"]["span_id"] for s in spans}
    children: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        parent = span.get("parent_id")
        if parent and parent not in known:
            children.setdefault(parent, []).append(span)
    made: list[dict[str, Any]] = []
    for parent_id, kids in children.items():
        is_job = any(k["name"].startswith(("node:", "worker:")) for k in kids)
        session = next(
            (k["attributes"][otel.SESSION_ID] for k in kids if otel.SESSION_ID in k["attributes"]),
            None,
        )
        made.append(
            {
                "name": "job (crashed segment)" if is_job else "unfinished span (crashed segment)",
                "context": {
                    "trace_id": kids[0]["context"]["trace_id"],
                    "span_id": parent_id,
                    "trace_state": "[]",
                },
                "kind": "SpanKind.INTERNAL",
                "parent_id": None,
                "start_time": min(k["start_time"] for k in kids),
                "end_time": max(k["end_time"] for k in kids),
                "status": {"status_code": "UNSET"},
                "attributes": {
                    otel.OBSERVATION_TYPE: "agent" if is_job else "span",
                    **({otel.SESSION_ID: session} if session else {}),
                    "reporter.placeholder": True,
                },
                "events": [],
                "links": [],
            }
        )
    jobs = [m for m in made if m["name"] == "job (crashed segment)"]
    if len(jobs) == 1:
        for m in made:
            if m is not jobs[0]:
                m["parent_id"] = jobs[0]["context"]["span_id"]
    return [*spans, *made]


def export_trace(
    runs_dir: Path, run_id: str, out: Path, *, synthetic: bool, label: str | None = None
) -> dict[str, Any]:
    """Write one run's spans as `{label, synthetic, run_id, exported_at, spans}`."""
    spans = add_placeholder_parents(read_spans(runs_dir, run_id))
    spans.sort(key=lambda s: s.get("start_time") or "")
    document = {
        "label": label or (SYNTHETIC_LABEL if synthetic else "real run"),
        "synthetic": synthetic,
        "run_id": run_id,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_spans": len(spans),
        "n_placeholder_spans": sum(1 for s in spans if s["attributes"].get("reporter.placeholder")),
        "spans": spans,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document
