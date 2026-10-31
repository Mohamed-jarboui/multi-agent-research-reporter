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


def export_trace(
    runs_dir: Path, run_id: str, out: Path, *, synthetic: bool, label: str | None = None
) -> dict[str, Any]:
    """Write one run's spans as `{label, synthetic, run_id, exported_at, spans}`."""
    spans = read_spans(runs_dir, run_id)
    spans.sort(key=lambda s: s.get("start_time") or "")
    document = {
        "label": label or (SYNTHETIC_LABEL if synthetic else "real run"),
        "synthetic": synthetic,
        "run_id": run_id,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_spans": len(spans),
        "spans": spans,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return document
