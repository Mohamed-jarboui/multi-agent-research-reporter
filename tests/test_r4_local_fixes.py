"""Fixes from the R4 local test: log masking, crash placeholders, failed-span provider, calls."""

from __future__ import annotations

import io
import logging

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from uvicorn.logging import AccessFormatter

from fakes import FakeLLM
from reporter import otel
from reporter.agent.llm_call import generate_charged
from reporter.budget import Budget, BudgetExhausted, BudgetMeter
from reporter.llm.base import AllProvidersFailedError, RateLimitError
from reporter.service.app import install_key_mask
from reporter.service.telemetry import add_placeholder_parents, export_trace, setup_tracing
from reporter.trace import StepRecord, Trace
from svc import TOPIC, crash_after, make_service

SECRET = "sk-FAKEKEY-0123456789"


def capture(formatter: logging.Formatter) -> tuple[logging.Logger, io.StringIO]:
    install_key_mask()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger = logging.getLogger("uvicorn.access.test.child")
    logger.handlers, logger.propagate = [handler], False
    logger.setLevel(logging.INFO)
    return logger, stream


def test_a_uvicorn_access_record_with_five_args_still_formats(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    logger, stream = capture(AccessFormatter("%(client_addr)s - %(request_line)s %(status_code)s"))
    logger.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5", "GET", "/health", "1.1", 200)
    assert "GET /health HTTP/1.1" in stream.getvalue() and "200" in stream.getvalue()


def test_a_record_holding_a_key_is_masked_and_one_without_keeps_its_args(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    logger, stream = capture(logging.Formatter("%(message)s"))
    logger.info("key is %s", SECRET)
    logger.info("n is %d", 7)
    assert SECRET not in stream.getvalue() and "key is ***" in stream.getvalue()
    assert "n is 7" in stream.getvalue()


def test_an_exception_message_holding_a_key_is_masked(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    logger, stream = capture(logging.Formatter("%(message)s"))
    try:
        raise RuntimeError(f"upstream rejected {SECRET}")
    except RuntimeError:
        logger.exception("call failed")
    assert SECRET not in stream.getvalue() and "rejected ***" in stream.getvalue()


def span(name, span_id, parent, start, end, **attrs):
    return {
        "name": name,
        "context": {"trace_id": "0xaa", "span_id": span_id, "trace_state": "[]"},
        "parent_id": parent,
        "start_time": start,
        "end_time": end,
        "status": {"status_code": "UNSET"},
        "attributes": {"langfuse.session.id": "rpt_x", **attrs},
    }


def test_missing_parents_get_one_synthetic_placeholder_each():
    spans = [
        span("job", "0xj2", None, "T05", "T09"),
        span("node:supervisor", "0xn1", "0xj1", "T01", "T02"),  # job 1 never ended
        span("worker:s1", "0xw1", "0xj1", "T02", "T04"),
        span("llm.generate", "0xl1", "0xw2", "T02", "T03"),  # worker s2 never ended
        span("llm.generate", "0xl2", "0xw2", "T03", "T05"),
    ]
    out = add_placeholder_parents(spans)
    ids = {s["context"]["span_id"] for s in out}
    assert all(s["parent_id"] in ids for s in out if s["parent_id"])
    made = [s for s in out if s["attributes"].get("reporter.placeholder")]
    assert len(made) == 2
    job = next(s for s in made if s["name"] == "job (crashed segment)")
    assert job["context"]["span_id"] == "0xj1"
    assert (job["start_time"], job["end_time"]) == ("T01", "T04")
    assert job["attributes"]["langfuse.session.id"] == "rpt_x"
    worker = next(s for s in made if s is not job)
    assert worker["parent_id"] == "0xj1" and worker["start_time"] == "T02"
    assert add_placeholder_parents(out) == out  # nothing left to add


def test_a_resumed_job_exports_a_whole_tree_without_placeholders(tmp_path):
    runs = tmp_path / "data" / "runs"
    setup_tracing(runs)
    try:
        first = make_service(
            tmp_path, max_parallel_workers=1, hooks_factory=crash_after("worker:s1")
        )
        jid = first.client.post("/reports", json=TOPIC).json()["id"]
        first.runner.drain()
        second = make_service(tmp_path, max_parallel_workers=1)
        second.runner.startup()
        second.runner.drain()
    finally:
        otel.set_provider(None)
    doc = export_trace(runs, jid, tmp_path / "t.json", synthetic=True)
    ids = {s["context"]["span_id"] for s in doc["spans"]}
    assert all(s["parent_id"] in ids for s in doc["spans"] if s["parent_id"])
    # a simulated crash still ends its job span (a real one does not), so nothing is synthetic
    assert doc["n_placeholder_spans"] == 0
    assert len([s for s in doc["spans"] if s["name"] == "job"]) == 2


def failing_call(err):
    memory = InMemorySpanExporter()
    setup_tracing(None, extra_exporters=[memory])
    try:
        with pytest.raises(type(err)):
            generate_charged(
                FakeLLM([err]),
                StepRecord(index=0, kind="plan"),
                "p",
                "s",
                10,
                BudgetMeter(Budget()),
            )
    finally:
        otel.set_provider(None)
    return memory.get_finished_spans()[0].attributes


def test_a_failed_call_names_the_last_provider_and_compact_failures():
    err = AllProvidersFailedError("all failed: raw text sk-LEAK", provider="groq")
    err.failures = [
        {"provider": "gemini", "reason": "RateLimitError 429", "limit": "per_day_requests"},
        {"provider": "groq", "reason": "EmptyReplyError"},
        {"provider": "groq", "reason": "EmptyReplyError"},
    ]
    attrs = failing_call(err)
    assert attrs["gen_ai.system"] == "groq"
    assert attrs["reporter.provider_failures"] == (
        "gemini:RateLimitError 429 per_day_requests;groq:EmptyReplyError x2"
    )
    assert "sk-LEAK" not in str(dict(attrs))


def test_a_single_provider_error_still_gets_the_provider_name():
    attrs = failing_call(RateLimitError("x", provider="gemini", status=429))
    assert attrs["gen_ai.system"] == "gemini"


def test_a_step_refused_by_the_budget_is_not_counted_as_a_model_call():
    meter = BudgetMeter(Budget(max_steps=1))
    meter.add_step()
    meter.add_step()
    refused = StepRecord(index=1, kind="write")
    with pytest.raises(BudgetExhausted):
        generate_charged(FakeLLM([]), refused, "p", "s", 10, meter, enforce_limit=True)
    assert refused.calls == 0
    ran = StepRecord(index=2, kind="plan")
    ran.calls = 2  # an empty-reply retry
    trace = Trace(run_id="r", system="multi", topic="t", started_at="now", steps=[refused, ran])
    trace.finish(1.0)
    assert trace.totals.llm_calls == 2
    trace.steps.append(StepRecord(index=3, kind="plan"))  # calls not recorded: counts as one
    trace.finish(1.0)
    assert trace.totals.llm_calls == 3


def test_the_result_has_a_coverage_object_that_adds_up_to_the_sections(tmp_path):
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.runner.drain()
    result = svc.client.get(f"/reports/{jid}").json()["result"]
    coverage = result["coverage"]
    assert set(coverage) == {"sections_ok", "sections_weak", "sections_missing"}
    assert sum(coverage.values()) == len(result["sections"])
