"""OpenTelemetry spans of a service run, the file exporter and the trace export."""

from __future__ import annotations

import json

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fakes import FakeLLM, FakeTool
from reporter import otel
from reporter.agent.llm_call import generate_charged
from reporter.budget import Budget, BudgetMeter
from reporter.llm.base import RateLimitError
from reporter.service.telemetry import export_trace, read_spans, setup_tracing
from reporter.trace import StepRecord
from svc import TOPIC, crash_after, make_service


@pytest.fixture
def memory():
    exporter = InMemorySpanExporter()
    setup_tracing(None, extra_exporters=[exporter])
    yield exporter
    otel.set_provider(None)


def run_job(svc):
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.runner.drain()
    return jid


def by_name(spans, prefix):
    return [s for s in spans if s.name.startswith(prefix)]


def test_a_run_has_job_node_worker_llm_and_tool_spans_with_the_gen_ai_attributes(tmp_path, memory):
    svc = make_service(tmp_path)
    jid = run_job(svc)
    spans = memory.get_finished_spans()
    names = {s.name for s in spans}
    assert {"job", "node:supervisor", "node:merge", "node:critic", "node:writer"} <= names
    assert {"worker:s1", "worker:s2", "worker:s3", "llm.generate", "tool:fetch_page"} <= names

    llm = by_name(spans, "llm.generate")[0]
    attrs = llm.attributes
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == attrs["gen_ai.response.model"] == "fake-model"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 50
    assert attrs["langfuse.observation.type"] == "generation"
    assert "gen_ai.usage.cost" not in attrs  # the fake model has no price: left out, not 0
    assert all(s.attributes["langfuse.session.id"] == jid for s in spans)
    job = by_name(spans, "job")[0]
    assert job.attributes["langfuse.trace.name"] == "report"
    assert job.attributes["langfuse.observation.type"] == "agent"
    assert by_name(spans, "tool:fetch_page")[0].attributes["langfuse.observation.type"] == "tool"
    assert by_name(spans, "tool:fetch_page")[0].attributes["tool.ok"] is True


def test_worker_spans_are_children_of_the_job_span_even_in_other_threads(tmp_path, memory):
    svc = make_service(tmp_path, max_parallel_workers=3)
    run_job(svc)
    spans = memory.get_finished_spans()
    job = by_name(spans, "job")[0]
    workers = by_name(spans, "worker:")
    assert len(workers) == 3
    assert {w.parent.span_id for w in workers} == {job.context.span_id}
    assert {w.context.trace_id for w in workers} == {job.context.trace_id}
    worker_ids = {w.context.span_id for w in workers}
    llm_in_workers = [s for s in by_name(spans, "llm.generate") if s.parent.span_id in worker_ids]
    tools_in_workers = [s for s in by_name(spans, "tool:") if s.parent.span_id in worker_ids]
    assert len(llm_in_workers) == 3 * 4 and len(tools_in_workers) == 3 * 2
    assert {w.attributes["reporter.phase"] for w in workers} == {"research"}


def test_no_prompt_or_page_text_is_recorded(tmp_path, memory):
    svc = make_service(tmp_path)
    run_job(svc)
    text = json.dumps([dict(s.attributes) for s in memory.get_finished_spans()])
    assert "Your section" not in text and "Facts of" not in text


def test_a_resumed_job_has_two_job_spans_with_one_session_and_cached_nodes_are_marked(
    tmp_path, memory
):
    first = make_service(tmp_path, max_parallel_workers=1, hooks_factory=crash_after("worker:s1"))
    jid = run_job(first)
    second = make_service(tmp_path, max_parallel_workers=1)
    second.runner.startup()
    second.runner.drain()
    spans = memory.get_finished_spans()
    jobs = by_name(spans, "job")
    assert len(jobs) == 2 and {j.attributes["langfuse.session.id"] for j in jobs} == {jid}
    assert [j.attributes["reporter.resumed"] for j in jobs] == [False, True]
    cached = [s for s in spans if s.attributes.get("reporter.node.cached")]
    assert {s.name for s in cached} == {"node:supervisor", "worker:s1"}


def test_a_failed_call_gets_error_status_and_the_limit_type_but_no_message(tmp_path, memory):
    error = RateLimitError("429 with secret sk-LEAK", provider="gemini", status=429)
    error.limit, error.retry_after_s = "per_day_requests", 90.0
    step = StepRecord(index=0, kind="plan")
    with pytest.raises(RateLimitError):
        generate_charged(FakeLLM([error]), step, "p", "s", 10, BudgetMeter(Budget()))
    span = by_name(memory.get_finished_spans(), "llm.generate")[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "RateLimitError"
    assert span.attributes["reporter.llm.limit_type"] == "per_day_requests"
    assert span.attributes["reporter.llm.retry_after_s"] == 90.0
    assert "sk-LEAK" not in json.dumps(span.to_json())


def test_the_cost_attribute_is_set_when_the_price_is_known(memory):
    step = StepRecord(index=0, kind="plan")
    generate_charged(
        FakeLLM(["ok"]), step, "p", "s", 10, BudgetMeter(Budget()), cost_fn=lambda m, i, o: 0.0125
    )
    span = by_name(memory.get_finished_spans(), "llm.generate")[0]
    assert span.attributes["gen_ai.usage.cost"] == 0.0125


def test_a_failing_tool_span_records_ok_false(memory):
    from reporter.tools import call_tool

    out = call_tool(FakeTool("fetch_page"), {"url": "https://nowhere.example"})
    span = by_name(memory.get_finished_spans(), "tool:fetch_page")[0]
    assert out.error and span.attributes["tool.ok"] is False


def test_without_a_provider_spans_are_no_ops_and_nothing_is_written(tmp_path):
    svc = make_service(tmp_path)
    jid = run_job(svc)
    assert svc.store.get_job(jid)["state"] == "succeeded"
    assert not list(svc.settings.runs_dir.glob("*.spans.jsonl"))


def test_the_file_exporter_round_trips_and_the_export_script_labels_synthetic_runs(tmp_path):
    runs = tmp_path / "data" / "runs"  # where the service reads them back
    setup_tracing(runs)
    try:
        svc = make_service(tmp_path)
        jid = run_job(svc)
    finally:
        otel.set_provider(None)
    spans = read_spans(runs, jid)
    assert {s["name"] for s in spans} >= {"job", "worker:s1", "llm.generate"}
    assert all(s["attributes"]["langfuse.session.id"] == jid for s in spans)
    out = tmp_path / "out" / "example.json"
    doc = export_trace(runs, jid, out, synthetic=True)
    saved = json.loads(out.read_text("utf-8"))
    assert saved["synthetic"] is True and "not measurements" in saved["label"]
    assert saved["run_id"] == jid and saved["n_spans"] == len(saved["spans"]) == len(doc["spans"])
    real = export_trace(runs, jid, out, synthetic=False)
    assert real["synthetic"] is False and "fake" not in real["label"]
    # the trace endpoint serves the same spans of a finished job
    got = svc.client.get(f"/reports/{jid}/trace").json()
    assert got["run_id"] == jid and len(got["spans"]) == len(spans)
