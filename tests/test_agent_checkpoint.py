"""Agent-side pieces of R4: meter restore, node output memo, summary merge, run without a store."""

from __future__ import annotations

from datetime import date

from fakes import FakeClock, TeamLLM, team_tools
from reporter.agent import MultiAgent
from reporter.agent.checkpoint import (
    decode_output,
    encode_output,
    events_for,
    merge_counts,
)
from reporter.agent.schemas import Finding, SectionBrief, WorkerResult
from reporter.budget import Budget, BudgetMeter
from reporter.llm.base import Generation
from reporter.tools import Page
from reporter.trace import StepRecord, Trace


def test_meter_snapshot_and_restore_continue_the_clock():
    clock = FakeClock()
    seen: list[int] = []
    meter = BudgetMeter(Budget(), clock, on_call=seen.append)
    meter.add_step()
    meter.add_call(300, 0.5)
    clock.now = 40.0
    snap = meter.snapshot()
    assert snap == {"steps": 1, "tokens": 300, "cost_usd": 0.5, "elapsed_s": 40.0}
    assert seen == [300]  # every charged call is reported

    clock.now = 1000.0  # the restart: the clock is somewhere else entirely
    fresh = BudgetMeter(Budget(), clock)
    fresh.restore(snap)
    assert (fresh.steps, fresh.tokens, fresh.cost_usd) == (1, 300, 0.5)
    assert fresh.elapsed_s == 40.0  # downtime does not count
    clock.now = 1005.0
    assert fresh.elapsed_s == 45.0


def test_merge_counts_sums_integers_recursively():
    old = {"providers": ["g"], "calls": {"g": 3}, "failures": {"g": {"RateLimitError 429": 1}}}
    new = {
        "providers": ["g", "q"],
        "calls": {"g": 2, "q": 1},
        "failures": {"g": {"RateLimitError 429": 2}, "q": {"ServerError 500": 1}},
    }
    assert merge_counts(old, new) == {
        "providers": ["g", "q"],
        "calls": {"g": 5, "q": 1},
        "failures": {"g": {"RateLimitError 429": 3}, "q": {"ServerError 500": 1}},
    }
    assert merge_counts({}, new) == new


def test_node_output_round_trips_only_the_keys_that_were_set():
    result = WorkerResult(
        section_id="s1",
        index=0,
        status="complete",
        findings=[Finding(claim="c", source_ids=[1])],
        pages=[Page(url="https://a.example", title="A", text="t")],
    )
    text = encode_output({"results": [result]})
    assert set(decode_output(text)) == {"results"}
    assert decode_output(text)["results"] == [result]
    brief = SectionBrief(id="s1", title="T", goal="g", scope=["p"])
    assert decode_output(encode_output({"sections": [brief], "report": ""}))["report"] == ""


def test_events_for_a_worker_and_the_plan():
    brief = SectionBrief(id="s1", title="T", goal="g", scope=["p"])
    assert events_for("supervisor", {"sections": [brief]}) == [
        ("plan", {"sections": [{"id": "s1", "title": "T"}]})
    ]
    result = WorkerResult(section_id="s2", index=1, status="incomplete")
    assert events_for("worker:s2", {"results": [result]}) == [
        ("worker_done", {"section_id": "s2", "status": "incomplete", "pages": 0, "findings": 0})
    ]
    assert events_for("retry:s2", {"retry_results": [result]})[0][0] == "retry"
    assert events_for("merge", {}) == []


def test_trace_dump_json_round_trips_with_resumes():
    trace = Trace(run_id="r", system="multi", topic="t", started_at="now")
    trace.add_step(StepRecord(index=0, kind="plan"))
    trace.resumes.append({"from_key": "worker:s1", "attempt": 1})
    again = Trace.model_validate_json(trace.dump_json())
    assert again.resumes == trace.resumes and len(again.steps) == 1


def test_generation_provider_defaults_to_empty():
    assert Generation("t", "m", 1, 2).provider == ""
    assert Generation("t", "m", 1, 2, provider="groq").provider == "groq"


def test_a_run_without_a_store_is_unchanged(tmp_path):
    agent = MultiAgent(
        TeamLLM(3), team_tools(), today=date(2026, 10, 31), runs_dir=tmp_path / "runs"
    )
    result = agent.run("Topic")
    assert result.trace.resumes == [] and result.trace.writer["status"] == "ok"
    assert "-" in result.trace.run_id  # the old id format, not rpt_...
    custom = agent.run("Topic", run_id="rpt_custom")
    assert custom.trace.run_id == "rpt_custom"
