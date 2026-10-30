"""The multi-agent graph with a fake model and fake tools."""

import json
import threading
import time
from datetime import date

import pytest

from fakes import (
    FakeClock,
    FakeLLM,
    FakeTool,
    TeamLLM,
    action,
    page,
    sections_json,
    team_pages,
    team_tools,
)
from reporter.agent import MultiAgent
from reporter.agent.llm_call import LLMGate
from reporter.agent.multi import parse_plan
from reporter.agent.worker import WorkerContext, run_worker
from reporter.budget import Budget, BudgetMeter, MultiBudget
from reporter.llm.base import AllProvidersFailedError, ProviderError, RateLimitError
from reporter.trace import Trace

TODAY = date(2026, 10, 30)


def agent(llm, tools, tmp_path, budget=None, **kw) -> MultiAgent:
    kw.setdefault("critic", False)  # these tests are about R1; R2 is in test_critic_writer.py
    kw.setdefault("writer", False)
    return MultiAgent(llm, tools, budget, today=TODAY, runs_dir=tmp_path / "runs", **kw)


def statuses(result) -> dict[str, str]:
    return {s.brief.id: s.status for s in result.merged.sections}


def test_happy_path_returns_a_cited_digest_and_a_full_trace(tmp_path):
    result = agent(TeamLLM(3), team_tools(), tmp_path).run("Topic")

    assert statuses(result) == {"s1": "complete", "s2": "complete", "s3": "complete"}
    assert len(result.merged.sources) == 6
    assert "[1][2]" in result.report and "## Sources" in result.report
    assert result.trace_path.exists()
    trace = result.trace
    assert trace.system == "multi" and len(trace.sections) == 3
    assert {w.section_id for w in trace.workers} == {"s1", "s2", "s3"}
    assert all(w.started_at and w.ended_s >= w.started_s for w in trace.workers)
    assert trace.concurrency == {"max_parallel_workers": 3, "max_parallel_llm_calls": 2}
    assert trace.totals.llm_calls == 1 + 3 * 4  # plan + (2 fetch + finish + final) per section
    assert trace.stop_reason == "all_sections_done"


def test_workers_really_run_at_the_same_time(tmp_path):
    barrier = threading.Barrier(3, timeout=5)

    class BarrierFetch(FakeTool):
        def run(self, args):
            if args.url.endswith("-1.example/p"):  # first page of each section
                barrier.wait()  # raises BrokenBarrierError unless all 3 workers are here
                time.sleep(0.05)
            return super().run(args)

    fetch = BarrierFetch("fetch_page", team_pages())
    result = agent(TeamLLM(3), team_tools(fetch), tmp_path).run("Topic")

    assert set(statuses(result).values()) == {"complete"}
    spans = [(w.started_s, w.ended_s) for w in result.trace.workers]
    assert max(s for s, _ in spans) < min(e for _, e in spans)  # all intervals overlap
    # steps of different workers are interleaved and still get unique indexes
    indexes = [s.index for s in result.trace.steps]
    assert indexes == list(range(1, len(indexes) + 1))
    assert {s.worker for s in result.trace.steps if s.kind != "plan"} == {"s1", "s2", "s3"}


def test_max_parallel_workers_is_respected(tmp_path):
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    class CountingFetch(FakeTool):
        def run(self, args):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.03)
            try:
                return super().run(args)
            finally:
                with lock:
                    state["now"] -= 1

    fetch = CountingFetch("fetch_page", team_pages())
    budget = MultiBudget(max_parallel_workers=1)
    result = agent(TeamLLM(3), team_tools(fetch), tmp_path, budget).run("Topic")

    assert set(statuses(result).values()) == {"complete"}
    assert state["peak"] == 1
    spans = sorted((w.started_s, w.ended_s) for w in result.trace.workers)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:], strict=False))  # no overlap


def test_a_failed_attempt_is_retried_once(tmp_path):
    def on_call(kind, title, nth):
        if kind == "think" and title == "Part 1" and nth == 1:
            return ProviderError("boom")

    result = agent(TeamLLM(3, on_call), team_tools(), tmp_path).run("Topic")

    assert statuses(result)["s1"] == "complete"
    assert result.merged.sections[0].attempts == 2
    s1 = [w for w in result.trace.workers if w.section_id == "s1"]
    assert sorted((w.attempt, w.status) for w in s1) == [(1, "incomplete"), (2, "complete")]
    assert "boom" in (next(w for w in s1 if w.attempt == 1).error or "")


def test_a_section_that_fails_twice_is_incomplete_and_the_run_goes_on(tmp_path):
    def on_call(kind, title, nth):
        if title == "Part 2" and kind == "think":
            return ProviderError("always down")

    result = agent(TeamLLM(3, on_call), team_tools(), tmp_path).run("Topic")

    assert statuses(result) == {"s1": "complete", "s2": "incomplete", "s3": "complete"}
    assert result.merged.incomplete == ["s2"]
    assert "INCOMPLETE" in result.report and "always down" not in result.report
    assert len([w for w in result.trace.workers if w.section_id == "s2"]) == 2
    assert result.trace.stop_reason == "sections_incomplete"


def test_zero_pages_counts_as_a_failure(tmp_path):
    def on_call(kind, title, nth):
        if title == "Part 3" and kind == "think":
            return action("finish")  # refused once, then accepted with 0 pages

    result = agent(TeamLLM(3, on_call), team_tools(), tmp_path).run("Topic")

    assert statuses(result)["s3"] == "incomplete"
    assert statuses(result)["s1"] == "complete"
    assert "no page was fetched" in (result.merged.sections[2].error or "")


def test_rate_limit_triggers_a_shared_cooldown_and_one_retry(tmp_path):
    clock = FakeClock()
    hit = {"done": False}

    class RateLimited(TeamLLM):
        def generate(self, prompt, **kw):
            if "Your section: Part 1" in prompt and not hit["done"]:
                hit["done"] = True
                raise AllProvidersFailedError("all failed") from RateLimitError("429")
            return super().generate(prompt, **kw)

    result = agent(RateLimited(3), team_tools(), tmp_path, clock=clock, sleep=clock.sleep).run(
        "Topic"
    )

    assert set(statuses(result).values()) == {"complete"}
    waits = result.trace.rate_limit_waits
    assert waits and all(w["waited_s"] > 0 for w in waits)
    assert clock.sleeps[0] == pytest.approx(10.0)
    assert [w.attempt for w in result.trace.workers if w.section_id == "s1"] == [1]


def test_cost_ceiling_stops_new_calls_and_does_not_retry(tmp_path):
    per_call = 0.004
    ceiling = 0.01
    budget = MultiBudget(run=Budget(max_steps=36, max_cost_usd=ceiling))
    result = agent(
        TeamLLM(3), team_tools(), tmp_path, budget, cost_fn=lambda m, i, o: per_call
    ).run("Topic")

    assert result.trace.totals.cost_usd <= ceiling + budget.max_parallel_llm_calls * per_call + 1e-9
    assert "incomplete" in statuses(result).values()
    exhausted = [w for w in result.trace.workers if w.stop_reason == "budget_exhausted"]
    for w in exhausted:  # no second attempt after BudgetExhausted
        assert len([x for x in result.trace.workers if x.section_id == w.section_id]) == 1


def test_supervisor_retries_once_when_the_plan_is_unusable(tmp_path):
    def on_call(kind, title, nth):
        if kind == "plan" and nth == 1:
            return "not json"

    llm = TeamLLM(3, on_call)
    result = agent(llm, team_tools(), tmp_path).run("Topic")

    assert llm.counts[("plan", "")] == 2
    assert len(result.merged.sections) == 3


def test_supervisor_short_plan_is_kept_with_a_trace_note(tmp_path):
    result = agent(TeamLLM(2), team_tools(), tmp_path).run("Topic")
    assert len(result.merged.sections) == 2
    assert any(n.startswith("plan_short") for n in result.trace.notes)


# --------------------------------------------------------------- worker alone


def make_ctx(llm, tools, budget=None) -> WorkerContext:
    budget = budget or MultiBudget()
    meter = BudgetMeter(budget.run)
    return WorkerContext(
        topic="T",
        today="2026-10-30",
        llm=llm,
        tools=tools,
        budget=budget,
        meter=meter,
        gate=LLMGate(2),
        trace=Trace(run_id="r", system="multi", topic="T", started_at="now"),
    )


def one_brief():
    return parse_plan(sections_json(3), "T")[0]


def test_worker_drops_unfetched_sources_and_unsupported_findings_and_caps_confidence():
    final = json.dumps(
        {
            "findings": [
                {"claim": "kept", "sources": [1, 7]},  # 7 was never fetched
                {"claim": "dropped", "sources": [7]},  # nothing left to cite
                {"claim": "", "sources": [1]},
            ],
            "confidence": "high",
            "open_questions": ["still open?"],
        }
    )
    script = [action("fetch_page", url=page(1).url), action("finish"), action("finish"), final]
    llm = FakeLLM(script)
    tools = {"fetch_page": FakeTool("fetch_page", {page(1).url: page(1)})}

    res = run_worker(make_ctx(llm, tools), one_brief(), 0)

    assert res.status == "complete" and res.attempts == 1
    assert [(f.claim, f.source_ids) for f in res.findings] == [("kept", [1])]
    assert res.confidence == "low"  # only one page was fetched
    assert res.open_questions == ["still open?"]
    assert "finish refused" in llm.prompts[2]  # the first finish was refused (1 page < 2)


def test_worker_with_only_invalid_sources_has_no_findings():
    final = json.dumps({"findings": [{"claim": "x", "sources": [4]}]})
    # two attempts: each fetches one page, finishes twice, then finalizes with a bad source
    script = [action("fetch_page", url=page(1).url), action("finish"), action("finish"), final]
    llm = FakeLLM(script + script)
    tools = {"fetch_page": FakeTool("fetch_page", {page(1).url: page(1)})}

    res = run_worker(make_ctx(llm, tools), one_brief(), 0)

    assert res.status == "incomplete" and res.findings == [] and res.attempts == 2
    assert len(res.pages) == 1  # the page is kept for the record


def test_meter_is_safe_under_threads():
    meter = BudgetMeter(Budget(max_steps=8000))

    def hammer():
        for _ in range(1000):
            meter.add_step()
            meter.add_call(1, 0.001)

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert meter.steps == 8000 and meter.tokens == 8000
    assert meter.cost_usd == pytest.approx(8.0)
    assert meter.hard_limit_reached() == "budget_steps"
