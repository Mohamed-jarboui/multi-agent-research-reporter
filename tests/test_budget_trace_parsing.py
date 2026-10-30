import json

import pytest

from reporter.agent.parsing import cited_numbers, extract_json, strip_sources_section
from reporter.budget import Budget, BudgetMeter
from reporter.pricing import call_cost_usd
from reporter.trace import StepRecord, ToolCallRecord, Trace


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_budget_rejects_nonsense():
    with pytest.raises(ValueError):
        Budget(max_steps=0)
    with pytest.raises(ValueError):
        Budget(stop_fraction=1.5)


def test_meter_stops_at_80_percent_not_before():
    budget = Budget(max_steps=10, max_tokens=1000, max_seconds=100, max_cost_usd=1)
    meter = BudgetMeter(budget, Clock())
    assert meter.stop_reason() is None
    meter.add_call(tokens=799, cost_usd=0.1)
    assert meter.stop_reason() is None
    meter.add_call(tokens=1, cost_usd=0.0)
    assert meter.stop_reason() == "budget_tokens"


@pytest.mark.parametrize(
    ("what", "expected"),
    [("steps", "budget_steps"), ("seconds", "budget_seconds"), ("cost", "budget_cost")],
)
def test_meter_each_dimension(what, expected):
    clock = Clock()
    budget = Budget(max_steps=10, max_tokens=10**6, max_seconds=100, max_cost_usd=1)
    meter = BudgetMeter(budget, clock)
    if what == "steps":
        for _ in range(8):
            meter.add_step()
    elif what == "seconds":
        clock.now = 80
    else:
        meter.add_call(0, 0.8)
    assert meter.stop_reason() == expected


def test_cost_is_none_for_unknown_model_or_unreported_tokens():
    assert call_cost_usd("gemini-3.6-flash", 100, 50) == 0.0  # free tier
    assert call_cost_usd("mystery-model", 100, 50) is None
    assert call_cost_usd("gemini-3.6-flash", None, 50) is None


@pytest.mark.parametrize(
    "text",
    ['{"a": 1}', '```json\n{"a": 1}\n```', 'Sure! {"a": 1} hope that helps', 'x {bad} {"a": 1}'],
)
def test_extract_json(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError):
        extract_json("no json here [1, 2]")


def test_strip_sources_section_and_cited_numbers():
    text = "# T\n\nClaim [1][3]. Again [1].\n\n## Sources\n\n1. a\n2. b"
    assert strip_sources_section(text) == "# T\n\nClaim [1][3]. Again [1]."
    assert cited_numbers(strip_sources_section(text)) == [1, 3]


def test_trace_totals_count_missing_tokens_instead_of_guessing(tmp_path):
    trace = Trace(run_id="r1", system="baseline", topic="t", started_at="now")
    trace.add_step(StepRecord(index=1, kind="plan", input_tokens=10, output_tokens=5, cost_usd=0.0))
    call = ToolCallRecord(name="web_search", input={}, ok=True, latency_s=0.1, observation_chars=3)
    trace.add_step(StepRecord(index=2, kind="think", tool_call=call))
    trace.finish(latency_s=2.0)
    t = trace.totals
    assert (t.llm_calls, t.react_steps, t.tool_calls) == (2, 1, 1)
    assert t.total_tokens == 15
    assert t.calls_without_token_counts == 1 and t.calls_without_price == 1
    path = trace.save(tmp_path / "runs")
    assert json.loads(path.read_text(encoding="utf-8"))["run_id"] == "r1"
