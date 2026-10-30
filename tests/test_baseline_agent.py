"""The baseline agent end to end with a scripted LLM and fake tools."""

import json
from datetime import date

import pytest

from fakes import FakeLLM, FakeTool, action, page, plan
from reporter.agent import BaselineAgent
from reporter.budget import Budget
from reporter.llm.base import ProviderError

# the agent refuses "finish" twice while questions lack sources, so tests send it three times
FINISH = [action("finish")] * 3
REPORT = "# Topic\n\nClaim one [1]. Claim two [2][9].\n\n## Sources\n\n1. made up by the model"


def make_agent(replies, tmp_path, budget=None, urls=(1, 2, 3, 4)):
    tools = {
        "web_search": FakeTool("web_search"),
        "fetch_page": FakeTool("fetch_page", {page(n).url: page(n) for n in urls}),
    }
    llm = FakeLLM(replies)
    agent = BaselineAgent(
        llm, tools, budget or Budget(), today=date(2026, 1, 1), runs_dir=tmp_path / "runs"
    )
    return agent, llm, tools


def fetch(n, answers):
    return action("fetch_page", answers, url=page(n).url)


def test_stops_when_every_question_has_two_sources_and_writes_cited_report(tmp_path):
    agent, llm, tools = make_agent(
        [
            plan("Q one?", "Q two?"),
            action("web_search", [1, 2], query="topic"),
            fetch(1, [1]),
            fetch(2, [1, 2]),
            fetch(3, [2]),  # after this both questions have 2 sources
            "# Topic\n\nFirst [1]. Second [2][3].",
        ],
        tmp_path,
    )
    result = agent.run("Topic")

    assert result.trace.stop_reason == "coverage_met"
    assert result.trace.totals.react_steps == 4
    assert result.trace.coverage == {"Q one?": 2, "Q two?": 2}
    assert [s.url for s in result.trace.sources] == [page(n).url for n in (1, 2, 3)]
    assert result.report.startswith("# Topic")
    assert "## Sources" in result.report
    assert f"1. [Site 1]({page(1).url})" in result.report
    assert result.trace.checks == {"invalid_citations": [], "uncited_sources": []}
    assert "[1] Site 1" in llm.prompts[-1]  # the writer sees numbered sources


def test_models_own_sources_section_is_replaced_and_bad_citations_are_flagged(tmp_path):
    agent, _, _ = make_agent(
        [plan("Q?"), fetch(1, [1]), fetch(2, [1]), REPORT],
        tmp_path,
    )
    result = agent.run("Topic")
    assert "made up by the model" not in result.report
    assert result.report.count("## Sources") == 1
    assert result.trace.checks["invalid_citations"] == [9]


def test_budget_80_percent_stops_research_and_still_writes(tmp_path):
    # max_steps=5 -> 80% is 4 steps; the questions are never covered
    replies = [plan("Q?")] + [action("web_search", [1], query=f"q{i}") for i in range(10)]
    replies.append("# Topic\n\nNothing found.")
    agent, _, tools = make_agent(replies, tmp_path, Budget(max_steps=5))
    result = agent.run("Topic")
    assert result.trace.stop_reason == "budget_steps"
    assert result.trace.totals.react_steps == 4
    assert len(tools["web_search"].calls) == 4


def test_token_budget_is_checked_after_each_step(tmp_path):
    # every fake call uses 150 tokens; 80% of 1000 is reached on the 6th call (plan + 5 steps)
    replies = [plan("Q?")] + [action("web_search", [1], query=f"q{i}") for i in range(10)]
    agent, _, _ = make_agent(replies, tmp_path, Budget(max_tokens=1000))
    result = agent.run("Topic")  # no pages -> no write call
    assert result.trace.stop_reason == "budget_tokens"
    assert result.trace.totals.react_steps == 5


def test_model_can_finish_early(tmp_path):
    agent, _, _ = make_agent(
        [plan("Q?"), fetch(1, [1]), *FINISH, "# T\n\nOnly one source [1]."], tmp_path
    )
    result = agent.run("Topic")
    assert result.trace.stop_reason == "model_finished"
    assert result.trace.coverage == {"Q?": 1}


def test_early_finish_is_refused_twice_with_a_reason_then_accepted(tmp_path):
    agent, llm, _ = make_agent(
        [plan("Q?"), fetch(1, [1]), *FINISH, "# T\n\nOnly one source [1]."], tmp_path
    )
    result = agent.run("Topic")
    assert result.trace.totals.react_steps == 4  # fetch, two refused finishes, accepted finish
    assert "finish refused" in llm.prompts[3]
    assert "finish refused" not in llm.prompts[1]


def test_no_pages_means_no_invented_report(tmp_path):
    agent, llm, _ = make_agent([plan("Q?"), *FINISH], tmp_path)
    result = agent.run("Topic")
    assert "No report could be written" in result.report
    assert len(llm.prompts) == 4  # plan and three finish attempts, no write call


def test_tool_error_is_shown_to_the_model_and_the_run_continues(tmp_path):
    agent, llm, _ = make_agent(
        [
            plan("Q?"),
            action("fetch_page", [1], url="https://gone.example/x"),
            fetch(1, [1]),
            *FINISH,
            "# T\n\n[1]",
        ],
        tmp_path,
    )
    result = agent.run("Topic")
    assert "HTTP 404" in llm.prompts[2]  # third prompt is the think step after the failure
    first_call = result.trace.steps[1].tool_call
    assert first_call is not None and not first_call.ok and "404" in (first_call.error or "")
    assert result.trace.totals.tool_calls == 2


def test_invalid_tool_input_and_unknown_action_do_not_crash(tmp_path):
    agent, _, _ = make_agent(
        [
            plan("Q?"),
            action("fetch_page", [1], url="ftp://bad"),
            action("teleport"),
            *FINISH,
        ],
        tmp_path,
    )
    result = agent.run("Topic")
    calls = [s.tool_call for s in result.trace.steps if s.tool_call]
    assert [c.ok for c in calls] == [False, False]


def test_duplicate_fetch_is_refused_without_calling_the_tool(tmp_path):
    agent, _, tools = make_agent(
        [plan("Q?"), fetch(1, [1]), fetch(1, [1]), *FINISH, "# T\n\n[1]"], tmp_path
    )
    result = agent.run("Topic")
    assert len(tools["fetch_page"].calls) == 1
    assert result.trace.coverage == {"Q?": 1}


def test_unparseable_reply_is_recovered_and_three_in_a_row_abort(tmp_path):
    agent, _, _ = make_agent(
        [
            plan("Q?"),
            "I think we should search",
            fetch(1, [1]),
            "nope",
            "nope",
            "nope",
            "# T\n\n[1]",
        ],
        tmp_path,
    )
    result = agent.run("Topic")
    assert result.trace.stop_reason == "model_output_unparseable"
    assert result.trace.totals.react_steps == 5


def test_plan_falls_back_to_the_topic_when_not_json(tmp_path):
    agent, _, _ = make_agent(["no json", *FINISH], tmp_path)
    result = agent.run("My topic")
    assert result.trace.questions == ["My topic"]


def test_llm_failure_while_researching_still_writes_from_what_was_read(tmp_path):
    agent, _, _ = make_agent(
        [plan("Q?"), fetch(1, [1]), ProviderError("down"), "# T\n\nPartial [1]."], tmp_path
    )
    result = agent.run("Topic")
    assert result.trace.stop_reason == "llm_error"
    assert "Partial [1]" in result.report


def test_trace_is_saved_with_steps_tokens_and_latency(tmp_path):
    ticks = iter(range(0, 1000))
    llm = FakeLLM([plan("Q?"), fetch(1, [1]), *FINISH, "# T\n\n[1]"])
    tools = {"fetch_page": FakeTool("fetch_page", {page(1).url: page(1)})}
    agent = BaselineAgent(llm, tools, runs_dir=tmp_path / "runs", clock=lambda: next(ticks))
    result = agent.run("Topic")

    saved = json.loads(result.trace_path.read_text(encoding="utf-8"))
    assert result.trace_path.parent == tmp_path / "runs"
    assert saved["run_id"] == result.trace.run_id
    kinds = [s["kind"] for s in saved["steps"]]
    assert kinds == ["plan", "think", "think", "think", "think", "write"]
    assert saved["totals"]["input_tokens"] == 600 and saved["totals"]["output_tokens"] == 300
    assert saved["totals"]["cost_usd"] == 0.0
    assert saved["totals"]["latency_s"] > 0
    assert saved["steps"][1]["tool_call"]["name"] == "fetch_page"
    assert saved["report"] == result.report


def test_trace_is_saved_even_when_the_run_fails(tmp_path):
    agent, _, _ = make_agent([ProviderError("planner down")], tmp_path)
    with pytest.raises(ProviderError):
        agent.run("Topic")
    assert agent.last_trace_path is not None
    saved = json.loads(agent.last_trace_path.read_text(encoding="utf-8"))
    assert "planner down" in saved["error"]


def test_empty_reply_is_retried_once_and_charged_to_the_same_step(tmp_path):
    agent, llm, _ = make_agent([plan("Q?"), "", *FINISH], tmp_path)
    result = agent.run("Topic")
    think = result.trace.steps[1]
    assert think.action == "finish"
    assert think.input_tokens == 200 and think.output_tokens == 100  # two calls
    assert "retried" in think.note
    assert result.trace.totals.react_steps == 3  # the retry is not a separate step
    assert len(llm.prompts) == 5  # plan, empty + retry, two more finishes
    assert llm.temperatures[:3] == [0.0, 0.0, 0.7]  # the retry changes temperature
