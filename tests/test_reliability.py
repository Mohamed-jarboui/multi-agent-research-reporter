"""Reliability fixes: Groq request options, empty replies, router failure reasons, blocked fetches,
candidate URLs and seeded retries. Fakes only, no network."""

import json
from types import SimpleNamespace

import httpx
import pytest

from fakes import FakeInput, FakeLLM, FakeTool, action, page
from reporter.agent.critic import feedback_text
from reporter.agent.schemas import Review, SectionReview, WorkerResult
from reporter.agent.worker import failure_reason, run_worker
from reporter.llm import AllProvidersFailedError, Generation, RateLimitError, ServerError
from reporter.llm.base import EmptyReplyError, error_from_status
from reporter.llm.groq import REASONING_HEADROOM, GroqProvider
from reporter.tools import ToolError, ToolOutput
from reporter.tools.fetch import BROWSER_HEADERS, FetchPage
from test_llm_router import FakeProvider, make_llm
from test_multi_agent import make_ctx, one_brief

# ------------------------------------------------------------------------ groq options


def groq_with_fake_client(models=None):
    seen: list[dict] = []

    def create(**kwargs):
        seen.append(kwargs)
        message = SimpleNamespace(content="ok")
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    provider = GroqProvider("test-key", models=models)
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    return provider, seen


def test_gpt_oss_gets_no_low_temperature_low_effort_no_reasoning_and_token_headroom():
    provider, seen = groq_with_fake_client()
    provider.generate("hi", temperature=0.0, max_output_tokens=1000)
    provider.generate("hi", temperature=0.7, max_output_tokens=1000)

    first, second = seen
    assert "temperature" not in first  # R0: temperature 0 gave reasoning-only replies
    assert second["temperature"] == 0.7
    for call in seen:
        assert call["reasoning_effort"] == "low" and call["include_reasoning"] is False
        assert call["max_tokens"] == 1000 + REASONING_HEADROOM


def test_other_models_keep_their_temperature_and_limits():
    provider, seen = groq_with_fake_client({"small": "llama-x", "large": "llama-x"})
    provider.generate("hi", temperature=0.0, max_output_tokens=500)
    assert seen[0]["temperature"] == 0.0 and seen[0]["max_tokens"] == 500
    assert "include_reasoning" not in seen[0] and "reasoning_effort" not in seen[0]


# ------------------------------------------------------------ router: empty, reasons, cooldown


def test_an_empty_reply_is_a_retryable_provider_error_counted_once():
    primary = FakeProvider("groq", ["", "  ", "answer"])
    llm, sleeps = make_llm(primary)

    assert llm.generate("hi").text == "answer"
    assert primary.calls == 3 and llm.empty_replies == 2
    assert [f["reason"] for f in llm.failures] == ["EmptyReplyError", "EmptyReplyError"]
    assert sleeps == [1.0, 2.0]


def test_persistent_empty_replies_fall_back_to_the_next_provider():
    first = FakeProvider("groq", [""] * 3)
    second = FakeProvider("gemini", ["from gemini"])
    llm, _ = make_llm(first, second)

    assert llm.generate("hi").text == "from gemini"
    assert llm.summary()["empty_replies"] == 3
    assert llm.summary()["calls"] == {"groq": 3, "gemini": 1}


def test_all_providers_failed_says_why_without_raw_text():
    secret = "org_01abc https://console.example/billing"
    first = FakeProvider("gemini", [error_from_status(429, secret, provider="gemini")] * 3)
    second = FakeProvider("groq", [ServerError(secret)] * 3)
    llm, _ = make_llm(first, second)

    with pytest.raises(AllProvidersFailedError) as caught:
        llm.generate("hi")

    message = str(caught.value)
    assert "gemini: RateLimitError 429 x3" in message and "groq: ServerError x3" in message
    assert "org_01abc" not in message and "https" not in message
    assert llm.summary()["failures"] == {
        "gemini": {"RateLimitError 429": 3},
        "groq": {"ServerError": 3},
    }


def test_a_provider_that_hit_a_rate_limit_is_tried_again_on_the_next_call():
    gemini = FakeProvider("gemini", [RateLimitError("429")] * 3 + ["gemini is back"])
    groq = FakeProvider("groq", ["from groq"])
    llm, _ = make_llm(gemini, groq)

    assert llm.generate("one").text == "from groq"
    assert llm.generate("two").text == "gemini is back"  # not disabled for the run
    assert EmptyReplyError("x").status is None


# ---------------------------------------------------------------------------- fetch


def html(text="Readable words. " * 30):
    return httpx.Response(
        200, text=f"<html><body><p>{text}</p></body></html>", headers={"content-type": "text/html"}
    )


def test_fetch_retries_once_with_browser_headers_after_a_403():
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("user-agent", ""))
        return httpx.Response(403) if len(seen) == 1 else html()

    tool = FetchPage(httpx.Client(transport=httpx.MockTransport(handler)))
    out = tool.run(tool.input_model(url="https://pubs.example/a"))

    assert out.pages and len(seen) == 2
    assert seen[1] == BROWSER_HEADERS["User-Agent"] and seen[0] != seen[1]


def test_fetch_gives_up_after_the_second_403_and_says_to_pick_another_result():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429)

    tool = FetchPage(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ToolError) as caught:
        tool.run(tool.input_model(url="https://pubs.example/a"))

    assert len(calls) == 2
    assert "HTTP 429 twice" in str(caught.value) and "different search result" in str(caught.value)


# ------------------------------------------------------- worker: candidates and seeded retry


class SearchTool(FakeTool):
    def run(self, args: FakeInput) -> ToolOutput:
        self.calls.append(args)
        urls = [
            "https://blocked.example/a",
            "https://blocked.example/b",
            "https://files.example/paper.pdf",
            "https://ok1.example/p",
            "https://ok2.example/p",
        ]
        return ToolOutput(
            observation="\n".join(f"{i}. T\n   {u}\n   snippet" for i, u in enumerate(urls, 1))
        )


class BlockingFetch(FakeTool):
    def run(self, args: FakeInput) -> ToolOutput:
        if "blocked" in args.url:
            self.calls.append(args)
            raise ToolError(f"{args.url} answered HTTP 403 twice. Pick another source.")
        return super().run(args)


def ok_pages():
    pages = {}
    for n in (1, 2):
        pages[f"https://ok{n}.example/p"] = page(n).model_copy(
            update={"url": f"https://ok{n}.example/p"}
        )
    return pages


FINAL = json.dumps({"findings": [{"claim": "c", "sources": [1, 2]}], "confidence": "high"})


def test_after_a_403_the_worker_is_told_which_search_results_remain():
    llm = FakeLLM(
        [
            action("web_search", query="q"),
            action("fetch_page", url="https://blocked.example/a"),
            action("fetch_page", url="https://ok1.example/p"),
            action("fetch_page", url="https://ok1.example/p"),  # a repeat: costs one step only
            action("fetch_page", url="https://ok2.example/p"),
            action("finish"),
            FINAL,
        ]
    )
    tools = {
        "web_search": SearchTool("web_search"),
        "fetch_page": BlockingFetch("fetch_page", ok_pages()),
    }
    ctx = make_ctx(llm, tools)

    result = run_worker(ctx, one_brief(), 0, max_attempts=1)

    assert result.status == "complete" and len(result.pages) == 2
    after_403 = llm.prompts[2]
    hint = "fetch the first one next): https://ok1.example/p, https://ok2.example/p"
    assert hint in after_403  # blocked.example/b and the .pdf are not offered
    offered = [ln for ln in after_403.splitlines() if ln.startswith("Untried search results")]
    assert offered and "blocked.example/b" not in offered[0] and ".pdf" not in offered[0]
    repeat = llm.prompts[4]
    assert "already read this page" in repeat and "https://ok2.example/p" in repeat
    assert result.failed_fetches == [{"url": "https://blocked.example/a", "reason": "HTTP 403"}]
    assert result.queries == ["q"]


def test_a_blocked_host_is_not_fetched_again():
    llm = FakeLLM(
        [
            action("fetch_page", url="https://blocked.example/a"),
            action("fetch_page", url="https://blocked.example/b"),
            action("finish"),
            action("finish"),
        ]
    )
    fetch = BlockingFetch("fetch_page")
    ctx = make_ctx(llm, {"web_search": SearchTool("web_search"), "fetch_page": fetch})

    run_worker(ctx, one_brief(), 0, max_attempts=1)

    assert [c.url for c in fetch.calls] == ["https://blocked.example/a"]  # the second never left


def test_a_retry_starts_with_a_fetch_of_the_best_untried_candidate():
    llm = FakeLLM([action("fetch_page", url="https://ok2.example/p"), action("finish"), FINAL])
    fetch = BlockingFetch("fetch_page", ok_pages())
    ctx = make_ctx(llm, {"web_search": SearchTool("web_search"), "fetch_page": fetch})
    seeds = ["https://blocked.example/a", "https://ok1.example/p", "https://ok2.example/p"]

    result = run_worker(ctx, one_brief(), 0, max_attempts=1, phase="retry", seed_urls=seeds)

    assert [c.url for c in fetch.calls][:2] == [
        "https://blocked.example/a",
        "https://ok1.example/p",
    ]
    assert "ok1.example/p" in llm.prompts[0]  # already read before the first model call
    assert len(llm.prompts) == 3 and result.status == "complete"  # the seed cost no model call
    assert "s1: seeded fetch ok" in ctx.trace.notes


def test_retry_feedback_carries_failed_urls_reasons_and_a_rephrase_hint():
    review = Review(sections={"s1": SectionReview(section_id="s1", verdict="fail")})
    review.sections["s1"].issues = []
    previous = WorkerResult(
        section_id="s1",
        index=0,
        status="incomplete",
        failed_fetches=[{"url": "https://pubs.example/a", "reason": "HTTP 403"}],
        queries=["carbon capture cost"],
    )

    text = feedback_text(review, "s1", [], previous)

    assert "https://pubs.example/a (HTTP 403)" in text and "do not retry" in text
    assert "carbon capture cost" in text and "Rephrase" in text


def test_failure_reason_names_the_cause_briefly():
    assert failure_reason("x answered HTTP 403 twice") == "HTTP 403"
    assert failure_reason("x timed out. Pick another") == "timed out"
    assert failure_reason("???") == "fetch failed"
    assert Generation("t", "m", 1, 1).text == "t"


def test_the_router_summary_lands_in_the_trace(tmp_path):
    from datetime import date

    from fakes import TeamLLM, team_tools
    from reporter.agent import MultiAgent

    class Routed(TeamLLM):
        def summary(self):
            return {"providers": ["gemini", "groq"], "empty_replies": 4}

    agent = MultiAgent(Routed(3), team_tools(), today=date(2026, 10, 30), runs_dir=tmp_path)
    result = agent.run("Topic")

    assert result.trace.llm == {"providers": ["gemini", "groq"], "empty_replies": 4}
