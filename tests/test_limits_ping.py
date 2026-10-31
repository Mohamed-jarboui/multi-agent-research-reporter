"""429 limit types in the trace, baseline trace.llm parity and `reporter ping`. Fakes only."""

import json

import pytest

from fakes import FakeTool, action, page, plan
from reporter import cli
from reporter.agent import BaselineAgent
from reporter.budget import Budget
from reporter.llm import AllProvidersFailedError, AuthError, ServerError
from reporter.llm.base import RateLimitError, error_from_status
from reporter.llm.limits import LIMIT_TYPES, classify_gemini, classify_groq, parse_duration_s
from reporter.llm.ping import PingResult, ping_all
from test_llm_router import FakeProvider, make_llm

# ------------------------------------------------------------------ classification


@pytest.mark.parametrize(
    ("message", "limit", "retry"),
    [
        ("... on tokens per minute (TPM): Limit 8000. try again in 6.5s", "per_minute_tokens", 6.5),
        (
            "... on requests per minute (RPM): Limit 30. try again in 2s.",
            "per_minute_requests",
            2.0,
        ),
        (
            "... on tokens per day (TPD): Limit 1000. try again in 1h2m3.5s",
            "per_day_tokens",
            3723.5,
        ),
        ("... on requests per day (RPD): Limit 1000. try again in 12m", "per_day_requests", 720.0),
        ("Too many requests, slow down", "unknown", None),
    ],
)
def test_groq_message_gives_the_limit_type_and_retry_delay(message, limit, retry):
    assert classify_groq(message) == (limit, retry)


def gemini_body(quota_id: str, delay: str | None = "34s") -> dict:
    metric = "generativelanguage.googleapis.com/x"
    details: list[dict] = [{"violations": [{"quotaMetric": metric, "quotaId": quota_id}]}]
    if delay:
        details.append({"retryDelay": delay})
    return {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": details}}


@pytest.mark.parametrize(
    ("quota_id", "limit"),
    [
        ("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "per_day_requests"),
        ("GenerateContentInputTokensPerModelPerMinute-FreeTier", "per_minute_tokens"),
        ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "per_minute_requests"),
        ("SomethingElse", "quota_exhausted"),
    ],
)
def test_gemini_quota_id_gives_the_limit_type(quota_id, limit):
    assert classify_gemini(gemini_body(quota_id)) == (limit, 34.0)


def test_gemini_without_details_is_unknown_or_quota_exhausted():
    assert classify_gemini(None, "boom") == ("unknown", None)
    assert classify_gemini({}, "You exceeded your current quota") == ("quota_exhausted", None)
    assert classify_gemini("not a dict", "") == ("unknown", None)


def test_durations():
    assert parse_duration_s("250ms") == 0.25 and parse_duration_s("nothing") is None


# --------------------------------------------------------------- router trace and wording


def limited(limit, retry=None, provider="gemini"):
    err = error_from_status(429, "org_01abc https://x.example", provider=provider, limit=limit)
    err.retry_after_s = retry
    return err


def test_failure_records_keep_only_the_limit_enum_and_retry_delay():
    first = FakeProvider("gemini", [limited("per_day_requests", 34.0)] * 3)
    second = FakeProvider("groq", [limited("per_minute_tokens", provider="groq"), "fine"])
    llm, _ = make_llm(first, second)

    assert llm.generate("hi").text == "fine"

    gemini = [f for f in llm.failures if f["provider"] == "gemini"]
    assert all(f["limit"] == "per_day_requests" and f["retry_after_s"] == 34.0 for f in gemini)
    assert llm.summary()["limits"] == {
        "gemini": {"per_day_requests": 3},
        "groq": {"per_minute_tokens": 1},
    }
    assert {f["limit"] for f in llm.failures} <= set(LIMIT_TYPES)
    assert "org_01abc" not in json.dumps(llm.failures) + json.dumps(llm.summary())


def test_a_429_without_a_known_limit_is_unknown_and_other_errors_have_none():
    first = FakeProvider("gemini", [RateLimitError("x", provider="gemini", status=429)] * 3)
    second = FakeProvider("groq", [ServerError("x")] * 3)
    llm, _ = make_llm(first, second)

    with pytest.raises(AllProvidersFailedError) as caught:
        llm.generate("hi")

    message = str(caught.value)
    assert "gemini: RateLimitError 429 x3 (unknown)" in message
    assert message.endswith("groq: ServerError x3)")
    assert all("limit" not in f for f in llm.failures if f["provider"] == "groq")


def test_all_providers_failed_names_the_limit_types():
    first = FakeProvider("gemini", [limited("per_day_tokens")] * 3)
    second = FakeProvider("groq", [limited("per_minute_requests", provider="groq")] * 3)
    llm, _ = make_llm(first, second)

    with pytest.raises(AllProvidersFailedError) as caught:
        llm.generate("hi")

    message = str(caught.value)
    assert "gemini: RateLimitError 429 x3 (per_day_tokens)" in message
    assert "groq: RateLimitError 429 x3 (per_minute_requests)" in message


# ------------------------------------------------------------------- baseline parity


def test_baseline_trace_has_the_same_llm_summary_as_the_multi_path(tmp_path):
    url = page(1).url
    provider = FakeProvider(
        "gemini",
        [
            limited("per_minute_tokens", 5.0),
            plan("Q?"),
            "",
            action("fetch_page", [1], url=url),
            action("finish"),
            action("finish"),
            action("finish"),
            "# T\n\nA fact [1].",
        ],
    )
    llm, _ = make_llm(provider)
    tools = {"fetch_page": FakeTool("fetch_page", {url: page(1)})}
    agent = BaselineAgent(llm, tools, Budget(max_steps=8), runs_dir=tmp_path)

    agent.run("T")

    saved = json.loads(agent.last_trace_path.read_text(encoding="utf-8"))["llm"]
    assert saved["providers"] == ["gemini"]
    assert saved["failures"] == {"gemini": {"RateLimitError 429": 1, "EmptyReplyError": 1}}
    assert saved["limits"] == {"gemini": {"per_minute_tokens": 1}}
    assert saved["empty_replies"] == 1
    assert saved["calls"]["gemini"] >= 6


# -------------------------------------------------------------------------- ping


class PingFake:
    def __init__(self, name, outcome):
        self.name, self.outcome, self.calls, self.kwargs = name, outcome, 0, {}

    def generate(self, prompt, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


def test_ping_makes_one_small_call_per_provider_and_reports_limit_and_latency():
    ticks = iter([0.0, 1.5, 2.0, 2.25, 3.0, 3.1, 4.0, 4.5])
    providers = {
        "gemini": PingFake("gemini", "ok"),
        "groq": PingFake("groq", limited("per_day_requests", 34.0, provider="groq")),
        "other": PingFake("other", AuthError("sk-secret-key", provider="other", status=401)),
        "odd": PingFake("odd", RuntimeError("raw secret text")),
        "none": None,
    }
    results = ping_all(providers, clock=lambda: next(ticks))

    assert [p.calls for p in providers.values() if p] == [1, 1, 1, 1]
    assert providers["gemini"].kwargs["max_output_tokens"] <= 16
    lines = [r.line() for r in results]
    assert lines[0] == "gemini: ok (1.5s)"
    assert (
        lines[1] == "groq: failed RateLimitError 429 limit=per_day_requests retry_after=34s (0.2s)"
    )
    assert lines[2].startswith("other: failed AuthError 401 ") and "limit" not in lines[2]
    assert lines[3].startswith("odd: failed RuntimeError")
    assert lines[4] == "none: not configured"
    assert "secret" not in "\n".join(lines)


@pytest.mark.parametrize(
    ("outcomes", "code"), [(["ok", ServerError("x")], 0), ([ServerError("x"), None], 1)]
)
def test_ping_command_exit_code(outcomes, code, capsys):
    names = ("gemini", "groq")
    providers = {
        n: None if o is None else PingFake(n, o) for n, o in zip(names, outcomes, strict=True)
    }
    assert cli.run_ping(providers) == code
    assert len(capsys.readouterr().out.strip().splitlines()) == 2


def test_ping_subcommand_uses_the_built_providers(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "reporter.llm.ping.build_providers", lambda: {"gemini": PingFake("gemini", "ok")}
    )
    monkeypatch.chdir(tmp_path)  # no .env here
    assert cli.main(["ping"]) == 0
    assert capsys.readouterr().out.startswith("gemini: ok")


def test_a_pingresult_line_has_only_class_and_status():
    line = PingResult("g", "failed", "ServerError", None, None, 0.0).line()
    assert line == "g: failed ServerError (0.0s)"
