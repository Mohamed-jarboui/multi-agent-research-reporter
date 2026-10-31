"""The opt-in gateway provider: wire mapping, error mapping, secrecy, ping and chain rules."""

import json
import logging

import httpx
import pytest

from bench_fakes import make_trace
from reporter.bench.aggregate import aggregate
from reporter.bench.runner import PlanItem, build_record
from reporter.llm import (
    AllProvidersFailedError,
    AuthError,
    FallbackLLM,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from reporter.llm.factory import build_default_llm
from reporter.llm.openai_compat import (
    ModelUnavailableError,
    OpenAICompatProvider,
    ProviderConfigError,
)
from reporter.llm.ping import build_providers, ping_provider
from test_bench_score import topic

KEY = "sk-secret-key-123"
LEAKY = {"error": {"message": "echo of body: sk-secret-key-123 PROMPT-TEXT"}}


def provider(handler, **kwargs) -> OpenAICompatProvider:
    return OpenAICompatProvider(KEY, model="m-1", transport=httpx.MockTransport(handler), **kwargs)


def reply(content="hello", usage=True):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = {"prompt_tokens": 7, "completion_tokens": 3}
    return httpx.Response(200, json=body)


def test_success_sends_the_documented_request_and_reads_usage():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return reply()

    gen = provider(handler, base_url="https://gw.example/v1/").generate(
        "hi", system="sys", temperature=0.2, max_output_tokens=9
    )
    assert seen["url"] == "https://gw.example/v1/chat/completions"
    assert seen["auth"] == f"Bearer {KEY}"
    assert seen["body"] == {
        "model": "m-1",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 9,
    }
    assert (gen.text, gen.model, gen.provider) == ("hello", "m-1", "agentrouter")
    assert (gen.input_tokens, gen.output_tokens) == (7, 3)


def test_missing_usage_gives_none_not_a_guess():
    gen = provider(lambda r: reply(usage=False)).generate("hi")
    assert gen.input_tokens is None and gen.output_tokens is None


def test_default_base_url_and_env_settings(monkeypatch):
    monkeypatch.setenv("AGENTROUTER_API_KEY", KEY)
    monkeypatch.setenv("AGENTROUTER_MODEL", "env-model")
    monkeypatch.delenv("AGENTROUTER_BASE_URL", raising=False)
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return reply()

    p = OpenAICompatProvider(transport=httpx.MockTransport(handler))
    assert p.generate("hi").model == "env-model"
    assert urls == ["https://agentrouter.org/v1/chat/completions"]


def test_missing_model_is_a_clear_config_error_without_the_key(monkeypatch):
    monkeypatch.delenv("AGENTROUTER_MODEL", raising=False)
    with pytest.raises(ProviderConfigError) as info:
        OpenAICompatProvider(KEY)
    assert "AGENTROUTER_MODEL" in str(info.value) and KEY not in str(info.value)


@pytest.mark.parametrize(
    ("status", "body", "kind"),
    [
        (401, LEAKY, AuthError),
        (403, LEAKY, AuthError),
        (404, LEAKY, ModelUnavailableError),
        (503, {"error": {"message": "No available channel for model x"}}, ModelUnavailableError),
        (429, LEAKY, RateLimitError),
        (500, LEAKY, ServerError),
        (503, LEAKY, ServerError),
    ],
)
def test_status_mapping_never_leaks_key_or_bodies(status, body, kind, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(kind) as info:
        provider(lambda r: httpx.Response(status, json=body)).generate("PROMPT-TEXT")
    error = info.value
    assert error.status == status
    for text in (str(error), error.short(), repr(error), caplog.text):
        assert KEY not in text and "PROMPT-TEXT" not in text and "echo" not in text
    if kind is ModelUnavailableError:
        assert "check AGENTROUTER_MODEL" in str(error)
    assert isinstance(error, NotFoundError) == (kind is ModelUnavailableError)


def test_429_limit_and_retry_after():
    body = {"error": {"message": "limit on tokens per minute (TPM) reached"}}
    p = provider(lambda r: httpx.Response(429, json=body, headers={"Retry-After": "4"}))
    with pytest.raises(RateLimitError) as info:
        p.generate("hi")
    assert info.value.limit == "per_minute_tokens" and info.value.retry_after_s == 4.0
    with pytest.raises(RateLimitError) as info:
        provider(lambda r: httpx.Response(429, text="slow down")).generate("hi")
    assert info.value.limit == "unknown"


def test_connection_failure_is_a_server_error_without_the_url():
    def boom(request):
        raise httpx.ConnectError("cannot reach https://x with sk-secret-key-123")

    with pytest.raises(ServerError) as info:
        provider(boom).generate("hi")
    assert KEY not in str(info.value) and "https" not in str(info.value)


def test_empty_content_is_an_empty_reply_in_the_router():
    llm = FallbackLLM([provider(lambda r: reply(""))], max_attempts=1, sleep=lambda s: None)
    with pytest.raises(AllProvidersFailedError):
        llm.generate("hi")
    assert llm.empty_replies == 1


def test_unparseable_reply_is_a_server_error():
    with pytest.raises(ServerError):
        provider(lambda r: httpx.Response(200, text="<html>")).generate("hi")


# ---------------------------------------------------------------- ping lines


def test_ping_line_formats():
    ok = ping_provider("agentrouter", provider(lambda r: reply()), clock=iter([0.0, 1.2]).__next__)
    assert ok.line() == "agentrouter: ok (1.2s)"
    bad = ping_provider(
        "agentrouter",
        provider(lambda r: httpx.Response(401, json=LEAKY)),
        clock=iter([0.0, 0.5]).__next__,
    )
    assert bad.line() == "agentrouter: failed AuthError 401 hint=unknown (0.5s)"
    assert ping_provider("agentrouter", None).line() == "agentrouter: not configured"


HINT_CASES = [
    ("invalid_token", 401, "Invalid token"),
    ("invalid_token", 401, "Incorrect API key provided"),
    ("invalid_token", 401, "token not found"),
    ("invalid_token", 401, "无效的令牌"),
    ("invalid_token", 401, "未提供令牌"),
    ("token_disabled_or_expired", 401, "This token has expired"),
    ("token_disabled_or_expired", 403, "该令牌已被禁用"),
    ("no_credit", 402, "Insufficient quota"),
    ("no_credit", 403, "user quota is not enough: 余额不足"),
    ("no_credit", 403, "额度不足"),
    ("client_not_allowed", 403, "Unauthorized client detected"),
    ("client_not_allowed", 403, "不支持的客户端"),
    ("client_not_allowed", 403, "This key is only for Claude Code"),
    ("model_not_allowed", 403, "No access to model x"),
    ("model_not_allowed", 403, "该分组无权访问此模型"),
    ("model_unavailable", 503, "No available channel for model x"),
    ("model_unavailable", 404, "model not found"),
    ("model_unavailable", 503, "无可用渠道"),
    ("rate_limited", 429, "Too many requests"),
    ("rate_limited", 429, "请求过于频繁"),
    ("ip_blocked", 403, "Your IP is not allowed"),
    ("ip_blocked", 403, "禁止访问"),
    ("ip_blocked", 403, "Forbidden"),
    ("unknown", 500, "something odd happened"),
]


def hint_of(status, body=None, **kwargs):
    def handler(request):
        if body is None:
            return httpx.Response(status, **kwargs)
        return httpx.Response(status, json=body)

    with pytest.raises(Exception) as info:
        provider(handler).generate("hi")
    return info.value.hint


@pytest.mark.parametrize(("token", "status", "text"), HINT_CASES)
def test_hint_categories_from_json_error_bodies(token, status, text):
    assert hint_of(status, {"error": {"message": text}}) == token


def test_hint_reads_code_and_type_and_flat_message():
    assert hint_of(403, {"error": {"message": "x", "code": "insufficient_quota"}}) == "no_credit"
    assert hint_of(403, {"error": {"message": "x", "type": "Insufficient Quota"}}) == "no_credit"
    assert hint_of(401, {"message": "Invalid API key"}) == "invalid_token"


@pytest.mark.parametrize(
    ("token", "text"),
    [
        ("invalid_token", "invalid token"),
        ("no_credit", "额度不足"),
        ("unknown", "<html>oops</html>"),
    ],
)
def test_hint_from_plain_text_bodies(token, text):
    assert hint_of(401, text=text) == token


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_hint_empty_or_odd_body_is_unknown(status):
    assert hint_of(status, text="") == "unknown"
    assert hint_of(status, ["a list"]) == "unknown"


def test_hint_never_carries_key_or_prompt_text_to_the_ping_line():
    prompt = "PROMPT-TEXT " * 200
    body = {"error": {"message": f"Invalid token {KEY} {prompt}", "code": KEY}}
    clock = iter([0.0, 0.5]).__next__
    result = ping_provider(
        "agentrouter", provider(lambda r: httpx.Response(401, json=body)), clock=clock
    )
    assert result.hint == "invalid_token"
    assert result.line() == "agentrouter: failed AuthError 401 hint=invalid_token (0.5s)"
    with pytest.raises(AuthError) as info:
        provider(lambda r: httpx.Response(401, json=body)).generate("hi")
    error = info.value
    assert error.hint in HINTS
    for text in (str(error), repr(error), error.short(), result.line()):
        assert KEY not in text and "PROMPT" not in text


HINTS = {t for t, _, _ in HINT_CASES} | {"unknown"}


def test_hint_is_always_in_the_closed_set_and_absent_on_success():
    assert {t for t, _, _ in HINT_CASES} <= HINTS
    ok = ping_provider("agentrouter", provider(lambda r: reply()), clock=iter([0.0, 1.0]).__next__)
    assert ok.hint is None and "hint" not in ok.line()


def clear_env(monkeypatch):
    for name in (
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "AGENTROUTER_API_KEY",
        "AGENTROUTER_MODEL",
        "AGENTROUTER_BASE_URL",
        "REPORTER_EXTRA_PROVIDERS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_build_providers_pings_the_gateway_only_when_its_key_is_set(monkeypatch):
    clear_env(monkeypatch)
    assert build_providers()["agentrouter"] is None
    monkeypatch.setenv("AGENTROUTER_API_KEY", KEY)
    result = ping_provider("agentrouter", build_providers()["agentrouter"])  # model missing
    assert result.status == "failed" and KEY not in result.line()
    monkeypatch.setenv("AGENTROUTER_MODEL", "m-1")
    assert build_providers()["agentrouter"].name == "agentrouter"


# ---------------------------------------------------------------- chain opt-in


def chain_names(monkeypatch, **env) -> list[str]:
    clear_env(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "g-key")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return [p.name for p in build_default_llm().providers]


def test_key_alone_does_not_put_the_gateway_in_the_chain(monkeypatch):
    names = chain_names(monkeypatch, AGENTROUTER_API_KEY=KEY, AGENTROUTER_MODEL="m-1")
    assert names == ["groq"]


def test_opt_in_puts_it_last(monkeypatch):
    names = chain_names(
        monkeypatch,
        AGENTROUTER_API_KEY=KEY,
        AGENTROUTER_MODEL="m-1",
        REPORTER_EXTRA_PROVIDERS="other, AgentRouter",
    )
    assert names == ["groq", "agentrouter"]


def test_opt_in_without_model_leaves_it_out_and_says_why(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        names = chain_names(
            monkeypatch, AGENTROUTER_API_KEY=KEY, REPORTER_EXTRA_PROVIDERS="agentrouter"
        )
    assert names == ["groq"]
    assert "AGENTROUTER_MODEL is not set" in caplog.text and KEY not in caplog.text


# ---------------------------------------------------------------- results


def test_gateway_used_flag_and_aggregate_note():
    item = PlanItem(topic("a"), "multi", 1)
    plain = build_record(item, make_trace("multi"), "", None)
    assert plain.gateway_used is False
    trace = make_trace("multi", providers=("groq", "agentrouter"))
    trace.llm["calls"] = {"groq": 2, "agentrouter": 1}
    used = build_record(item, trace, "", None)
    assert used.gateway_used is True
    notes = aggregate([topic("a")], {used.key: used}, [], 1, None)["notes"]
    assert any("agentrouter" in n for n in notes)
    clean = aggregate([topic("a")], {plain.key: plain}, [], 1, None)["notes"]
    assert not any("agentrouter" in n for n in clean)


# ---------------------------------------------------------------- benchmark preflight


def test_build_providers_without_gateway_never_builds_it(monkeypatch):
    clear_env(monkeypatch)
    monkeypatch.setenv("AGENTROUTER_API_KEY", KEY)
    monkeypatch.setenv("AGENTROUTER_MODEL", "m-1")
    assert "agentrouter" not in build_providers(gateway=False)
    assert "agentrouter" in build_providers()  # `reporter ping` still tests it


def test_preflight_ignores_the_gateway_unless_opted_in(monkeypatch):
    from reporter.bench.cli import main as bench_main  # noqa: F401  (import check only)
    from reporter.llm.factory import gateway_enabled

    clear_env(monkeypatch)
    monkeypatch.setenv("AGENTROUTER_API_KEY", KEY)
    monkeypatch.setenv("AGENTROUTER_MODEL", "m-1")
    assert gateway_enabled() is False
    assert "agentrouter" not in build_providers(gateway=gateway_enabled())
    monkeypatch.setenv("REPORTER_EXTRA_PROVIDERS", "agentrouter")
    assert gateway_enabled() is True
    assert "agentrouter" in build_providers(gateway=gateway_enabled())


def test_ping_line_names_the_missing_variable_but_no_value(monkeypatch):
    from reporter.llm.ping import ping_provider

    monkeypatch.setenv("AGENTROUTER_API_KEY", "sk-not-a-real-key-123456")
    monkeypatch.delenv("AGENTROUTER_MODEL", raising=False)
    try:
        OpenAICompatProvider()
    except ProviderConfigError as exc:
        misconfigured = exc
    else:  # pragma: no cover
        raise AssertionError("expected a config error")

    class Broken:
        name = "agentrouter"

        def generate(self, *args, **kwargs):
            raise misconfigured

    line = ping_provider("agentrouter", Broken()).line()
    assert "AGENTROUTER_MODEL is not set" in line
    assert "sk-not-a-real-key" not in line
