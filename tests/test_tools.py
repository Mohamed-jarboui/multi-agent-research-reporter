"""Tools against a mocked HTTP transport: no network."""

import json

import httpx
import pytest

from reporter.tools import FetchPage, WebSearch, call_tool
from reporter.tools.search import parse_duckduckgo

DDG_HTML = """
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa&rut=x">
    First <b>result</b></a>
  <a class="result__snippet" href="x">Snippet   one</a>
</div>
<div class="result">
  <a class="result__a" href="https://example.org/b">Second</a>
  <div class="result__snippet">Snippet two</div>
</div>
"""

ARTICLE = (
    "<html><head><title> Small models </title><style>p{}</style></head><body>"
    "<nav>menu menu</nav><script>var x = 1;</script>"
    "<article><h1>Heading</h1><p>" + "Useful sentence about models. " * 20 + "</p></article>"
    "<footer>copyright</footer></body></html>"
)
HTML = {"content-type": "text/html; charset=utf-8"}


def client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_parse_duckduckgo_unwraps_redirect_and_collects_snippets():
    hits = parse_duckduckgo(DDG_HTML)
    assert [h.url for h in hits] == ["https://example.org/a", "https://example.org/b"]
    assert hits[0].title == "First result"
    assert hits[0].snippet == "Snippet one"
    assert hits[1].snippet == "Snippet two"


def test_search_uses_tavily_when_key_is_set():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        results = [{"title": "T", "url": "https://t.example", "content": "about T"}]
        return httpx.Response(200, json={"results": results})

    tool = WebSearch(client(handler), api_key="k")
    out = call_tool(tool, {"query": "small language models", "max_results": 3})
    assert not out.error
    assert "https://t.example" in out.observation
    assert seen["url"].startswith("https://api.tavily.com/")
    assert seen["auth"] == "Bearer k"
    assert seen["body"]["max_results"] == 3


def test_search_falls_back_to_duckduckgo_when_tavily_fails():
    def handler(request: httpx.Request) -> httpx.Response:
        if "tavily" in request.url.host:
            return httpx.Response(432, json={"detail": "quota"})
        return httpx.Response(200, text=DDG_HTML)

    out = call_tool(WebSearch(client(handler), api_key="k"), {"query": "carbon capture"})
    assert not out.error
    assert out.observation.startswith("(Tavily failed: HTTP 432")
    assert "https://example.org/a" in out.observation


def test_search_without_key_goes_straight_to_duckduckgo(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    hosts = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(200, text=DDG_HTML)

    out = call_tool(WebSearch(client(handler)), {"query": "carbon capture"})
    assert not out.error
    assert hosts == ["html.duckduckgo.com"]


def test_search_with_no_results_is_an_informative_error():
    tool = WebSearch(client(lambda r: httpx.Response(200, text="<html></html>")), api_key="")
    out = call_tool(tool, {"query": "zzzz qqqq"})
    assert out.error
    assert "no results" in out.observation and "different keywords" in out.observation


def test_search_network_failure_is_an_error_observation_not_an_exception():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    out = call_tool(WebSearch(client(handler), api_key=""), {"query": "anything"})
    assert out.error and "unreachable" in out.observation


def test_input_validation_names_the_bad_field():
    tool = WebSearch(client(lambda r: httpx.Response(200)), api_key="")
    out = call_tool(tool, {"query": "ok", "max_results": 99})
    assert out.error and "max_results" in out.observation


def test_fetch_extracts_readable_text_and_drops_boilerplate():
    tool = FetchPage(client(lambda r: httpx.Response(200, text=ARTICLE, headers=HTML)))
    out = call_tool(tool, {"url": "https://example.org/x"})
    assert not out.error
    page = out.pages[0]
    assert page.title == "Small models"
    assert "Useful sentence" in page.text and "Heading" in page.text
    assert "menu" not in page.text and "var x" not in page.text and "copyright" not in page.text


def test_fetch_bounds_the_text_length():
    http = client(lambda r: httpx.Response(200, text=ARTICLE, headers=HTML))
    out = call_tool(FetchPage(http, max_chars=300), {"url": "https://example.org/x"})
    assert len(out.pages[0].text) == 300
    assert out.pages[0].truncated
    assert out.observation.endswith("[truncated]")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(404, text="nope"), "HTTP 404"),
        (httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}), "PDF"),
        (
            httpx.Response(200, text="<html><body>hi</body></html>", headers=HTML),
            "almost no readable text",
        ),
    ],
)
def test_fetch_failures_say_what_to_do(response, message):
    out = call_tool(FetchPage(client(lambda r: response)), {"url": "https://example.org/x"})
    assert out.error and message in out.observation
    assert "Pick another source" in out.observation


def test_fetch_timeout_is_an_error_observation():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    out = call_tool(FetchPage(client(handler)), {"url": "https://example.org/x"})
    assert out.error and "timed out" in out.observation


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.org/x",
        "not a url",
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://192.168.1.5/admin",
        "file:///etc/passwd",
    ],
)
def test_fetch_rejects_bad_and_local_urls_without_any_request(url):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    out = call_tool(FetchPage(client(handler)), {"url": url})
    assert out.error and "invalid input" in out.observation
