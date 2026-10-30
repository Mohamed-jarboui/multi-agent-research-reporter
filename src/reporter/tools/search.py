"""web_search: Tavily when TAVILY_API_KEY is set, DuckDuckGo's HTML endpoint otherwise."""

from __future__ import annotations

import os
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from pydantic import BaseModel, Field

from reporter.tools.base import ToolError, ToolOutput

TAVILY_URL = "https://api.tavily.com/search"
DDG_URL = "https://html.duckduckgo.com/html/"
USER_AGENT = "Mozilla/5.0 (compatible; reporter-research-agent/0.1)"


class SearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="search query")
    max_results: int = Field(default=5, ge=1, le=8, description="how many results, 1-8")


class SearchHit(BaseModel):
    title: str
    url: str
    snippet: str


class WebSearch:
    name: ClassVar[str] = "web_search"
    description: ClassVar[str] = (
        "Search the web. Returns titles, URLs and short snippets; use fetch_page to read one."
    )
    input_model: ClassVar[type[BaseModel]] = SearchInput

    def __init__(self, client: httpx.Client | None = None, api_key: str | None = None) -> None:
        self._client = client or httpx.Client(timeout=15.0, headers={"User-Agent": USER_AGENT})
        self._api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY")

    def run(self, args: SearchInput) -> ToolOutput:
        hits: list[SearchHit] | None = None
        note = ""
        if self._api_key:
            try:
                hits = self._tavily(args)
            except ToolError as exc:
                note = f"(Tavily failed: {exc}; used DuckDuckGo instead)\n"
        if hits is None:
            hits = self._duckduckgo(args)
        if not hits:
            raise ToolError(
                f"no results for {args.query!r}. Try fewer or different keywords, or a synonym."
            )
        lines = [f"{i}. {h.title}\n   {h.url}\n   {h.snippet}" for i, h in enumerate(hits, 1)]
        return ToolOutput(observation=note + "\n".join(lines))

    def _tavily(self, args: SearchInput) -> list[SearchHit]:
        try:
            response = self._client.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"query": args.query, "max_results": args.max_results},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
        except httpx.HTTPStatusError as exc:
            raise ToolError(f"HTTP {exc.response.status_code}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(type(exc).__name__) from exc
        return [
            SearchHit(
                title=r.get("title") or r["url"], url=r["url"], snippet=_clip(r.get("content", ""))
            )
            for r in results
            if r.get("url")
        ][: args.max_results]

    def _duckduckgo(self, args: SearchInput) -> list[SearchHit]:
        try:
            response = self._client.post(DDG_URL, data={"q": args.query})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ToolError(f"DuckDuckGo answered HTTP {exc.response.status_code}") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"DuckDuckGo is unreachable ({type(exc).__name__})") from exc
        return parse_duckduckgo(response.text)[: args.max_results]


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _DDGParser(HTMLParser):
    """Collects (title, url, snippet) from DuckDuckGo's HTML results page."""

    def __init__(self) -> None:
        super().__init__()
        self.hits: list[SearchHit] = []
        self._field: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result__a" in classes:
            self._flush()
            self._url = _unwrap(attributes.get("href") or "")
            self._field = "title"
        elif tag in ("a", "div") and "result__snippet" in classes:
            self._field = "snippet"

    def handle_endtag(self, tag: str) -> None:
        if tag in ("a", "div"):
            self._field = None

    def handle_data(self, data: str) -> None:
        if self._field == "title":
            self._title.append(data)
        elif self._field == "snippet":
            self._snippet.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        if self._url and self._title:
            self.hits.append(
                SearchHit(
                    title=" ".join("".join(self._title).split()),
                    url=self._url,
                    snippet=_clip("".join(self._snippet)),
                )
            )
        self._url, self._title, self._snippet = "", [], []


def _unwrap(href: str) -> str:
    """DuckDuckGo wraps result links as //duckduckgo.com/l/?uddg=<encoded target>."""
    if "uddg=" in href:
        target = parse_qs(urlparse(href).query).get("uddg")
        if target:
            return unquote(target[0])
    return "https:" + href if href.startswith("//") else href


def parse_duckduckgo(html: str) -> list[SearchHit]:
    parser = _DDGParser()
    parser.feed(html)
    parser.close()
    return [h for h in parser.hits if h.url.startswith("http")]
