"""fetch_page: download a page and reduce it to readable, length-bounded text."""

from __future__ import annotations

import ipaddress
import re
from html.parser import HTMLParser
from typing import ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field, field_validator

from reporter.tools.base import Page, ToolError, ToolOutput
from reporter.tools.search import USER_AGENT

MAX_DOWNLOAD_BYTES = 1_500_000  # stop reading a response after this many bytes
MAX_PAGE_CHARS = 8_000  # readable text kept per page
OBSERVATION_CHARS = 3_500  # what the model sees in its next step
MIN_TEXT_CHARS = 200  # below this a page is treated as empty (JavaScript shell, bot wall)


class FetchInput(BaseModel):
    url: str = Field(description="absolute http(s) URL, ideally taken from a search result")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("must be an absolute http:// or https:// URL")
        if _is_local(parsed.hostname):
            raise ValueError("local and private addresses are not allowed")
        return value


def _is_local(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg", "iframe"}
_BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section"}
_BLOCK |= {"article", "table", "blockquote", "pre"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in _SKIP:
            self._skip_depth += 1
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in _SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in _BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = (re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n"))
        return "\n".join(line for line in lines if line)


def extract_text(html: str) -> tuple[str, str]:
    """Return (title, readable text) for an HTML document."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), parser.text()


class FetchPage:
    name: ClassVar[str] = "fetch_page"
    description: ClassVar[str] = (
        "Download one web page and return its readable text (truncated). "
        "Only pages read with this tool can be cited."
    )
    input_model: ClassVar[type[BaseModel]] = FetchInput

    def __init__(self, client: httpx.Client | None = None, max_chars: int = MAX_PAGE_CHARS) -> None:
        self._client = client or httpx.Client(
            timeout=15.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
        self._max_chars = max_chars

    def run(self, args: FetchInput) -> ToolOutput:
        body, content_type = self._download(args.url)
        if "html" not in content_type and not content_type.startswith("text/"):
            raise ToolError(
                f"{args.url} is {content_type or 'an unknown type'}, not text or HTML "
                "(PDFs and binary files are not supported). Pick another source."
            )
        if "html" in content_type:
            title, text = extract_text(body)
        else:
            title, text = "", body.strip()
        if len(text) < MIN_TEXT_CHARS:
            raise ToolError(
                f"{args.url} has almost no readable text ({len(text)} characters); it may need "
                "JavaScript or block bots. Pick another source."
            )
        page = Page(
            url=args.url,
            title=title or args.url,
            text=text[: self._max_chars],
            truncated=len(text) > self._max_chars,
        )
        cut = " [truncated]" if page.truncated or len(page.text) > OBSERVATION_CHARS else ""
        shown = page.text[:OBSERVATION_CHARS]
        observation = f"Page: {page.title}\nURL: {page.url}\n\n{shown}{cut}"
        return ToolOutput(observation=observation, pages=[page])

    def _download(self, url: str) -> tuple[str, str]:
        try:
            with self._client.stream("GET", url) as response:
                if response.status_code >= 400:
                    raise ToolError(
                        f"{url} answered HTTP {response.status_code}. Pick another source."
                    )
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size >= MAX_DOWNLOAD_BYTES:
                        break
                content_type = response.headers.get("content-type", "").split(";")[0].strip()
                body = b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")
        except httpx.TimeoutException as exc:
            raise ToolError(f"{url} timed out. Pick another source.") from exc
        except httpx.HTTPError as exc:
            raise ToolError(f"{url} could not be fetched ({type(exc).__name__}).") from exc
        return body, content_type.lower()
