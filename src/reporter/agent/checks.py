"""Deterministic checks (no model): citation numbers, fetched URLs, empty sections."""

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from reporter.agent.parsing import citation_numbers
from reporter.agent.schemas import Issue, MergedResearch, SectionOut, SectionReview
from reporter.trace import SourceRecord, Trace

_URL = re.compile(r"https?://[^\s)\]>\"'<]+")


def normalize_url(url: str) -> str:
    """Key for 'the same page': lowercase scheme and host, no fragment, no trailing slash."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}{query}"


class CitationCheck(BaseModel):
    cited: list[int] = Field(default_factory=list)
    invalid_citations: list[int] = Field(default_factory=list)  # not in the sources list
    uncited_sources: list[int] = Field(default_factory=list)  # in the list, never cited
    unfetched_urls: list[str] = Field(default_factory=list)  # cited or written, never fetched

    @property
    def ok(self) -> bool:
        return not (self.invalid_citations or self.uncited_sources or self.unfetched_urls)


def check_citations(
    text: str, sources: list[SourceRecord], fetched_urls: Iterable[str]
) -> CitationCheck:
    """Check a report body (without its Sources section) against the global sources list.

    `unfetched_urls` holds the URL of every cited source, and every URL written in the text, that
    no fetch of this run produced.
    """
    fetched = {normalize_url(u) for u in fetched_urls}
    by_id = {s.id: s for s in sources}
    cited = citation_numbers(text)
    unfetched: list[str] = []
    candidates = [by_id[n].url for n in cited if n in by_id] + [
        m.group(0).rstrip(".,;:") for m in _URL.finditer(text)
    ]
    for url in candidates:
        if normalize_url(url) not in fetched and url not in unfetched:
            unfetched.append(url)
    return CitationCheck(
        cited=cited,
        invalid_citations=[n for n in cited if n not in by_id],
        uncited_sources=[s.id for s in sources if s.id not in cited],
        unfetched_urls=unfetched,
    )


def fetched_urls(trace: Trace, extra: Iterable[str] = ()) -> set[str]:
    """URLs fetched in this run: successful fetch_page calls in the trace plus fetched pages."""
    urls = {*extra}
    for step in trace.steps:
        call = step.tool_call
        if call is not None and call.ok and call.name == "fetch_page":
            urls.add(str(call.input.get("url", "")))
    return {u for u in urls if u}


def section_problem(section: SectionOut) -> str | None:
    """Why a section has nothing to write from, or None. No LLM call is needed for these."""
    if section.n_pages == 0:
        return f"no page was fetched ({section.error or section.stop_reason or 'unknown reason'})"
    if not section.findings:
        return "no finding with a valid source"
    return None


def deterministic_reviews(merged: MergedResearch) -> dict[str, SectionReview]:
    """A `fail` review for every section with no sources or no findings."""
    reviews: dict[str, SectionReview] = {}
    for section in merged.sections:
        if (problem := section_problem(section)) is not None:
            reviews[section.brief.id] = SectionReview(
                section_id=section.brief.id,
                verdict="fail",
                issues=[Issue(rubric=1, issue=problem, origin="deterministic")],
            )
    return reviews


# ------------------------------------------------------- text that may reach the report

_REASONS = [
    (("ratelimit", "rate limit", "429", "too many requests"), "rate limited"),
    (("budget_steps",), "step budget reached"),
    (("budget_seconds", "worker_time"), "time limit reached"),
    (("budget_tokens", "worker_tokens"), "token budget reached"),
    (("budget_cost",), "cost limit reached"),
    (("max_steps", "max_pages"), "per-section step limit reached"),
    (("unparseable", "unusable", "empty", "no finding"), "model output unusable"),
    (("providererror", "allprovidersfailed", "model call failed", "timeout"), "model unavailable"),
]
_BRACKET_TOKEN = re.compile(r"\[[A-Za-z_]+\]")
_ANY_URL = re.compile(r"https?://\S+|www\.\S+")


def plain_reason(*texts: str | None) -> str:
    """A short plain reason for an error or stop reason. Raw exception text, URLs, ids and
    internal tokens never reach a report; the detail stays in the trace."""
    joined = " ".join(t for t in texts if t).lower()
    for needles, reason in _REASONS:
        if any(n in joined for n in needles):
            return reason
    return "research did not complete" if joined else "unknown reason"


def section_reason(section: SectionOut) -> str:
    """Why a section has no findings, from its pages, stop reason and error. Plain words only.

    "blocked" is claimed only when a fetch really returned HTTP 403 or 429.
    """
    joined = f"{section.error or ''} {section.stop_reason}".lower()
    kinds = {"step": ("budget_steps",), "token": ("budget_tokens", "worker_tokens")}
    kinds["cost"] = ("budget_cost",)
    kinds["time"] = ("budget_seconds", "worker_time")
    budget = next((k for k, needles in kinds.items() if any(n in joined for n in needles)), "")
    if section.n_pages == 0:
        if section.blocked_fetches:
            return "no page could be fetched; the sites refused the requests (HTTP 403 or 429)"
        if budget:
            return f"no page could be fetched before the {budget} budget was reached"
        if any(n in joined for n in _REASONS[0][0]):
            return "no page could be fetched; the model was rate limited"
        return "no page could be fetched"
    if budget:
        return f"pages were fetched but the {budget} budget was reached before findings were made"
    return (
        "pages were fetched but no findings could be extracted "
        "(model unavailable or unusable replies)"
    )


def clean_text(text: str, limit: int = 220) -> str:
    """Model-written text for a report line: no URLs or [internal_tokens], one line, shortened."""
    text = _ANY_URL.sub("", text)
    text = _BRACKET_TOKEN.sub("", text)
    text = " ".join(text.split()).strip(" .;:")
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
