"""Section worker: a short ReAct loop over the R0 tools that returns a structured result.

One call to `run_worker` researches one section. It never raises: a failed attempt is retried once
(with a note about what went wrong), and a section that still fails comes back `incomplete`.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from reporter.agent import prompts
from reporter.agent.baseline import Decision
from reporter.agent.llm_call import LLM, LLMGate, generate_charged
from reporter.agent.parsing import extract_json
from reporter.agent.schemas import Confidence, Finding, SectionBrief, WorkerResult
from reporter.budget import BudgetExhausted, BudgetMeter, MultiBudget
from reporter.llm.base import Generation, ProviderError
from reporter.pricing import call_cost_usd
from reporter.tools import Page, Tool, ToolOutput, call_tool, describe
from reporter.trace import StepRecord, ToolCallRecord, Trace, WorkerRecord

THINK_MAX_TOKENS = 2_000
FINAL_MAX_TOKENS = 3_000
FINAL_CHARS_PER_PAGE = 2_500
MAX_FINDINGS = 8
MAX_PARSE_FAILURES = 3
MAX_FINISH_REFUSALS = 1
OLD_OBSERVATION_CHARS = 200
HISTORY_ITEMS = 5
STOP_FRACTION = 0.8  # of the worker's own time and token limits
MIN_RETRY_SECONDS = 30.0  # section time that must be left to start a second attempt
RETRY_TEMPERATURE = 0.7
MAX_ATTEMPTS = 2


@dataclass
class WorkerContext:
    """Everything the workers of one run share. Lives on the agent, not in the graph state."""

    topic: str
    today: str
    llm: LLM
    tools: dict[str, Tool]
    budget: MultiBudget
    meter: BudgetMeter
    gate: LLMGate
    trace: Trace
    clock: Callable[[], float] = time.monotonic
    stop_fraction: float | None = None  # None: the run budget's; retries use a later one
    cost_fn: Callable[[str, int | None, int | None], float | None] = call_cost_usd


@dataclass
class _Attempt:
    """What one attempt has gathered so far (kept even when the attempt fails half way)."""

    pages: dict[str, Page] = field(default_factory=dict)  # url -> page, fetch order
    steps: list[StepRecord] = field(default_factory=list)
    bad_urls: list[str] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)  # search-result URLs not yet tried
    tried: set[str] = field(default_factory=set)  # URLs fetched or attempted
    blocked_hosts: set[str] = field(default_factory=set)  # hosts that answered 403/429
    failures: dict[str, str] = field(default_factory=dict)  # url -> short reason
    queries: list[str] = field(default_factory=list)

    def add_candidates(self, urls: list[str]) -> None:
        for url in urls:
            if url not in self.candidates and self.usable(url):
                self.candidates.append(url)

    def usable(self, url: str) -> bool:
        """Worth fetching: not a PDF, not on a host that blocked us, not tried or read already."""
        return (
            not _is_pdf(url)
            and _host(url) not in self.blocked_hosts
            and url not in self.tried
            and url not in self.pages
        )

    def block_host(self, url: str) -> None:
        self.blocked_hosts.add(_host(url))
        self.candidates = [c for c in self.candidates if _host(c) != _host(url)]

    def hint(self, n: int = 3) -> str:
        """The next candidate URLs to try, for the model."""
        shown = self.candidates[:n]
        if not shown:
            return "No untried search results are left: search again with different keywords."
        return "Untried search results (fetch the first one next): " + ", ".join(shown)

    def tokens(self) -> int:
        return sum((s.input_tokens or 0) + (s.output_tokens or 0) for s in self.steps)

    def cost(self) -> float:
        return sum(s.cost_usd or 0.0 for s in self.steps)


_URL_LINE = re.compile(r"^\s+(https?://\S+)\s*$", re.MULTILINE)


def _is_pdf(url: str) -> bool:
    return urlsplit(url).path.lower().endswith(".pdf")


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def failure_reason(observation: str) -> str:
    """Short reason for a failed fetch (kept for the retry and the trace; no raw text)."""
    match = re.search(r"HTTP (\d{3})", observation)
    if match:
        return f"HTTP {match.group(1)}"
    for needle, reason in (
        ("timed out", "timed out"),
        ("almost no readable", "no readable text"),
        ("not text or HTML", "pdf or binary"),
        ("already read", "already read"),
    ):
        if needle in observation:
            return reason
    return "fetch failed"


class _AttemptFailed(Exception):
    """The attempt produced nothing usable; the message says why."""


def render_brief(brief: SectionBrief, topic: str, today: str) -> str:
    """The self-contained text a worker is given."""

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {i}" for i in items) or "- (nothing in particular)"

    return prompts.WORKER_BRIEF.format(
        topic=topic,
        today=today,
        title=brief.title,
        goal=brief.goal,
        scope=bullets(brief.scope),
        exclude=bullets(brief.exclude),
        expected_output=brief.expected_output,
    )


def parse_findings(
    data: dict[str, Any], n_pages: int
) -> tuple[list[Finding], Confidence, list[str]]:
    """Validate the finalize reply. A source number counts only if it names a fetched page."""
    raw = data.get("findings")
    if not isinstance(raw, list):
        raise ValueError("'findings' is missing or not a list")
    findings: list[Finding] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        numbers = item.get("sources", [])
        ids = sorted(
            {
                n
                for n in (numbers if isinstance(numbers, list) else [])
                if isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= n_pages
            }
        )
        if claim and ids:
            findings.append(Finding(claim=claim, source_ids=ids))
    findings = findings[:MAX_FINDINGS]
    confidence: Confidence = {"low": "low", "medium": "medium", "high": "high"}.get(
        str(data.get("confidence", "")).lower(), "low"
    )
    if n_pages < 2:
        confidence = "low"
    questions = data.get("open_questions", [])
    open_questions = [
        str(q).strip() for q in (questions if isinstance(questions, list) else []) if str(q).strip()
    ]
    return findings, confidence, open_questions


def run_worker(
    ctx: WorkerContext,
    brief: SectionBrief,
    index: int,
    *,
    feedback: str = "",
    max_attempts: int = MAX_ATTEMPTS,
    phase: str = "research",
    seed_urls: list[str] | None = None,
) -> WorkerResult:
    """Research one section: up to `max_attempts` attempts, then `incomplete`. Never raises.

    `feedback` (the critic's issues, for a retry) is shown to the first attempt as a note.
    """
    section_start = ctx.clock()
    best: WorkerResult | None = None
    note = feedback
    seeds = list(seed_urls or [])
    prior_tokens = 0
    attempts = 0
    for attempt_no in range(1, max_attempts + 1):
        attempts = attempt_no
        result, record, st = _run_attempt(
            ctx, brief, index, attempt_no, note, section_start, prior_tokens, phase, seeds
        )
        ctx.trace.add_worker(record)
        prior_tokens += st.tokens()
        if best is None or result.status == "complete" or len(result.pages) > len(best.pages):
            best = result
        if result.status == "complete" or result.stop_reason == "budget_exhausted":
            break
        time_left = ctx.budget.worker_max_seconds - (ctx.clock() - section_start)
        if ctx.meter.stop_reason(ctx.stop_fraction) or time_left < MIN_RETRY_SECONDS:
            break
        note = f"previous attempt failed: {result.error or result.stop_reason}."
        avoid = [*st.pages, *st.bad_urls]
        if avoid:
            note += " Avoid these URLs: " + ", ".join(avoid) + "."
        note += " Try different search keywords."
        seeds = result.untried_urls
    assert best is not None
    best.attempts = attempts
    return best


def _run_attempt(
    ctx: WorkerContext,
    brief: SectionBrief,
    index: int,
    attempt_no: int,
    note: str,
    section_start: float,
    prior_tokens: int,
    phase: str,
    seeds: list[str],
) -> tuple[WorkerResult, WorkerRecord, _Attempt]:
    st = _Attempt()
    st.add_candidates(seeds)
    started_s = ctx.meter.elapsed_s
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    findings: list[Finding] = []
    confidence: Confidence = "low"
    questions: list[str] = []
    stop_reason = ""
    error: str | None = None
    try:
        stop_reason = _research(ctx, brief, st, note, section_start, prior_tokens)
        if not st.pages:
            raise _AttemptFailed(f"no page was fetched (stopped: {stop_reason})")
        findings, confidence, questions = _finalize(ctx, brief, st, section_start)
        if not findings:
            raise _AttemptFailed("no finding with a valid source")
    except BudgetExhausted as exc:
        stop_reason, error = "budget_exhausted", f"budget exhausted ({exc})"
    except _AttemptFailed as exc:
        error = str(exc)
    except ProviderError as exc:
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # a bug in one worker must not take the whole run down
        error = f"worker crashed: {type(exc).__name__}: {exc}"
    status = "complete" if findings else "incomplete"
    result = WorkerResult(
        section_id=brief.id,
        index=index,
        status=status,
        findings=findings,
        pages=list(st.pages.values()),
        confidence=confidence,
        open_questions=questions,
        attempts=attempt_no,
        stop_reason=stop_reason,
        error=None if status == "complete" else error,
        untried_urls=st.candidates[:8],
        failed_fetches=[{"url": u, "reason": r} for u, r in list(st.failures.items())[:6]],
        queries=st.queries[:4],
    )
    record = WorkerRecord(
        section_id=brief.id,
        attempt=attempt_no,
        phase=phase,
        status=status,
        started_s=round(started_s, 3),
        ended_s=round(ctx.meter.elapsed_s, 3),
        started_at=started_at,
        steps=len(st.steps),
        tokens=st.tokens(),
        cost_usd=st.cost(),
        n_pages=len(st.pages),
        stop_reason=stop_reason,
        error=result.error,
    )
    return result, record, st


# ------------------------------------------------------------------ research loop


def _research(
    ctx: WorkerContext,
    brief: SectionBrief,
    st: _Attempt,
    note: str,
    section_start: float,
    prior_tokens: int,
) -> str:
    """Think/act until a stop rule fires; returns the stop reason."""
    b = ctx.budget
    brief_text = render_brief(brief, ctx.topic, ctx.today)
    retry_note = f"\nNote: {note}\n" if note else ""
    system = prompts.WORKER_THINK_SYSTEM.format(
        tools="\n".join(describe(t) for t in ctx.tools.values()), min_sources=b.min_sources
    )
    parse_failures = 0
    refusals = 0
    _seed_fetch(ctx, brief, st)
    for _ in range(b.worker_max_steps):
        if len(st.pages) >= b.max_pages_per_worker:
            return "max_pages"
        if reason := ctx.meter.stop_reason(ctx.stop_fraction):
            return reason
        if ctx.clock() - section_start >= STOP_FRACTION * b.worker_max_seconds:
            return "worker_time"
        if prior_tokens + st.tokens() >= STOP_FRACTION * b.worker_max_tokens:
            return "worker_tokens"
        ctx.meter.add_step()
        step = StepRecord(index=0, kind="think", worker=brief.id)
        prompt = prompts.WORKER_THINK_PROMPT.format(
            brief=brief_text,
            retry_note=retry_note,
            pages=_render_pages(st),
            history=_render_history(st),
            candidates=f"\n{st.hint()}\n" if st.candidates else "",
            budget=", ".join(f"{k} {v:.0%}" for k, v in ctx.meter.fractions().items()),
        )
        try:
            gen = _generate(ctx, brief, step, prompt, system, THINK_MAX_TOKENS, section_start)
        finally:
            ctx.trace.add_step(step)
            st.steps.append(step)
        try:
            decision = Decision.model_validate(extract_json(gen.text))
        except ValueError as exc:
            parse_failures += 1
            reason = str(exc).splitlines()[0][:120]
            step.note = f"unparseable reply ({reason}); it began: {gen.text[:150]!r}. {step.note}"
            step.note = step.note.strip()
            hint = "ERROR your last reply was not a JSON object; answer with only the JSON object"
            st.history.append(_item("none", {}, hint))
            if parse_failures >= MAX_PARSE_FAILURES:
                return "model_output_unparseable"
            continue
        parse_failures = 0
        step.thought, step.action = decision.thought, decision.action
        if decision.action == "finish":
            if len(st.pages) < b.min_sources and refusals < MAX_FINISH_REFUSALS:
                refusals += 1
                msg = f"ERROR finish refused: fewer than {b.min_sources} pages read. Keep going."
                st.history.append(_item("finish", {}, msg))
                continue
            return "model_finished"
        _act(ctx, step, decision, st)
    return "max_steps"


def _seed_fetch(ctx: WorkerContext, brief: SectionBrief, st: _Attempt) -> None:
    """Start a retry with a fetch of the best untried candidate instead of another search.

    No model call is needed for this; it tries up to 3 candidates until one page is read.
    """
    for url in list(st.candidates[:3]):
        if st.pages or not st.candidates:
            break
        output = _fetch(ctx, url, st)
        outcome = failure_reason(output.observation) if output.error else "ok"
        ctx.trace.notes.append(f"{brief.id}: seeded fetch {outcome}")
        if not output.error:
            st.history.append(_item("fetch_page", {"url": url}, output.observation))


def _fetch(ctx: WorkerContext, url: str, st: _Attempt) -> ToolOutput:
    """fetch_page with the worker's bookkeeping: tried urls, blocked hosts, failure reasons."""
    st.tried.add(url)
    if url in st.candidates:
        st.candidates.remove(url)
    if _is_pdf(url):
        output = ToolOutput(observation="ERROR PDFs are not supported.", error=True)
    elif _host(url) in st.blocked_hosts:
        output = ToolOutput(observation="ERROR this site blocked us earlier.", error=True)
    else:
        output = call_tool(ctx.tools["fetch_page"], {"url": url})
    for page in output.pages:
        st.pages.setdefault(page.url, page)
    if output.error:
        st.bad_urls.append(url)
        reason = failure_reason(output.observation)
        st.failures[url] = reason
        if reason in ("HTTP 403", "HTTP 429"):
            st.block_host(url)
    return output


def _act(ctx: WorkerContext, step: StepRecord, decision: Decision, st: _Attempt) -> None:
    tool = ctx.tools.get(decision.action)
    started = ctx.clock()
    url = str(decision.input.get("url", ""))
    if tool is None:
        known = ", ".join([*ctx.tools, "finish"])
        observation, ok = f"ERROR unknown action {decision.action!r}; use one of: {known}", False
    elif decision.action == "fetch_page" and url in st.pages:
        observation, ok = f"ERROR you already read this page. {st.hint()}", False
    elif decision.action == "fetch_page":
        output = _fetch(ctx, url, st)
        observation, ok = output.observation, not output.error
        if not ok:
            observation += f"\n{st.hint()}"
    else:
        output = call_tool(tool, decision.input)
        observation, ok = output.observation, not output.error
        for page in output.pages:
            st.pages.setdefault(page.url, page)
        if decision.action == "web_search":
            st.queries.append(str(decision.input.get("query", "")))
            if ok:
                st.add_candidates(_URL_LINE.findall(output.observation))
    step.tool_call = ToolCallRecord(
        name=decision.action,
        input=decision.input,
        ok=ok,
        latency_s=round(ctx.clock() - started, 3),
        observation_chars=len(observation),
        error=None if ok else observation.splitlines()[0][:300],
    )
    st.history.append(_item(decision.action, decision.input, observation))


# --------------------------------------------------------------------- finalize


def _finalize(
    ctx: WorkerContext, brief: SectionBrief, st: _Attempt, section_start: float
) -> tuple[list[Finding], Confidence, list[str]]:
    pages = list(st.pages.values())
    sources = "\n\n".join(
        f"[{i}] {p.title}\nURL: {p.url}\n{p.text[:FINAL_CHARS_PER_PAGE]}"
        for i, p in enumerate(pages, 1)
    )
    prompt = prompts.WORKER_FINAL_PROMPT.format(
        brief=render_brief(brief, ctx.topic, ctx.today), sources=sources, max_findings=MAX_FINDINGS
    )
    last_error = "no reply"
    for attempt in range(2):
        step = StepRecord(index=0, kind="finalize", worker=brief.id)
        try:
            gen = _generate(
                ctx,
                brief,
                step,
                prompt,
                prompts.WORKER_FINAL_SYSTEM,
                FINAL_MAX_TOKENS,
                section_start,
                temperature=0.0 if attempt == 0 else RETRY_TEMPERATURE,
            )
        finally:
            ctx.trace.add_step(step)
            st.steps.append(step)
        try:
            return parse_findings(extract_json(gen.text), len(pages))
        except ValueError as exc:
            last_error = str(exc).splitlines()[0][:120]
            step.note = f"unparseable finalize reply ({last_error})"
    raise _AttemptFailed(f"finalize reply was not usable: {last_error}")


# ---------------------------------------------------------------------- helpers


def _generate(
    ctx: WorkerContext,
    brief: SectionBrief,
    step: StepRecord,
    prompt: str,
    system: str,
    max_tokens: int,
    section_start: float,
    *,
    temperature: float = 0.0,
) -> Generation:
    remaining = ctx.budget.worker_max_seconds - (ctx.clock() - section_start)
    return generate_charged(
        ctx.llm,
        step,
        prompt,
        system,
        max_tokens,
        ctx.meter,
        ctx.gate,
        clock=ctx.clock,
        temperature=temperature,
        enforce_limit=True,
        max_wait_s=remaining,
        on_wait=lambda waited: ctx.trace.add_rate_limit_wait(brief.id, ctx.meter.elapsed_s, waited),
        cost_fn=ctx.cost_fn,
    )


def _item(action: str, tool_input: dict[str, Any], observation: str) -> dict[str, Any]:
    return {"action": action, "input": tool_input, "observation": observation}


def _render_pages(st: _Attempt) -> str:
    if not st.pages:
        return "(none yet)"
    return "\n".join(f"- {p.title} ({p.url})" for p in st.pages.values())


def _render_history(st: _Attempt) -> str:
    items = st.history[-HISTORY_ITEMS:]
    if not items:
        return "(none yet)"
    lines = []
    for n, item in enumerate(items):
        last = n == len(items) - 1
        text = item["observation"] if last else item["observation"][:OLD_OBSERVATION_CHARS]
        lines.append(f"{item['action']} {item['input']}\nResult: {text}")
    return "\n\n".join(lines)
