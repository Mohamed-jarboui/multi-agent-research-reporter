"""Multi-agent research (R1): a supervisor plans sections, workers research them in parallel, a
deterministic merge joins the results.

Graph:  START -> supervisor -> Send("worker", brief) x N -> merge -> END

The meter, the LLM gate and the trace live on the agent instance, not in the graph state (they are
not serializable). There is no LLM writer yet: `render_digest` is a plain Markdown digest of the
findings with global source numbers; the critic and the writer arrive in R2.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError, field_validator

from reporter.agent import prompts
from reporter.agent.llm_call import LLM, LLMGate, generate_charged
from reporter.agent.parsing import extract_json
from reporter.agent.schemas import (
    Finding,
    MergedResearch,
    SectionBrief,
    SectionOut,
    State,
    WorkerInput,
    WorkerResult,
)
from reporter.agent.worker import WorkerContext, run_worker
from reporter.budget import BudgetMeter, MultiBudget
from reporter.llm.base import ProviderError
from reporter.pricing import call_cost_usd
from reporter.tools import Tool
from reporter.trace import RUNS_DIR, SourceRecord, StepRecord, Trace, new_run_id

MIN_SECTIONS = 3
MAX_SECTIONS = 6
PLAN_MAX_TOKENS = 1_500
RETRY_TEMPERATURE = 0.7

GENERIC_SECTIONS = [
    ("Background and current state", "What is the topic and where does it stand today?"),
    ("Numbers and evidence", "What measurable evidence exists, with figures and dates?"),
    ("Limits, risks and outlook", "What are the limits and risks, and what comes next?"),
]


# ------------------------------------------------------------------------ plan


class _RawSection(BaseModel):
    title: str
    goal: str
    scope: list[str]
    exclude: list[str] = []
    expected_output: str = "3-6 findings with numbers and dates"

    @field_validator("scope", "exclude", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("title", "goal")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


def parse_sections(text: str) -> list[SectionBrief]:
    """Valid, distinct sections of a supervisor reply (at most 6); may be fewer than 3."""
    try:
        raw = extract_json(text).get("sections")
    except ValueError:
        return []
    if not isinstance(raw, list):
        return []
    briefs: list[SectionBrief] = []
    seen: set[str] = set()
    for item in raw:
        try:
            section = _RawSection.model_validate(item)
        except ValidationError:
            continue
        scope = [s.strip() for s in section.scope if str(s).strip()][:4]
        key = section.title.casefold()
        if not scope or key in seen:
            continue
        seen.add(key)
        briefs.append(
            SectionBrief(
                id=f"s{len(briefs) + 1}",
                title=section.title,
                goal=section.goal,
                scope=scope,
                exclude=[s.strip() for s in section.exclude if str(s).strip()],
                expected_output=section.expected_output.strip()
                or SectionBrief.model_fields["expected_output"].default,
            )
        )
        if len(briefs) == MAX_SECTIONS:
            break
    return briefs


def generic_plan(topic: str) -> list[SectionBrief]:
    """Three topic-independent sections, used when the supervisor gave nothing usable."""
    return [
        SectionBrief(
            id=f"s{i}",
            title=title,
            goal=f"{goal} (topic: {topic})",
            scope=[goal],
            exclude=[t for j, (t, _) in enumerate(GENERIC_SECTIONS, 1) if j != i],
        )
        for i, (title, goal) in enumerate(GENERIC_SECTIONS, 1)
    ]


def parse_plan(text: str, topic: str) -> list[SectionBrief]:
    """Sections from a reply, or the 3 generic sections when none is valid."""
    return parse_sections(text) or generic_plan(topic)


# ----------------------------------------------------------------------- merge


def normalize_url(url: str) -> str:
    """Key for 'the same page': lowercase scheme and host, no fragment, no trailing slash."""
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{path}{query}"


def merge_results(sections: list[SectionBrief], results: list[WorkerResult]) -> MergedResearch:
    """Join worker results. Source numbers follow section order, never completion order."""
    by_id = {r.section_id: r for r in results}
    merged = MergedResearch()
    global_ids: dict[str, int] = {}  # normalized url -> global id
    seen_questions: set[str] = set()
    for brief in sections:
        result = by_id.get(brief.id)
        if result is None:
            merged.sections.append(
                SectionOut(
                    brief=brief,
                    status="incomplete",
                    confidence="low",
                    findings=[],
                    error="no result",
                )
            )
            merged.incomplete.append(brief.id)
            continue
        local_to_global: dict[int, int] = {}
        for local, page in enumerate(result.pages, 1):
            key = normalize_url(page.url)
            if key not in global_ids:
                global_ids[key] = len(global_ids) + 1
                merged.sources.append(
                    SourceRecord(id=global_ids[key], url=page.url, title=page.title)
                )
                merged.pages[page.url] = page
            local_to_global[local] = global_ids[key]
        findings: list[Finding] = []
        for finding in result.findings:
            ids = sorted({local_to_global[n] for n in finding.source_ids if n in local_to_global})
            merged.dropped_citations += len(finding.source_ids) - len(
                [n for n in finding.source_ids if n in local_to_global]
            )
            if ids:
                findings.append(Finding(claim=finding.claim, source_ids=ids))
        for question in result.open_questions:
            if question.casefold() not in seen_questions:
                seen_questions.add(question.casefold())
                merged.open_questions.append((brief.id, question))
        status = result.status if findings else "incomplete"
        if status == "incomplete":
            merged.incomplete.append(brief.id)
        merged.sections.append(
            SectionOut(
                brief=brief,
                status=status,
                confidence=result.confidence,
                findings=findings,
                attempts=result.attempts,
                error=result.error,
            )
        )
    return merged


def render_digest(topic: str, merged: MergedResearch) -> str:
    """Markdown digest: findings per section cited as [n][m], never grouped, then the sources."""
    done = len(merged.sections) - len(merged.incomplete)
    lines = [
        f"# {topic}",
        "",
        f"Research digest: {done} of {len(merged.sections)} sections complete, "
        f"{len(merged.sources)} sources read.",
    ]
    for section in merged.sections:
        lines += [
            "",
            f"## {section.brief.title}",
            "",
            f"*Status: {section.status}; confidence: {section.confidence}.*",
            "",
        ]
        if not section.findings:
            reason = f" ({section.error})" if section.error else ""
            lines.append(f"INCOMPLETE: no usable finding for this section{reason}.")
        for finding in section.findings:
            cites = "".join(f"[{n}]" for n in finding.source_ids)
            lines.append(f"- {finding.claim} {cites}")
        questions = [q for sid, q in merged.open_questions if sid == section.brief.id]
        if questions:
            lines += ["", "Open questions:"] + [f"- {q}" for q in questions]
    lines += ["", "## Sources", ""]
    lines += [f"{s.id}. [{s.title}]({s.url})" for s in merged.sources]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- agent


@dataclass(frozen=True)
class MultiRunResult:
    report: str  # the digest
    merged: MergedResearch
    trace: Trace
    trace_path: Path


class MultiAgent:
    """One research run at a time per instance."""

    def __init__(
        self,
        llm: LLM,
        tools: dict[str, Tool],
        budget: MultiBudget | None = None,
        *,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        today: date | None = None,
        runs_dir: Path = RUNS_DIR,
        cost_fn: Callable[[str, int | None, int | None], float | None] = call_cost_usd,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.budget = budget or MultiBudget()
        self._clock = clock
        self._sleep = sleep
        self._today = (today or date.today()).isoformat()
        self._runs_dir = runs_dir
        self._cost_fn = cost_fn
        self._graph = self._build_graph()
        self.last_trace_path: Path | None = None
        self._ctx: WorkerContext

    # ------------------------------------------------------------------ public

    def run(self, topic: str) -> MultiRunResult:
        """Research `topic`. The trace is saved even when the run fails."""
        meter = BudgetMeter(self.budget.run, self._clock)
        trace = Trace(
            run_id=new_run_id(),
            system="multi",
            topic=topic,
            started_at=datetime.now(UTC).isoformat(timespec="seconds"),
            budget=asdict(self.budget),
            concurrency={
                "max_parallel_workers": self.budget.max_parallel_workers,
                "max_parallel_llm_calls": self.budget.max_parallel_llm_calls,
            },
        )
        self._ctx = WorkerContext(
            topic=topic,
            today=self._today,
            llm=self.llm,
            tools=self.tools,
            budget=self.budget,
            meter=meter,
            gate=LLMGate(self.budget.max_parallel_llm_calls, clock=self._clock, sleep=self._sleep),
            trace=trace,
            clock=self._clock,
            cost_fn=self._cost_fn,
        )
        initial: State = {
            "topic": topic,
            "sections": [],
            "results": [],
            "merged": None,
            "digest": "",
        }
        config: Any = {"max_concurrency": self.budget.max_parallel_workers}
        try:
            final = self._graph.invoke(initial, config)
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            trace.report = final["digest"]
            return MultiRunResult(
                final["digest"], final["merged"], trace, self._runs_dir / f"{trace.run_id}.json"
            )
        finally:
            trace.finish(meter.elapsed_s)
            self.last_trace_path = trace.save(self._runs_dir)

    # ------------------------------------------------------------------- graph

    def _build_graph(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("worker", self._worker)
        graph.add_node("merge", self._merge)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._fan_out, ["worker"])
        graph.add_edge("worker", "merge")
        graph.add_edge("merge", END)
        return graph.compile()

    def _supervisor(self, state: State) -> State:
        ctx = self._ctx
        topic = state["topic"]
        prompt = prompts.SUPERVISOR_PROMPT.format(
            topic=topic, today=self._today, low=MIN_SECTIONS, high=MAX_SECTIONS
        )
        best: list[SectionBrief] = []
        for attempt in range(2):
            step = StepRecord(index=0, kind="plan")
            text, failure = "", ""
            try:
                gen = generate_charged(
                    self.llm,
                    step,
                    prompt,
                    prompts.SUPERVISOR_SYSTEM,
                    PLAN_MAX_TOKENS,
                    ctx.meter,
                    ctx.gate,
                    clock=self._clock,
                    temperature=0.0 if attempt == 0 else RETRY_TEMPERATURE,
                    enforce_limit=True,
                    on_wait=lambda w: ctx.trace.add_rate_limit_wait(
                        "supervisor", ctx.meter.elapsed_s, w
                    ),
                    cost_fn=self._cost_fn,
                )
                text = gen.text
            except ProviderError as exc:
                failure = f"model call failed: {exc}"
                step.note = failure
            finally:
                ctx.trace.add_step(step)
            sections = parse_sections(text)
            if len(sections) > len(best):
                best = sections
            if len(best) >= MIN_SECTIONS:
                break
            reason = failure or f"only {len(sections)} valid sections"
            prompt += prompts.SUPERVISOR_RETRY_HINT.format(reason=reason, low=MIN_SECTIONS)
        if not best:
            best = generic_plan(topic)
            ctx.trace.notes.append("plan_generic: no valid section from the supervisor")
        elif len(best) < MIN_SECTIONS:
            ctx.trace.notes.append(f"plan_short: {len(best)} valid sections")
        ctx.trace.sections = [b.model_dump() for b in best]
        return {"sections": best}

    @staticmethod
    def _fan_out(state: State) -> list[Send]:
        return [
            Send("worker", {"brief": brief, "index": i})
            for i, brief in enumerate(state["sections"])
        ]

    def _worker(self, inp: WorkerInput) -> dict[str, list[WorkerResult]]:
        return {"results": [run_worker(self._ctx, inp["brief"], inp["index"])]}

    def _merge(self, state: State) -> State:
        merged = merge_results(state["sections"], state["results"])
        digest = render_digest(state["topic"], merged)
        trace = self._ctx.trace
        trace.sources = list(merged.sources)
        pages_read = {r.section_id: len(r.pages) for r in state["results"]}
        trace.coverage = {s.brief.title: pages_read.get(s.brief.id, 0) for s in merged.sections}
        trace.stop_reason = "all_sections_done" if not merged.incomplete else "sections_incomplete"
        return {"merged": merged, "digest": digest}
