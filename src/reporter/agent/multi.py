"""Multi-agent research: a supervisor plans sections, workers research them in parallel, a
deterministic merge joins the results (R1); a critic grades them, weak sections go back to their
worker once, and a writer produces the cited report (R2).

Graph:  START -> supervisor -> Send("worker") x N -> merge -> critic
        -> [Send("retry_worker") x K -> regrade] -> writer -> END

The meter, the LLM gate and the trace live on the agent instance, not in the graph state (they are
not serializable). `render_digest` is the plain Markdown digest of the findings (the R1 output); it
is the fallback when the writer fails and the output with `critic=False, writer=False`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ValidationError, field_validator

from reporter.agent import prompts
from reporter.agent.checks import (
    check_citations,
    deterministic_reviews,
    fetched_urls,
    normalize_url,
    plain_reason,
    section_problem,
    section_reason,
)
from reporter.agent.critic import (
    MIN_RETRY_SECONDS,
    attempt_rank,
    feedback_text,
    retry_candidates,
    run_critic,
)
from reporter.agent.llm_call import LLM, LLMGate, generate_charged, llm_summary
from reporter.agent.parsing import extract_json, strip_sources_section
from reporter.agent.schemas import (
    Finding,
    MergedResearch,
    Review,
    SectionBrief,
    SectionOut,
    SectionReview,
    State,
    WorkerInput,
    WorkerResult,
)
from reporter.agent.worker import WorkerContext, run_worker
from reporter.agent.writer import (
    assemble_report,
    no_report_text,
    run_writer,
    sanitize_body,
    usable_sections,
)
from reporter.budget import BudgetMeter, MultiBudget
from reporter.llm.base import ProviderError
from reporter.pricing import call_cost_usd
from reporter.tools import Tool
from reporter.trace import RUNS_DIR, SourceRecord, StepRecord, Trace, new_run_id

MIN_SECTIONS = 3
MAX_SECTIONS = 6
PLAN_MAX_TOKENS = 1_500
RETRY_TEMPERATURE = 0.7
RETRY_GATE_FRACTION = 0.85  # no retry starts once any limit is past this
RETRY_STOP_FRACTION = 0.9  # a retry worker stops researching here (research stops at 0.7)
STEPS_PER_RETRY = 3  # run steps one retry needs at least

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
                n_pages=len(result.pages),
                stop_reason=result.stop_reason,
                blocked_fetches=sum(
                    f.get("reason") in ("HTTP 403", "HTTP 429") for f in result.failed_fetches
                ),
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
            reason = f" ({section_reason(section)})"
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
    report: str  # the written report (the digest when the writer is off or failed)
    merged: MergedResearch
    trace: Trace
    trace_path: Path
    review: Review | None = None
    digest: str = ""


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
        critic: bool = True,
        writer: bool = True,
    ) -> None:
        self.use_critic = critic
        self.use_writer = writer
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
            "review": None,
            "retry_ids": [],
            "report": "",
        }
        config: Any = {"max_concurrency": self.budget.max_parallel_workers}
        try:
            final = self._graph.invoke(initial, config)
        except Exception as exc:
            trace.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            trace.report = final["report"]
            return MultiRunResult(
                final["report"],
                final["merged"],
                trace,
                self._runs_dir / f"{trace.run_id}.json",
                final["review"],
                final["digest"],
            )
        finally:
            trace.llm = llm_summary(self.llm)
            trace.finish(meter.elapsed_s)
            self.last_trace_path = trace.save(self._runs_dir)

    # ------------------------------------------------------------------- graph

    def _build_graph(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("worker", self._worker)
        graph.add_node("merge", self._merge)
        graph.add_node("critic", self._critic)
        graph.add_node("retry_worker", self._retry_worker)
        graph.add_node("regrade", self._regrade)
        graph.add_node("writer", self._writer)
        graph.add_edge(START, "supervisor")
        graph.add_conditional_edges("supervisor", self._fan_out, ["worker"])
        graph.add_edge("worker", "merge")
        graph.add_edge("merge", "critic")
        # one pass only: the retry branch has no way back to the critic
        graph.add_conditional_edges("critic", self._after_critic, ["retry_worker", "writer"])
        graph.add_edge("retry_worker", "regrade")
        graph.add_edge("regrade", "writer")
        graph.add_edge("writer", END)
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
        self._record_merge(merged, state["results"])
        return {"merged": merged, "digest": render_digest(state["topic"], merged)}

    def _record_merge(self, merged: MergedResearch, results: list[WorkerResult]) -> None:
        trace = self._ctx.trace
        trace.sources = list(merged.sources)
        pages_read = {r.section_id: len(r.pages) for r in results}
        trace.coverage = {s.brief.title: pages_read.get(s.brief.id, 0) for s in merged.sections}
        trace.stop_reason = "all_sections_done" if not merged.incomplete else "sections_incomplete"

    # ------------------------------------------------------------------ critic

    def _review(self, merged: MergedResearch, ids: list[str]) -> Review:
        """Deterministic verdicts first (no model), then one critic call for the rest."""
        reviews = {k: v for k, v in deterministic_reviews(merged).items() if k in ids}
        todo = [i for i in ids if i not in reviews]
        review = Review(sections=reviews, critic_status="ok" if self.use_critic else "skipped")
        if self.use_critic and todo:
            graded, failure = run_critic(self._ctx, merged, todo)
            if graded is not None:
                review.sections.update(graded.sections)
                review.contradictions = graded.contradictions
                review.coverage_gaps = graded.coverage_gaps
            else:
                review.critic_status = "failed"
                self._ctx.trace.notes.append(f"critic_failed: {failure}")
        for sid in todo:
            review.sections.setdefault(sid, SectionReview(section_id=sid, verdict="unreviewed"))
        return review

    def _critic(self, state: State) -> State:
        merged = state["merged"]
        assert merged is not None
        review = self._review(merged, [s.brief.id for s in merged.sections])
        trace = self._ctx.trace
        trace.critic = {
            "status": review.critic_status,
            "before": {k: v.verdict for k, v in review.sections.items()},
            "issues": {
                k: [i.model_dump() for i in v.issues]
                for k, v in review.sections.items()
                if v.issues
            },
            "contradictions": [c.model_dump() for c in review.contradictions],
            "coverage_gaps": review.coverage_gaps,
        }
        ids: list[str] = []
        if self.use_critic:
            ids = retry_candidates(review, set())[: self._retry_capacity()]
            if not ids and retry_candidates(review, set()):
                trace.critic["retry_skipped"] = "not enough budget left for a retry"
                ids = []
        return {"review": review, "retry_ids": ids}

    def _retry_capacity(self) -> int:
        """How many sections may be retried: what one retry needs, not the research share.

        Retries may use the reserve up to RETRY_GATE_FRACTION of every limit; each needs about
        STEPS_PER_RETRY run steps and MIN_RETRY_SECONDS of time.
        """
        meter, run = self._ctx.meter, self.budget.run
        if meter.stop_reason(RETRY_GATE_FRACTION):
            return 0
        if RETRY_GATE_FRACTION * run.max_seconds - meter.elapsed_s < MIN_RETRY_SECONDS:
            return 0
        return max(run.max_steps - meter.steps, 0) // STEPS_PER_RETRY

    def _after_critic(self, state: State) -> list[Send] | str:
        ids, review = state["retry_ids"], state["review"]
        if not ids or review is None:
            return "writer"
        sends: list[Send] = []
        for sid in ids:
            index = next(i for i, b in enumerate(state["sections"]) if b.id == sid)
            previous = next((r for r in state["results"] if r.section_id == sid), None)
            urls = [p.url for p in previous.pages] if previous else []
            payload = {
                "brief": state["sections"][index],
                "index": index,
                "feedback": feedback_text(review, sid, urls, previous),
                "seeds": previous.untried_urls if previous else [],
            }
            sends.append(Send("retry_worker", payload))
        return sends

    def _retry_worker(self, inp: WorkerInput) -> dict[str, list[WorkerResult]]:
        result = run_worker(
            replace(self._ctx, stop_fraction=RETRY_STOP_FRACTION),
            inp["brief"],
            inp["index"],
            feedback=inp.get("feedback", ""),
            max_attempts=1,
            phase="retry",
            seed_urls=inp.get("seeds", []),
        )
        return {"retry_results": [result]}

    def _regrade(self, state: State) -> State:
        """Grade the retried sections again and keep the better attempt of each."""
        sections, retried, before = state["sections"], state["retry_ids"], state["review"]
        assert before is not None
        originals = {r.section_id: r for r in state["results"]}
        retries = {r.section_id: r for r in state.get("retry_results", [])}
        swapped = [retries.get(b.id) or originals[b.id] for b in sections if b.id in originals]
        after = self._review(merge_results(sections, swapped), retried)

        chosen: dict[str, WorkerResult] = dict(originals)
        final = before.model_copy(deep=True)
        records: list[dict[str, Any]] = []
        for sid in retried:
            old, new = originals.get(sid), retries.get(sid)
            old_verdict = before.sections[sid].verdict
            new_verdict = after.sections[sid].verdict
            # a retry the critic could not grade is judged by the old verdict (pages and findings)
            judged = old_verdict if new_verdict == "unreviewed" else new_verdict
            better = new is not None and (
                old is None or attempt_rank(judged, new) > attempt_rank(old_verdict, old)
            )
            if better and new is not None:
                chosen[sid] = new
                final.sections[sid] = after.sections[sid]
            records.append(
                {
                    "section_id": sid,
                    "before": _attempt_summary(old_verdict, old),
                    "after": _attempt_summary(new_verdict, new),
                    "kept": "retry" if better else "original",
                    "issues": [i.issue for i in before.sections[sid].issues],
                }
            )
        replaced = {r["section_id"] for r in records if r["kept"] == "retry"}
        final.contradictions = [
            c for c in final.contradictions if not replaced & set(c.section_ids)
        ] + after.contradictions
        trace = self._ctx.trace
        trace.retries = records
        trace.critic["after"] = {sid: after.sections[sid].verdict for sid in retried}
        trace.critic["final"] = {k: v.verdict for k, v in final.sections.items()}
        results = [chosen[b.id] for b in sections if b.id in chosen]
        merged = merge_results(sections, results)
        self._record_merge(merged, results)
        return {"merged": merged, "digest": render_digest(state["topic"], merged), "review": final}

    # ------------------------------------------------------------------ writer

    def _writer(self, state: State) -> State:
        merged, review = state["merged"], state["review"]
        assert merged is not None and review is not None
        trace = self._ctx.trace
        digest = state["digest"]
        if not self.use_writer:
            report, stage = digest, "digest"
            trace.writer = {"status": "skipped"}
        elif not usable_sections(merged, review):
            report, stage = no_report_text(state["topic"], merged, review), "none"
            trace.writer = {"status": "no_report", "reason": "no section has a usable finding"}
        else:
            body, failure = run_writer(self._ctx, merged, review, self._today)
            if failure:
                warning = (
                    f"> Warning: the writer step failed ({plain_reason(failure)}); "
                    "this is the unedited research digest.\n\n"
                )
                report, stage = warning + digest, "digest"
                trace.writer = {"status": "fallback", "reason": failure}
                trace.notes.append(f"writer_failed: {failure}")
            else:
                body, removed = sanitize_body(body, merged, review)
                report, stage = assemble_report(body, merged, review), "report"
                trace.writer = {"status": "ok"} | {k: v for k, v in removed.items() if v}
        results = [*state["results"], *state.get("retry_results", [])]
        fetched = fetched_urls(trace, [p.url for r in results for p in r.pages])
        check = check_citations(strip_sources_section(report), merged.sources, fetched)
        trace.checks = {
            "stage": stage,
            **check.model_dump(),
            "research": {
                s.brief.id: problem for s in merged.sections if (problem := section_problem(s))
            },
        }
        return {"report": report}


def _attempt_summary(verdict: str, result: WorkerResult | None) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "pages": len(result.pages) if result else 0,
        "findings": len(result.findings) if result else 0,
    }
