"""Single-agent baseline: a ReAct loop (LangGraph) that plans questions, searches, reads pages and
writes a cited report. The multi-agent system has to beat this.

Graph:  plan -> think -> act -> check -+-> think   (loop)
                                       +-> write -> END

The provider layer has no native tool calling, so the model answers each turn with a JSON action
and the tools are called by this code. Stop rule: every planned question has at least
`min_sources` pages read, or any budget limit reaches 80%. A model "finish" before that is
refused twice, then accepted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from reporter.agent import prompts
from reporter.agent.llm_call import LLM, generate_charged, llm_summary
from reporter.agent.parsing import cited_numbers, extract_json, strip_sources_section
from reporter.budget import Budget, BudgetMeter
from reporter.llm.base import Generation, ProviderError
from reporter.tools import Page, Tool, call_tool, describe
from reporter.trace import RUNS_DIR, SourceRecord, StepRecord, ToolCallRecord, Trace, new_run_id

PLAN_MAX_TOKENS = 1_500
THINK_MAX_TOKENS = 2_000
WRITE_MAX_TOKENS = 6_000
WRITE_CHARS_PER_SOURCE = 3_000  # page text given to the writer, per source
OLD_OBSERVATION_CHARS = 200  # older tool results are shortened in the prompt
HISTORY_ITEMS = 6
MAX_CONSECUTIVE_PARSE_FAILURES = 3
MAX_FINISH_REFUSALS = 2  # "finish" before coverage is met is refused this many times


class Decision(BaseModel):
    """One ReAct turn as the model must write it."""

    thought: str = ""
    action: str
    input: dict[str, Any] = Field(default_factory=dict)
    answers: list[int] = Field(default_factory=list)


class State(TypedDict, total=False):
    questions: list[str]
    pages: dict[str, Page]  # url -> page; insertion order gives the source numbers
    coverage: dict[int, list[str]]  # question index -> urls read for it
    history: list[dict[str, Any]]
    pending: Decision | None
    parse_failures: int
    finish_refusals: int
    stop_reason: str
    report: str


@dataclass(frozen=True)
class RunResult:
    report: str
    trace: Trace
    trace_path: Path


class BaselineAgent:
    """One research run at a time per instance."""

    def __init__(
        self,
        llm: LLM,
        tools: dict[str, Tool],
        budget: Budget | None = None,
        *,
        min_sources: int = 2,
        clock: Callable[[], float] = time.monotonic,
        today: date | None = None,
        runs_dir: Path = RUNS_DIR,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.budget = budget or Budget()
        self.min_sources = min_sources
        self._clock = clock
        self._today = (today or date.today()).isoformat()
        self._runs_dir = runs_dir
        self._graph = self._build_graph()
        self.last_trace_path: Path | None = None
        self._meter: BudgetMeter
        self._trace: Trace
        self._topic = ""

    # ------------------------------------------------------------------ public

    def run(self, topic: str) -> RunResult:
        """Research `topic` and return the report. The trace is saved even when the run fails."""
        self._topic = topic
        self._meter = BudgetMeter(self.budget, self._clock)
        self._trace = Trace(
            run_id=new_run_id(),
            system="baseline",
            topic=topic,
            started_at=_now_iso(),
            budget=asdict(self.budget),
        )
        initial: State = {
            "questions": [],
            "pages": {},
            "coverage": {},
            "history": [],
            "pending": None,
            "parse_failures": 0,
            "finish_refusals": 0,
            "stop_reason": "",
            "report": "",
        }
        try:
            final = self._graph.invoke(initial, {"recursion_limit": 3 * self.budget.max_steps + 20})
        except Exception as exc:
            self._trace.error = f"{type(exc).__name__}: {exc}"
            raise
        else:
            self._trace.report = final["report"]
            return RunResult(final["report"], self._trace, self._trace_path)
        finally:
            self._trace.llm = llm_summary(self.llm)
            self._trace.finish(self._meter.elapsed_s)
            self.last_trace_path = self._trace.save(self._runs_dir)

    @property
    def _trace_path(self) -> Path:
        return self._runs_dir / f"{self._trace.run_id}.json"

    # ------------------------------------------------------------------- graph

    def _build_graph(self) -> Any:
        graph = StateGraph(State)
        graph.add_node("plan", self._plan)
        graph.add_node("think", self._think)
        graph.add_node("act", self._act)
        graph.add_node("check", self._check)
        graph.add_node("write", self._write)
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "think")
        graph.add_edge("think", "act")
        graph.add_edge("act", "check")
        graph.add_conditional_edges(
            "check", lambda s: "write" if s.get("stop_reason") else "think", ["think", "write"]
        )
        graph.add_edge("write", END)
        return graph.compile()

    def _plan(self, state: State) -> State:
        step = self._new_step("plan")
        gen = self._generate(
            step,
            prompts.PLAN_PROMPT.format(topic=self._topic, today=self._today, low=3, high=5),
            prompts.PLAN_SYSTEM,
            PLAN_MAX_TOKENS,
        )
        try:
            raw = extract_json(gen.text).get("questions")
            questions = [str(q).strip() for q in raw if str(q).strip()][:5]
        except (ValueError, TypeError):
            questions = []
        if not questions:
            questions = [self._topic]
            step.note = "plan was not valid JSON; using the topic as the only question"
        self._trace.add_step(step)
        self._trace.questions = questions
        return {"questions": questions}

    def _think(self, state: State) -> State:
        step = self._new_step("think")
        self._meter.add_step()
        prompt = prompts.THINK_PROMPT.format(
            topic=self._topic,
            today=self._today,
            questions=self._render_questions(state),
            pages=self._render_pages(state),
            history=self._render_history(state),
            budget=self._render_budget(),
        )
        system = prompts.THINK_SYSTEM.format(
            tools="\n".join(describe(t) for t in self.tools.values())
        )
        try:
            gen = self._generate(step, prompt, system, THINK_MAX_TOKENS)
        except ProviderError as exc:
            step.note = f"model call failed: {exc}"
            self._trace.add_step(step)
            return {"pending": None, "stop_reason": "llm_error"}
        try:
            decision = Decision.model_validate(extract_json(gen.text))
        except ValueError as exc:
            failures = state["parse_failures"] + 1
            reason = str(exc).splitlines()[0][:120]
            step.note = (
                f"unparseable reply ({reason}); it began: {gen.text[:150]!r}. {step.note}".strip()
            )
            self._trace.add_step(step)
            hint = "ERROR your last reply was not a JSON object; answer with only the JSON object"
            history = [*state["history"], _history_item(self._meter.steps, "none", {}, hint)]
            return {"pending": None, "parse_failures": failures, "history": history}
        step.thought, step.action = decision.thought, decision.action
        self._trace.add_step(step)
        return {"pending": decision, "parse_failures": 0}

    def _act(self, state: State) -> State:
        decision = state.get("pending")
        if decision is None:
            return {}
        if decision.action == "finish":
            missing = self._uncovered(state)
            refusals = state["finish_refusals"]
            if missing and refusals < MAX_FINISH_REFUSALS:
                numbers = ", ".join(str(i + 1) for i in missing)
                note = (
                    f"ERROR finish refused: questions {numbers} have fewer than "
                    f"{self.min_sources} sources. Keep researching."
                )
                history = [*state["history"], _history_item(self._meter.steps, "finish", {}, note)]
                return {"history": history, "pending": None, "finish_refusals": refusals + 1}
            return {"stop_reason": "model_finished", "pending": None}
        step = self._trace.steps[-1]  # the think step this action belongs to
        pages = dict(state["pages"])
        coverage = {k: list(v) for k, v in state["coverage"].items()}
        tool = self.tools.get(decision.action)
        started = self._clock()
        if tool is None:
            known = ", ".join([*self.tools, "finish"])
            observation, ok = (
                f"ERROR unknown action {decision.action!r}; use one of: {known}",
                False,
            )
        elif decision.action == "fetch_page" and str(decision.input.get("url", "")) in pages:
            observation, ok = "ERROR you already read this page; choose another source", False
        else:
            output = call_tool(tool, decision.input)
            observation, ok = output.observation, not output.error
            for page in output.pages:
                pages.setdefault(page.url, page)
                for number in decision.answers:
                    if 1 <= number <= len(state["questions"]):
                        urls = coverage.setdefault(number - 1, [])
                        if page.url not in urls:
                            urls.append(page.url)
        step.tool_call = ToolCallRecord(
            name=decision.action,
            input=decision.input,
            ok=ok,
            latency_s=round(self._clock() - started, 3),
            observation_chars=len(observation),
            error=None if ok else observation.splitlines()[0][:300],
        )
        self._trace.sources = [
            SourceRecord(id=i, url=p.url, title=p.title) for i, p in enumerate(pages.values(), 1)
        ]
        self._trace.coverage = {
            q: len(coverage.get(i, [])) for i, q in enumerate(state["questions"])
        }
        history = [
            *state["history"],
            _history_item(self._meter.steps, decision.action, decision.input, observation),
        ]
        return {"pages": pages, "coverage": coverage, "history": history, "pending": None}

    def _check(self, state: State) -> State:
        """Decide whether to keep researching; sets stop_reason when it should stop."""
        if state.get("stop_reason"):
            return {}
        if not self._uncovered(state):
            return {"stop_reason": "coverage_met"}
        reason = self._meter.stop_reason()
        if reason:
            return {"stop_reason": reason}
        if state["parse_failures"] >= MAX_CONSECUTIVE_PARSE_FAILURES:
            return {"stop_reason": "model_output_unparseable"}
        return {}

    def _uncovered(self, state: State) -> list[int]:
        """Indexes of planned questions that still have fewer than `min_sources` pages."""
        return [
            i
            for i in range(len(state["questions"]))
            if len(state["coverage"].get(i, [])) < self.min_sources
        ]

    def _write(self, state: State) -> State:
        self._trace.stop_reason = state["stop_reason"]
        pages = list(state["pages"].values())
        if not pages:
            body = (
                f"# {self._topic}\n\nNo report could be written: no web page could be read "
                f"(stopped because: {state['stop_reason']}). Check the trace in runs/ for the "
                "tool errors."
            )
            self._trace.checks = {"invalid_citations": [], "uncited_sources": []}
            return {"report": body}
        step = self._new_step("write")
        sources = "\n\n".join(
            f"[{i}] {p.title}\nURL: {p.url}\n{p.text[:WRITE_CHARS_PER_SOURCE]}"
            for i, p in enumerate(pages, 1)
        )
        gen = self._generate(
            step,
            prompts.WRITE_PROMPT.format(
                topic=self._topic,
                today=self._today,
                questions=self._render_questions(state, with_counts=False),
                sources=sources,
            ),
            prompts.WRITE_SYSTEM,
            WRITE_MAX_TOKENS,
        )
        self._trace.add_step(step)
        body = strip_sources_section(gen.text)
        used = cited_numbers(body)
        self._trace.checks = {
            "invalid_citations": [n for n in used if not 1 <= n <= len(pages)],
            "uncited_sources": [n for n in range(1, len(pages) + 1) if n not in used],
        }
        listing = "\n".join(f"{i}. [{p.title}]({p.url})" for i, p in enumerate(pages, 1))
        return {"report": f"{body}\n\n## Sources\n\n{listing}\n"}

    # ----------------------------------------------------------------- helpers

    def _new_step(self, kind: str) -> StepRecord:
        return StepRecord(index=len(self._trace.steps) + 1, kind=kind)  # type: ignore[arg-type]

    def _generate(self, step: StepRecord, prompt: str, system: str, max_tokens: int) -> Generation:
        return generate_charged(
            self.llm, step, prompt, system, max_tokens, self._meter, clock=self._clock
        )

    def _render_questions(self, state: State, *, with_counts: bool = True) -> str:
        lines = []
        for i, q in enumerate(state["questions"]):
            count = f" [{len(state['coverage'].get(i, []))} sources]" if with_counts else ""
            lines.append(f"{i + 1}. {q}{count}")
        return "\n".join(lines)

    @staticmethod
    def _render_pages(state: State) -> str:
        if not state["pages"]:
            return "(none yet)"
        return "\n".join(f"- {p.title} ({p.url})" for p in state["pages"].values())

    @staticmethod
    def _render_history(state: State) -> str:
        items = state["history"][-HISTORY_ITEMS:]
        if not items:
            return "(none yet)"
        lines = []
        for n, item in enumerate(items):
            last = n == len(items) - 1
            text = item["observation"] if last else item["observation"][:OLD_OBSERVATION_CHARS]
            lines.append(f"Step {item['step']}: {item['action']} {item['input']}\nResult: {text}")
        return "\n\n".join(lines)

    def _render_budget(self) -> str:
        used = self._meter.fractions()
        return ", ".join(f"{name} {fraction:.0%}" for name, fraction in used.items())


def _history_item(step: int, action: str, tool_input: dict[str, Any], observation: str) -> dict:
    return {"step": step, "action": action, "input": tool_input, "observation": observation}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
