"""Fakes for the benchmark tests: an agent that writes a trace, and an LLM that scores."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reporter.bench.topics import Topic
from reporter.llm.base import Generation
from reporter.trace import SourceRecord, StepRecord, ToolCallRecord, Trace, new_run_id

URL1 = "https://example.org/a"
URL2 = "https://example.org/b"
GOOD_REPORT = (
    "# Topic\n\nSmall models run on phones using quantization techniques [1]. "
    "They have clear quality limits compared with cloud models [2].\n\n## Sources\n\n"
    f"1. [A]({URL1})\n2. [B]({URL2})\n"
)


def topic(topic_id: str = "t1", points: int = 3) -> Topic:
    return Topic(
        id=topic_id,
        topic=f"topic {topic_id}",
        domain="tech",
        must_cover=[f"point {i}" for i in range(1, points + 1)],
    )


def make_trace(
    system: str,
    *,
    ok: bool = True,
    report: str = GOOD_REPORT,
    limits: dict[str, dict[str, int]] | None = None,
    providers: tuple[str, ...] = ("gemini", "groq"),
    tokens: tuple[int, int] = (100, 50),
    latency: float = 10.0,
    cost: float = 0.0,
    model: str = "fake-model",
    fetched: tuple[str, ...] = (URL1, URL2),
) -> Trace:
    trace = Trace(run_id=new_run_id(), system=system, topic="topic", started_at="now")
    for url in fetched:
        trace.add_step(
            StepRecord(
                index=0,
                kind="think",
                model=model,
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                cost_usd=cost,
                tool_call=ToolCallRecord(
                    name="fetch_page",
                    input={"url": url},
                    ok=True,
                    latency_s=0.1,
                    observation_chars=9,
                ),
            )
        )
    if ok:
        trace.sources = [
            SourceRecord(id=1, url=URL1, title="A"),
            SourceRecord(id=2, url=URL2, title="B"),
        ]
        trace.report = report
        trace.stop_reason = "coverage_met"
        if system == "multi":
            trace.writer = {"status": "ok"}
    else:
        trace.report = "No report could be written."
        trace.stop_reason = "model_output_unparseable"
        trace.writer = {"status": "no_report", "reason": "no section has a usable finding"}
    trace.llm = {"providers": list(providers), "limits": limits or {}, "calls": {}, "failures": {}}
    trace.finish(latency)
    return trace


class FakeAgent:
    """Saves a trace like the real agents; `outcome` is a Trace to save or an Exception to raise
    (the trace is saved first, as the real agents do)."""

    def __init__(self, outcome: Trace | Exception, runs_dir: Path) -> None:
        self.outcome = outcome
        self.runs_dir = runs_dir
        self.last_trace_path: Path | None = None

    def run(self, topic: str) -> Any:
        if isinstance(self.outcome, Exception):
            trace = make_trace("baseline", ok=False)
            trace.error = str(self.outcome)
            self.last_trace_path = trace.save(self.runs_dir)
            raise self.outcome
        self.last_trace_path = self.outcome.save(self.runs_dir)
        return self.outcome


class AgentFactory:
    """`script(system, call_number)` returns the outcome for each run; calls are recorded."""

    def __init__(self, script: Callable[[str, int], Trace | Exception]) -> None:
        self.script = script
        self.calls: list[str] = []

    def __call__(self, system: str, runs_dir: Path) -> FakeAgent:
        self.calls.append(system)
        return FakeAgent(self.script(system, len(self.calls)), runs_dir)


class ScoringLLM:
    """Answers coverage prompts with the quotes in `quotes` (point number -> quote) and judge
    prompts with `winner` (a function of the prompt's A and B texts, or a fixed string)."""

    def __init__(
        self,
        quotes: dict[int, str] | None = None,
        winner: str | Callable[[str], str] = "A",
        model: str = "judge-model",
    ) -> None:
        self.quotes = quotes if quotes is not None else {}
        self.winner = winner
        self.model = model
        self.prompts: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> Generation:
        self.prompts.append(prompt)
        if "=== Report A ===" in prompt:
            winner = self.winner(prompt) if callable(self.winner) else self.winner
            text = json.dumps({"winner": winner, "reason": "because"})
        else:
            points = [{"n": n, "matched": True, "quote": q} for n, q in sorted(self.quotes.items())]
            text = json.dumps({"points": points})
        return Generation(text, self.model, 10, 5)
