"""Run traces: every run is saved to runs/<run_id>.json with steps, tool calls, tokens, cost and
latency, so runs can be compared without re-running them."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

RUNS_DIR = Path("runs")


class ToolCallRecord(BaseModel):
    name: str
    input: dict[str, Any]
    ok: bool
    latency_s: float
    observation_chars: int
    error: str | None = None


class StepRecord(BaseModel):
    """One model call, with the tool call it led to (if any)."""

    index: int
    kind: Literal["plan", "think", "write", "finalize"]
    thought: str = ""
    action: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    llm_latency_s: float = 0.0
    tool_call: ToolCallRecord | None = None
    note: str = ""
    worker: str | None = None  # section id in multi-agent runs


class SourceRecord(BaseModel):
    id: int
    url: str
    title: str


class WorkerRecord(BaseModel):
    """One worker attempt on one section. Times are seconds since the run started."""

    section_id: str
    attempt: int
    status: Literal["complete", "incomplete"]
    started_s: float
    ended_s: float
    started_at: str  # ISO time
    steps: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    n_pages: int = 0
    stop_reason: str = ""
    error: str | None = None


class Totals(BaseModel):
    react_steps: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    calls_without_token_counts: int = 0
    calls_without_price: int = 0


class Trace(BaseModel):
    run_id: str
    system: str
    topic: str
    started_at: str
    budget: dict[str, Any] = Field(default_factory=dict)
    questions: list[str] = Field(default_factory=list)
    steps: list[StepRecord] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    coverage: dict[str, int] = Field(default_factory=dict)  # question -> sources read for it
    stop_reason: str = ""
    checks: dict[str, Any] = Field(default_factory=dict)
    totals: Totals = Field(default_factory=Totals)
    report: str = ""
    error: str | None = None
    # multi-agent runs
    sections: list[dict[str, Any]] = Field(default_factory=list)  # the supervisor's briefs
    workers: list[WorkerRecord] = Field(default_factory=list)
    rate_limit_waits: list[dict[str, Any]] = Field(default_factory=list)
    concurrency: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def add_step(self, step: StepRecord) -> None:
        """Append a step and give it the next index (safe to call from several threads)."""
        with self._lock:
            step.index = len(self.steps) + 1
            self.steps.append(step)

    def add_worker(self, record: WorkerRecord) -> None:
        with self._lock:
            self.workers.append(record)

    def add_rate_limit_wait(self, section_id: str, at_s: float, waited_s: float) -> None:
        with self._lock:
            self.rate_limit_waits.append(
                {"section_id": section_id, "at_s": round(at_s, 3), "waited_s": round(waited_s, 3)}
            )

    def finish(self, latency_s: float) -> None:
        """Fill `totals` from the steps."""
        t = Totals(latency_s=round(latency_s, 3))
        for s in self.steps:
            t.llm_calls += 1
            t.react_steps += s.kind == "think"
            t.tool_calls += s.tool_call is not None
            if s.input_tokens is None or s.output_tokens is None:
                t.calls_without_token_counts += 1
            t.input_tokens += s.input_tokens or 0
            t.output_tokens += s.output_tokens or 0
            if s.cost_usd is None:
                t.calls_without_price += 1
            t.cost_usd += s.cost_usd or 0.0
        t.total_tokens = t.input_tokens + t.output_tokens
        self.totals = t

    def save(self, directory: Path = RUNS_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.run_id}.json"
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
