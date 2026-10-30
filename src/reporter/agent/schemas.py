"""Data shapes of the multi-agent run: section briefs, worker results and the merged research."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from pydantic import BaseModel, Field

from reporter.tools import Page
from reporter.trace import SourceRecord

Confidence = Literal["low", "medium", "high"]
Status = Literal["complete", "incomplete"]


class SectionBrief(BaseModel):
    """Everything a worker needs to research one section (with the topic and the date)."""

    id: str
    title: str
    goal: str
    scope: list[str]  # 1-4 points to cover
    exclude: list[str] = Field(default_factory=list)  # what other sections cover
    expected_output: str = "3-6 findings with numbers and dates"


class Finding(BaseModel):
    claim: str
    source_ids: list[
        int
    ]  # worker-local (1-based, fetch order) in a WorkerResult; global after merge


class WorkerResult(BaseModel):
    section_id: str
    index: int
    status: Status
    findings: list[Finding] = Field(default_factory=list)
    pages: list[Page] = Field(default_factory=list)  # fetched pages only, in fetch order
    confidence: Confidence = "low"
    open_questions: list[str] = Field(default_factory=list)
    attempts: int = 1
    stop_reason: str = ""
    error: str | None = None


class SectionOut(BaseModel):
    brief: SectionBrief
    status: Status
    confidence: Confidence
    findings: list[Finding]  # source_ids are global
    attempts: int = 0
    error: str | None = None
    n_pages: int = 0  # pages fetched for this section
    stop_reason: str = ""  # why the worker stopped researching (max_steps, worker_time, ...)


class MergedResearch(BaseModel):
    sources: list[SourceRecord] = Field(default_factory=list)  # global ids 1..K
    pages: dict[str, Page] = Field(default_factory=dict)  # url -> page, for the critic (R2)
    sections: list[SectionOut] = Field(default_factory=list)
    open_questions: list[tuple[str, str]] = Field(default_factory=list)  # (section_id, question)
    incomplete: list[str] = Field(default_factory=list)  # section ids
    dropped_citations: int = 0


Verdict = Literal["pass", "weak", "fail", "unreviewed"]
VERDICT_RANK: dict[str, int] = {"fail": 0, "weak": 1, "unreviewed": 2, "pass": 3}


class Issue(BaseModel):
    rubric: int = 0  # 1 sourced, 2 supported, 3 no contradiction, 4 coverage; 0 = other
    issue: str
    evidence: str = ""
    origin: Literal["deterministic", "critic"] = "critic"


class SectionReview(BaseModel):
    section_id: str
    verdict: Verdict
    issues: list[Issue] = Field(default_factory=list)


class Contradiction(BaseModel):
    section_ids: list[str]
    issue: str
    evidence: str = ""


class Review(BaseModel):
    """What the checks and the critic said about the merged research."""

    sections: dict[str, SectionReview] = Field(default_factory=dict)
    contradictions: list[Contradiction] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    critic_status: Literal["ok", "failed", "skipped"] = "skipped"


class State(TypedDict, total=False):
    topic: str
    sections: list[SectionBrief]
    results: Annotated[list[WorkerResult], operator.add]
    retry_results: Annotated[list[WorkerResult], operator.add]
    merged: MergedResearch | None
    digest: str
    review: Review | None
    retry_ids: list[str]
    report: str


class WorkerInput(TypedDict):
    """The payload of one Send: a worker needs nothing else."""

    brief: SectionBrief
    index: int
    feedback: NotRequired[str]  # critic issues, for a retry
