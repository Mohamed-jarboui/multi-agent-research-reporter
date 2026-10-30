"""Data shapes of the multi-agent run: section briefs, worker results and the merged research."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

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


class MergedResearch(BaseModel):
    sources: list[SourceRecord] = Field(default_factory=list)  # global ids 1..K
    pages: dict[str, Page] = Field(default_factory=dict)  # url -> page, for the critic (R2)
    sections: list[SectionOut] = Field(default_factory=list)
    open_questions: list[tuple[str, str]] = Field(default_factory=list)  # (section_id, question)
    incomplete: list[str] = Field(default_factory=list)  # section ids
    dropped_citations: int = 0


class State(TypedDict, total=False):
    topic: str
    sections: list[SectionBrief]
    results: Annotated[list[WorkerResult], operator.add]
    merged: MergedResearch | None
    digest: str


class WorkerInput(TypedDict):
    """The payload of one Send: a worker needs nothing else."""

    brief: SectionBrief
    index: int
