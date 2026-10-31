"""One record per (topic, system, run index), saved as soon as the run finishes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

RECORDS_DIR = Path("results/benchmark_runs")
JUDGEMENTS_DIR = Path("results/benchmark_judgements")
TRACES_DIR = Path("runs/benchmark")
SYSTEMS = ("baseline", "multi")

Status = Literal["success", "failed", "not_run"]


class CoverageScore(BaseModel):
    """Coverage of the must-cover points; `covered` counts only points whose quote was verified."""

    model: str
    points: list[dict[str, Any]]  # {point, claimed, covered, quote}
    covered: int
    total: int


class CitationScore(BaseModel):
    cited: int  # distinct citation numbers in the report body
    valid: int  # exist in the sources list and were fetched in this run
    invalid_citations: list[int] = Field(default_factory=list)
    uncited_sources: list[int] = Field(default_factory=list)
    unfetched_urls: list[str] = Field(default_factory=list)
    clean: bool = True


class RunRecord(BaseModel):
    topic_id: str
    system: str
    run_index: int
    status: Status
    reason: str = ""
    limit: str | None = None  # limit type behind a failure (llm.limits), if any
    run_id: str = ""
    trace_path: str = ""
    finished_at: str = ""
    report: str = ""
    words: int = 0
    tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    calls_without_token_counts: int = 0
    calls_without_price: int = 0
    latency_s: float = 0.0
    models: list[str] = Field(default_factory=list)
    tokens_by_model: dict[str, list[int]] = Field(default_factory=dict)  # model -> [in, out]
    llm: dict[str, Any] = Field(default_factory=dict)
    citations: CitationScore | None = None
    coverage: CoverageScore | None = None
    scoring_error: str = ""

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.topic_id, self.system, self.run_index)

    @property
    def finished(self) -> bool:
        return self.status in ("success", "failed")


def record_path(directory: Path, topic_id: str, system: str, run_index: int) -> Path:
    return directory / f"{topic_id}__{system}__r{run_index}.json"


def save_record(record: RunRecord, directory: Path = RECORDS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = record_path(directory, *record.key)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_records(directory: Path = RECORDS_DIR) -> dict[tuple[str, str, int], RunRecord]:
    records: dict[tuple[str, str, int], RunRecord] = {}
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        record = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        records[record.key] = record
    return records
