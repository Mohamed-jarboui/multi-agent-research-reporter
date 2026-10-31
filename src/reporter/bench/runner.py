"""The benchmark loop: resumable, interleaved, quota-aware. See docs/design/r3.md.

One record per (topic, system, run index) is written the moment a run finishes. A re-run skips
finished records. When every provider is out of its per-day quota the loop stops, records the
remaining runs as `not_run` and returns exit code 3.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from reporter.agent.llm_call import LLM
from reporter.agent.parsing import strip_sources_section
from reporter.bench.records import (
    JUDGEMENTS_DIR,
    RECORDS_DIR,
    SYSTEMS,
    TRACES_DIR,
    RunRecord,
    load_records,
    save_record,
)
from reporter.bench.scoring import (
    Judgement,
    judge_run_pair,
    score_citations,
    score_coverage,
    word_count,
)
from reporter.bench.topics import TOPICS_PATH, Topic, TopicsError, load_topics
from reporter.llm.base import ProviderError
from reporter.llm.ping import PingResult
from reporter.trace import Trace

DAILY_LIMITS = frozenset({"per_day_tokens", "per_day_requests"})
EXIT_OK, EXIT_CONFIG, EXIT_QUOTA = 0, 2, 3
MAX_SCORING_FAILURES = 3  # consecutive; scoring is then paused for the rest of the invocation


class Agent(Protocol):
    last_trace_path: Path | None

    def run(self, topic: str) -> Any: ...


AgentFactory = Callable[[str, Path], Agent]  # (system, traces dir) -> a fresh agent and model


@dataclass(frozen=True)
class PlanItem:
    topic: Topic
    system: str
    run_index: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.topic.id, self.system, self.run_index)


@dataclass
class BenchConfig:
    topics: list[Topic]
    systems: tuple[str, ...] = SYSTEMS
    runs_per_topic: int = 2
    max_runs: int | None = None
    records_dir: Path = RECORDS_DIR
    judgements_dir: Path = JUDGEMENTS_DIR
    traces_dir: Path = TRACES_DIR
    seed: int = 0
    swap_check: bool = False
    retry_failed: bool = False
    score: bool = True
    max_consecutive_failures: int = 6


@dataclass
class BenchOutcome:
    exit_code: int
    message: str
    executed: int = 0
    not_run: int = 0


def plan_runs(topics: list[Topic], systems: tuple[str, ...], runs_per_topic: int) -> list[PlanItem]:
    """Topic by topic; for each run index both systems, the first one alternating, so a quota that
    runs out part way does not favour one system."""
    items: list[PlanItem] = []
    for t, topic in enumerate(topics):
        for k in range(1, runs_per_topic + 1):
            order = list(systems)
            if (t + k) % 2 == 0:
                order.reverse()
            items += [PlanItem(topic, system, k) for system in order]
    return items


# ------------------------------------------------------------------ reading a finished trace


def daily_exhausted(llm_summary: dict[str, Any]) -> set[str]:
    """Providers that returned a per-day limit 429 during the run."""
    limits = llm_summary.get("limits") or {}
    return {p for p, kinds in limits.items() if DAILY_LIMITS & set(kinds)}


def dominant_limit(llm_summary: dict[str, Any]) -> str | None:
    """The limit type that explains a failure best: a daily one first, else the most frequent."""
    counts: Counter[str] = Counter()
    for kinds in (llm_summary.get("limits") or {}).values():
        counts.update(kinds)
    for name in sorted(DAILY_LIMITS):
        if counts.get(name):
            return name
    return counts.most_common(1)[0][0] if counts else None


def classify(system: str, trace: Trace | None, error: str) -> tuple[str, str]:
    """('success' | 'failed', reason). Success means a written report: for the baseline at least
    one page was read; for multi the writer wrote it (the digest fallback is a failure)."""
    if error:
        return "failed", error
    if trace is None:
        return "failed", "no trace was saved"
    if not trace.report.strip():
        return "failed", trace.stop_reason or "empty report"
    if system == "multi":
        status = trace.writer.get("status", "")
        if status != "ok":
            return "failed", f"writer {status or 'did not run'}: {trace.writer.get('reason', '')}"
    elif not trace.sources:
        return "failed", f"no page was read ({trace.stop_reason or 'unknown'})"
    return "success", trace.stop_reason


def build_record(item: PlanItem, trace: Trace | None, error: str, trace_path: Path | None):
    status, reason = classify(item.system, trace, error)
    record = RunRecord(
        topic_id=item.topic.id,
        system=item.system,
        run_index=item.run_index,
        status=status,  # type: ignore[arg-type]
        reason=reason[:300],
        finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
        trace_path=str(trace_path) if trace_path else "",
    )
    if trace is None:
        return record
    totals = trace.totals
    by_model: dict[str, list[int]] = {}
    for step in trace.steps:
        if step.model:
            used = by_model.setdefault(step.model, [0, 0])
            used[0] += step.input_tokens or 0
            used[1] += step.output_tokens or 0
    record.run_id = trace.run_id
    record.tokens = totals.total_tokens
    record.input_tokens = totals.input_tokens
    record.output_tokens = totals.output_tokens
    record.cost_usd = totals.cost_usd
    record.calls_without_token_counts = totals.calls_without_token_counts
    record.calls_without_price = totals.calls_without_price
    record.latency_s = totals.latency_s
    record.models = sorted(by_model)
    record.tokens_by_model = by_model
    record.llm = trace.llm
    if status == "failed":
        record.limit = dominant_limit(trace.llm)
    else:
        record.report = trace.report
        record.words = word_count(strip_sources_section(trace.report))
        record.citations = score_citations(trace.report, trace)
    return record


def load_trace(path: Path | None) -> Trace | None:
    if path is None or not path.is_file():
        return None
    return Trace.model_validate_json(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- the loop


@dataclass
class Benchmark:
    config: BenchConfig
    agent_factory: AgentFactory
    scoring_llm: LLM | None = None
    ping_fn: Callable[[], list[PingResult]] | None = None
    log: Callable[[str], None] = print
    exhausted: set[str] = field(default_factory=set)
    known_providers: set[str] = field(default_factory=set)
    _scoring_failures: int = 0

    # ---- planning

    def pending(self) -> list[PlanItem]:
        cfg = self.config
        done = load_records(cfg.records_dir)
        out = []
        for item in plan_runs(cfg.topics, cfg.systems, cfg.runs_per_topic):
            record = done.get(item.key)
            if record is None or record.status == "not_run":
                out.append(item)
            elif record.status == "failed" and cfg.retry_failed:
                out.append(item)
        return out

    # ---- main entry

    def run(self, *, dry_run: bool = False) -> BenchOutcome:
        pending = self.pending()
        total = len(self.config.topics) * len(self.config.systems) * self.config.runs_per_topic
        self.log(
            f"planned {total} runs, {total - len(pending)} already finished, {len(pending)} pending"
        )
        if dry_run:
            for item in pending:
                self.log(f"  would run {item.topic.id} {item.system} r{item.run_index}")
            return BenchOutcome(EXIT_OK, "dry run: nothing was called")

        stop = self._preflight()
        if isinstance(stop, BenchOutcome):
            if stop.exit_code == EXIT_QUOTA:
                stop.not_run = self._mark_not_run(pending, stop.message)
            return stop

        self._catch_up_scoring()
        executed = consecutive_failures = 0
        for position, item in enumerate(pending):
            if self.config.max_runs is not None and executed >= self.config.max_runs:
                return BenchOutcome(EXIT_OK, f"stopped after --max-runs {executed}", executed)
            record = self._execute(item)
            executed += 1
            consecutive_failures = consecutive_failures + 1 if record.status == "failed" else 0
            reason = self._stop_reason(consecutive_failures)
            if reason:
                rest = pending[position + 1 :]
                skipped = self._mark_not_run(rest, reason)
                return BenchOutcome(EXIT_QUOTA, reason, executed, skipped)
        return BenchOutcome(EXIT_OK, f"finished {executed} runs", executed)

    # ---- preflight and stop rule

    def _preflight(self) -> BenchOutcome | None:
        if self.ping_fn is None:
            return None
        results = self.ping_fn()
        for r in results:
            self.log(f"preflight {r.line()}")
        configured = [r for r in results if r.status != "not configured"]
        if not configured:
            return BenchOutcome(
                EXIT_CONFIG, "no provider is configured: set GEMINI_API_KEY and/or GROQ_API_KEY"
            )
        self.known_providers |= {r.name for r in configured}
        self.exhausted |= {r.name for r in configured if not r.ok and r.limit in DAILY_LIMITS}
        if self.exhausted >= self.known_providers:
            return BenchOutcome(
                EXIT_QUOTA,
                "every provider is out of its per-day quota (preflight); try again later",
            )
        return None

    def _stop_reason(self, consecutive_failures: int) -> str:
        if self.known_providers and self.exhausted >= self.known_providers:
            names = ", ".join(sorted(self.exhausted))
            return (
                f"every provider is out of its per-day quota ({names}); try again after the reset"
            )
        if consecutive_failures >= self.config.max_consecutive_failures:
            return f"{consecutive_failures} runs failed in a row; stopping to save quota"
        return ""

    def _mark_not_run(self, items: list[PlanItem], reason: str) -> int:
        """Record unstarted runs as not_run; a finished record (e.g. an earlier failure that
        --retry-failed was about to redo) is kept."""
        done = load_records(self.config.records_dir)
        items = [i for i in items if i.key not in done or not done[i.key].finished]
        for item in items:
            save_record(
                RunRecord(
                    topic_id=item.topic.id,
                    system=item.system,
                    run_index=item.run_index,
                    status="not_run",
                    reason=reason,
                    finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
                ),
                self.config.records_dir,
            )
        return len(items)

    # ---- one run

    def _execute(self, item: PlanItem) -> RunRecord:
        self.log(f"run {item.topic.id} {item.system} r{item.run_index}")
        agent = self.agent_factory(item.system, self.config.traces_dir)
        error = ""
        try:
            agent.run(item.topic.topic)
        except Exception as exc:  # a failed run is a result, not a crash
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
        trace = load_trace(agent.last_trace_path)
        record = build_record(item, trace, error, agent.last_trace_path)
        if trace is not None:
            self.known_providers |= set(trace.llm.get("providers") or [])
            self.exhausted |= daily_exhausted(trace.llm)
        save_record(record, self.config.records_dir)
        self.log(f"  -> {record.status} {record.reason[:80]} ({record.tokens} tokens)")
        self._score(record)
        return record

    # ---- scoring

    def _scoring_on(self) -> bool:
        return (
            self.config.score
            and self.scoring_llm is not None
            and self._scoring_failures < MAX_SCORING_FAILURES
        )

    def _score_failed(self, exc: Exception) -> str:
        self._scoring_failures += 1
        return exc.short() if isinstance(exc, ProviderError) else type(exc).__name__

    def _score(self, record: RunRecord) -> None:
        """Coverage for a successful run, then the judge for its pair if the other side is done."""
        if record.status != "success":
            return
        topic = next(t for t in self.config.topics if t.id == record.topic_id)
        if record.coverage is None and self._scoring_on():
            assert self.scoring_llm is not None
            try:
                record.coverage = score_coverage(self.scoring_llm, topic, record.report)
                record.scoring_error = ""
                self._scoring_failures = 0
            except Exception as exc:
                record.scoring_error = f"coverage: {self._score_failed(exc)}"
            save_record(record, self.config.records_dir)
        self._judge(topic, record.run_index)

    def _judge(self, topic: Topic, run_index: int) -> None:
        cfg = self.config
        path = cfg.judgements_dir / f"{topic.id}__r{run_index}.json"
        if path.exists() or not self._scoring_on() or set(cfg.systems) != set(SYSTEMS):
            return
        records = load_records(cfg.records_dir)
        pair = {s: records.get((topic.id, s, run_index)) for s in SYSTEMS}
        if any(r is None or r.status != "success" for r in pair.values()):
            return
        assert self.scoring_llm is not None
        models = {m for r in pair.values() if r for m in r.models}
        try:
            judgement = judge_run_pair(
                self.scoring_llm,
                topic,
                run_index,
                {s: r.report for s, r in pair.items() if r},
                models,
                seed=cfg.seed,
                swap_check=cfg.swap_check,
            )
        except Exception as exc:
            self.log(f"  judge failed for {topic.id} r{run_index}: {self._score_failed(exc)}")
            return
        self._scoring_failures = 0
        save_judgement(judgement, cfg.judgements_dir)

    def _catch_up_scoring(self) -> None:
        """Score finished runs whose coverage or judgement is missing (an earlier quota stop)."""
        if not self._scoring_on():
            return
        for record in list(load_records(self.config.records_dir).values()):
            if record.status == "success" and record.key[0] in {t.id for t in self.config.topics}:
                self._score(record)


def save_judgement(judgement: Judgement, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{judgement.topic_id}__r{judgement.run_index}.json"
    path.write_text(judgement.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- CLI plumbing


def default_agent_factory(system: str, traces_dir: Path) -> Agent:
    """A fresh model chain per run, so each trace's provider summary covers that run only."""
    from reporter.agent import BaselineAgent, MultiAgent
    from reporter.llm import build_default_llm
    from reporter.tools import default_tools

    llm = build_default_llm()
    if system == "baseline":
        return BaselineAgent(llm, default_tools(), runs_dir=traces_dir)
    return MultiAgent(llm, default_tools(), runs_dir=traces_dir)


def build_scoring_llm(prefer: str = "auto") -> LLM:
    """The judge and coverage model chain. 'auto' tries Groq first because the writers try Gemini
    first, which lowers (but does not remove) self-preference; the judge model is recorded."""
    from reporter.llm.ping import PROVIDER_KEYS, build_providers
    from reporter.llm.router import FallbackLLM

    found = {n: p for n, p in build_providers().items() if p is not None}
    order = [prefer] if prefer in PROVIDER_KEYS else ["groq", "gemini"]
    order += [n for n in ("groq", "gemini") if n not in order]
    providers = [found[n] for n in order if n in found]
    if not providers:
        raise ValueError("Set GEMINI_API_KEY and/or GROQ_API_KEY (see .env.example)")
    return FallbackLLM(providers)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_benchmark", description="Run the R3 benchmark.")
    p.add_argument("--topics-file", type=Path, default=TOPICS_PATH)
    p.add_argument("--topics", nargs="*", help="only these topic ids")
    p.add_argument("--systems", nargs="*", choices=SYSTEMS, default=list(SYSTEMS))
    p.add_argument("--runs-per-topic", type=int, default=2)
    p.add_argument("--max-runs", type=int, help="stop after this many runs in this invocation")
    p.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--judgements-dir", type=Path, default=JUDGEMENTS_DIR)
    p.add_argument("--traces-dir", type=Path, default=TRACES_DIR)
    p.add_argument("--seed", type=int, default=0, help="seed for the judge's A/B order")
    p.add_argument("--swap-check", action="store_true", help="judge twice with A/B swapped")
    p.add_argument("--judge-provider", choices=["auto", "gemini", "groq"], default="auto")
    p.add_argument("--no-score", action="store_true", help="skip coverage and the judge")
    p.add_argument("--retry-failed", action="store_true", help="redo runs recorded as failed")
    p.add_argument("--max-consecutive-failures", type=int, default=6)
    p.add_argument("--no-preflight", action="store_true", help="skip the quota ping")
    p.add_argument("--dry-run", action="store_true", help="print the plan, call nothing")
    return p


def main(argv: list[str] | None = None) -> int:
    from reporter.cli import load_env_file

    args = build_parser().parse_args(argv)
    load_env_file()
    sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
    try:
        topics = load_topics(args.topics_file, expected=None if args.topics else 12)
        if args.topics:
            unknown = set(args.topics) - {t.id for t in topics}
            if unknown:
                raise TopicsError(f"unknown topic ids: {', '.join(sorted(unknown))}")
            topics = [t for t in topics if t.id in set(args.topics)]
    except (TopicsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CONFIG
    config = BenchConfig(
        topics=topics,
        systems=tuple(args.systems),
        runs_per_topic=args.runs_per_topic,
        max_runs=args.max_runs,
        records_dir=args.records_dir,
        judgements_dir=args.judgements_dir,
        traces_dir=args.traces_dir,
        seed=args.seed,
        swap_check=args.swap_check,
        retry_failed=args.retry_failed,
        score=not args.no_score,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    scoring_llm: LLM | None = None
    ping_fn = None
    if not args.dry_run:
        from reporter.llm.ping import build_providers, ping_all

        if not args.no_preflight:
            ping_fn = lambda: ping_all(build_providers())  # noqa: E731
        if config.score:
            try:
                scoring_llm = build_scoring_llm(args.judge_provider)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return EXIT_CONFIG
    bench = Benchmark(config, default_agent_factory, scoring_llm, ping_fn)
    outcome = bench.run(dry_run=args.dry_run)
    print(f"\n{outcome.message} (executed {outcome.executed}, not_run {outcome.not_run})")
    if outcome.exit_code == EXIT_QUOTA:
        print(
            "quota stop: re-run the same command later; finished runs are skipped.", file=sys.stderr
        )
    return outcome.exit_code


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
