"""The runner: one thread, one job at a time. It takes jobs from the SQLite queue, resumes the
jobs a crash left running, runs the agent with checkpoints, and writes the outcome.

Logs carry job ids, states and reason codes only; a raw provider error never reaches a log, an
event or an API answer.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from typing import Any

from reporter.agent.checkpoint import JobCancelled, RunHooks
from reporter.agent.llm_call import LLM
from reporter.agent.multi import MultiAgent, MultiRunResult
from reporter.service.config import Settings
from reporter.service.quota import provider_quota_exhausted, retry_delay_s
from reporter.service.store import (
    InvalidTransition,
    JobCheckpoints,
    JobStore,
    SimulatedCrash,
    iso,
    next_midnight,
)
from reporter.tools import Tool

log = logging.getLogger("reporter.service.runner")

CRASH_EXIT_CODE = 75
PURGE_EVERY_S = 3600.0
IDLE_WAKE_S = 2.0

# reason code -> the only text a client ever sees about it
MESSAGES: dict[str, str] = {
    "provider_quota": "The free model quota is used up for now; try again later.",
    "daily_cap": "The daily limit of this demo is reached; try again after 00:00 UTC.",
    "budget_exhausted": "The run budget ran out before a full report could be written.",
    "writer_failed": "The research finished but the writer step failed; a digest is returned.",
    "no_report": "No section had a finding backed by a fetched page, so no report was written.",
    "crashed_repeatedly": "The job was interrupted too many times and was stopped.",
    "internal_error": "The job failed because of an internal error.",
    "cancelled_by_user": "The job was cancelled.",
}

HooksFactory = Callable[[str, JobCheckpoints], RunHooks | None]


class CrashHook:
    """Test hook: dies right after the checkpoint of `after_key`, once per job.

    `claim_crash_hook` is set in the database before the death, so the restarted process does
    not die again. `mode="exit"` is a real process death (os._exit); `mode="raise"` is for
    in-process tests: it freezes the job's store and raises SimulatedCrash.
    """

    def __init__(
        self,
        store: JobStore,
        job_id: str,
        checkpoints: JobCheckpoints,
        after_key: str,
        mode: str = "exit",
    ) -> None:
        self._store, self._job_id, self._checkpoints = store, job_id, checkpoints
        self._after_key, self._mode = after_key, mode

    def after_checkpoint(self, key: str) -> None:
        if key != self._after_key or not self._store.claim_crash_hook(self._job_id):
            return
        if self._mode == "exit":
            os._exit(CRASH_EXIT_CODE)
        self._checkpoints.freeze()
        raise SimulatedCrash(key)


def build_result(run: MultiRunResult) -> dict[str, Any]:
    """The public result of a finished job."""
    review, trace = run.review, run.trace
    sections = []
    for section in run.merged.sections:
        sid = section.brief.id
        verdict = review.sections.get(sid) if review else None
        sections.append(
            {
                "id": sid,
                "title": section.brief.title,
                "status": section.status,
                "verdict": verdict.verdict if verdict else "unreviewed",
                "issue_count": len(verdict.issues) if verdict else 0,
            }
        )
    checks = trace.checks
    return {
        "report_markdown": run.report,
        "sources": [{"id": s.id, "title": s.title, "url": s.url} for s in run.merged.sources],
        "citation_check": {
            name: checks.get(name, [])
            for name in ("invalid_citations", "uncited_sources", "unfetched_urls")
        },
        "sections": sections,
        "writer_status": str(trace.writer.get("status", "")),
    }


def build_usage(run: MultiRunResult, agent: MultiAgent) -> dict[str, Any]:
    """Budget and trace summary. The cost is None when a call was unpriced (never a made-up 0)."""
    trace, totals = run.trace, run.trace.totals
    meter = agent.meter.snapshot()
    return {
        "budget": {
            "limit": asdict(agent.budget.run),
            "used": {
                "steps": int(meter["steps"]),
                "tokens": int(meter["tokens"]),
                "cost_usd": round(meter["cost_usd"], 6),
                "seconds": round(meter["elapsed_s"], 1),
            },
        },
        "trace": {
            "llm_calls": totals.llm_calls,
            "total_tokens": totals.total_tokens,
            "cost_usd": None if totals.calls_without_price else round(totals.cost_usd, 6),
            "latency_s": round(totals.latency_s, 1),
            "models": sorted({s.model for s in trace.steps if s.model}),
            "resumes": len(trace.resumes),
        },
    }


class Runner:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        *,
        make_llm: Callable[[], LLM],
        make_tools: Callable[[], dict[str, Tool]],
        now: Callable[[], datetime] | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        hooks_factory: HooksFactory | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self._make_llm, self._make_tools = make_llm, make_tools
        self._now = now or (lambda: datetime.now(UTC))
        self._clock, self._sleep = clock, sleep
        self._hooks_factory = hooks_factory
        self._resume_queue: list[str] = []
        self.processed: list[str] = []  # job ids in the order they were run (a testing aid)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ thread

    def start(self) -> None:
        self.startup()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="runner", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        last_purge = time.monotonic()
        while not self._stop.is_set():
            worked = False
            try:
                worked = self.step()
                if time.monotonic() - last_purge > PURGE_EVERY_S:
                    self.store.purge()
                    last_purge = time.monotonic()
            except Exception as exc:  # the loop must survive anything a job does
                log.error("runner step failed: %s", type(exc).__name__)
            if not worked:
                self._wake.wait(IDLE_WAKE_S)
                self._wake.clear()

    # ------------------------------------------------------------------ startup

    def startup(self) -> None:
        """Purge old rows, then decide for each job a crash left running: resume it (before any
        queued job) or, after `max_resumes` restarts, end it as failed."""
        self.store.purge()
        for job in self.store.running_jobs():
            if job["resume_count"] >= self.settings.max_resumes:
                self.store.finish(
                    job["id"],
                    "failed",
                    "crashed_repeatedly",
                    events=[("failed", {"state": "failed", "reason": "crashed_repeatedly"})],
                )
                log.warning("job %s failed: crashed_repeatedly", job["id"])
                continue
            self.store.mark_resumed(job["id"])
            self._resume_queue.append(job["id"])
            log.info("job %s will resume", job["id"])

    # ------------------------------------------------------------------ one step

    def drain(self) -> None:
        """Run until there is nothing left to do (used by tests; the thread uses `step`)."""
        while self.step():
            pass

    def step(self) -> bool:
        """Run one job (a resumed one first). Returns False when there is none."""
        while self._resume_queue:
            job = self.store.get_job(self._resume_queue.pop(0))
            if job is not None and job["state"] == "running":
                self._run(job, resumed=True)
                return True
        job = self.store.next_queued()
        if job is None:
            return False
        reason = self._start_refusal()
        if reason is not None:
            self._end_without_running(job["id"], reason)
            return True
        try:
            self.store.start_job(job["id"])
        except InvalidTransition:  # cancelled in the meantime
            return True
        self._run(self.store.get_job(job["id"]) or job, resumed=False)
        return True

    def _start_refusal(self) -> str | None:
        """Why the next job may not start (the cap or the latch may have changed in the queue)."""
        s = self.settings
        usage = self.store.usage_today()
        left = s.daily_max_tokens - usage["tokens"]
        if (
            usage["jobs_started"] >= s.daily_max_jobs
            or left < s.min_token_fraction * s.run_max_tokens
        ):
            return "daily_cap"
        if self.store.latch_until() is not None:
            return "provider_quota"
        return None

    def _end_without_running(self, job_id: str, reason: str) -> None:
        if reason == "daily_cap":
            data = {
                "kind": "daily_cap",
                "limit_type": "jobs_or_tokens",
                "resets_at": iso(next_midnight(self._now())),
            }
        else:
            until = self.store.latch_until()
            data = {
                "kind": "provider_limit",
                "limit_type": "provider_quota",
                "resets_at": iso(until) if until else None,
            }
        events = [("quota", data), ("done", {"state": "degraded", "reason": reason})]
        self.store.finish(job_id, "degraded", reason, events=events)
        self.processed.append(job_id)
        log.info("job %s degraded: %s (no model call)", job_id, reason)

    # ------------------------------------------------------------------ run a job

    def _budget_for(self, checkpoints: JobCheckpoints, resumed: bool):
        s = self.settings
        multi = s.multi_budget()
        used_here = 0
        if resumed and (snap := checkpoints.load_snapshot()) is not None:
            used_here = int(snap.meter.get("tokens", 0))
        left_today = s.daily_max_tokens - self.store.usage_today()["tokens"] + used_here
        tokens = max(1, min(s.run_max_tokens, left_today))
        return replace(multi, run=replace(multi.run, max_tokens=tokens))

    def _run(self, job: dict[str, Any], *, resumed: bool) -> None:
        job_id = job["id"]
        checkpoints = JobCheckpoints(self.store, job_id)
        agent = MultiAgent(
            self._make_llm(),  # a new router per job: its summary is per instance
            self._make_tools(),
            self._budget_for(checkpoints, resumed),
            clock=self._clock,
            sleep=self._sleep,
            runs_dir=self.settings.runs_dir,
            on_call=self.store.add_usage,
        )
        hooks = self._hooks(job_id, checkpoints)
        self.processed.append(job_id)
        log.info("job %s running (resumed=%s)", job_id, resumed)
        try:
            run = agent.run(job["topic"], run_id=job_id, store=checkpoints, hooks=hooks)
        except SimulatedCrash:
            checkpoints.freeze()
            return  # the job stays 'running', as after a real process death
        except JobCancelled:
            self.store.finish(
                job_id,
                "cancelled",
                "cancelled_by_user",
                events=[("failed", {"state": "cancelled", "reason": "cancelled_by_user"})],
            )
            log.info("job %s cancelled", job_id)
            return
        except Exception as exc:
            self.store.finish(
                job_id,
                "failed",
                "internal_error",
                events=[("failed", {"state": "failed", "reason": "internal_error"})],
            )
            log.error("job %s failed: internal_error (%s)", job_id, type(exc).__name__)
            return
        self._complete(job_id, run, agent)

    def _hooks(self, job_id: str, checkpoints: JobCheckpoints) -> RunHooks | None:
        if self._hooks_factory is not None:
            return self._hooks_factory(job_id, checkpoints)
        if self.settings.test_hooks and self.settings.crash_after:
            return CrashHook(self.store, job_id, checkpoints, self.settings.crash_after)
        return None

    def _complete(self, job_id: str, run: MultiRunResult, agent: MultiAgent) -> None:
        writer_status = str(run.trace.writer.get("status", ""))
        events: list[tuple[str, dict[str, Any]]] = []
        if writer_status == "ok":
            state, reason = "succeeded", None
        elif provider_quota_exhausted(run.trace.llm):
            state, reason = "degraded", "provider_quota"
            events.append(("quota", self._latch(run, agent)))
        elif agent.meter.hard_limit_reached():
            state, reason = "degraded", "budget_exhausted"
        elif writer_status == "fallback":
            state, reason = "degraded", "writer_failed"
        else:
            state, reason = "degraded", "no_report"
        events.append(("done", {"state": state, "reason": reason}))
        self.store.finish(
            job_id,
            state,
            reason,
            result=build_result(run),
            usage=build_usage(run, agent),
            events=events,
        )
        log.info("job %s %s (%s)", job_id, state, reason or "ok")

    def _latch(self, run: MultiRunResult, agent: MultiAgent) -> dict[str, Any]:
        """Remember that every provider is out of quota, so no new job starts a doomed run."""
        failures = getattr(agent.llm, "failures", [])
        until = self._now() + timedelta(seconds=retry_delay_s(failures))
        self.store.set_latch(until, "provider_quota")
        limits = run.trace.llm.get("limits", {})
        kinds = sorted({k for per in limits.values() for k in per})
        return {
            "kind": "provider_limit",
            "limit_type": kinds[0] if kinds else "unknown",
            "resets_at": iso(until),
        }
