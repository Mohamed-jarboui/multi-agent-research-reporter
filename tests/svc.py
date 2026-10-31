"""Helpers for the service tests: a fake UTC clock and an app wired to fake models and tools."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from fakes import FakeClock, TeamLLM, team_tools
from reporter.service.app import create_app
from reporter.service.config import Settings
from reporter.service.runner import CrashHook
from reporter.service.store import JobStore

TOPIC = {"topic": "state of solid-state batteries"}
KEY = "key-0123456789"


class FakeNow:
    """A UTC clock that only moves when told to."""

    def __init__(self, start: datetime | None = None) -> None:
        self.value = start or datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def crash_after(key: str):
    """hooks_factory: die (in-process) right after the checkpoint `key`, once per job."""

    def factory(job_id, checkpoints):
        store = checkpoints._store
        return CrashHook(store, job_id, checkpoints, key, mode="raise")

    return factory


def make_service(tmp_path, *, make_llm=None, make_tools=None, now=None, **overrides) -> Any:
    """An app (no background thread: tests call `runner.drain()`) on `tmp_path/data`."""
    values: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "storage": "persistent",
        "daily_max_jobs": 50,
        "daily_max_tokens": 50_000_000,
        "max_parallel_workers": 2,
        "post_burst": 1000,
        "reads_per_minute": 100_000,
    }
    values.update({k: v for k, v in overrides.items() if k in Settings.__dataclass_fields__})
    settings = Settings(**values)
    clock = now or FakeNow()
    llms: list[Any] = []
    agent_clock = FakeClock()  # model cooldowns advance this, not real time

    def llm_factory():
        llm = (make_llm or (lambda: TeamLLM(3)))()
        llms.append(llm)
        return llm

    app = create_app(
        settings,
        make_llm=llm_factory,
        make_tools=make_tools or team_tools,
        now=clock,
        hooks_factory=overrides.get("hooks_factory"),
        sleep=overrides.get("sleep") or _no_sleep,
        monotonic=overrides.get("monotonic") or clock_seconds(clock),
        agent_clock=agent_clock,
        agent_sleep=agent_clock.sleep,
        run_thread=overrides.get("run_thread", False),
        tracing=False,
    )
    store: JobStore = app.state.store
    return SimpleNamespace(
        settings=settings,
        app=app,
        client=TestClient(app),
        store=store,
        runner=app.state.runner,
        now=clock,
        llms=llms,
    )


async def _no_sleep(seconds: float) -> None:
    return None


def clock_seconds(now: FakeNow):
    return lambda: now().timestamp()
