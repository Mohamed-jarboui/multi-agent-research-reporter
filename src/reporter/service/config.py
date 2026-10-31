"""Service settings, read from the environment. The numbers are demo starting values, not
measured ones."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from reporter.budget import Budget, MultiBudget

log = logging.getLogger("reporter.service")

VERSION = "0.1.0"
SECRET_ENV_NAMES = ("GEMINI_API_KEY", "GROQ_API_KEY", "TAVILY_API_KEY")


def resolve_data_dir(
    environ: Mapping[str, str],
    persistent: Path = Path("/data"),
    fallback: Path = Path("var"),
) -> tuple[Path, str]:
    """(folder, "persistent" | "ephemeral"): `REPORTER_DATA_DIR`, else `/data` when it can be
    written, else `./var` with a warning (a restart there may lose the jobs)."""
    chosen = environ.get("REPORTER_DATA_DIR", "").strip()
    if chosen:
        return Path(chosen), "persistent"
    if persistent.is_dir() and os.access(persistent, os.W_OK):
        return persistent, "persistent"
    log.warning(
        "no persistent storage at %s: using %s; jobs may be lost on restart", persistent, fallback
    )
    return fallback, "ephemeral"


def _int(environ: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(environ.get(name, default))
    except ValueError:
        return default


def _float(environ: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("var")
    storage: str = "ephemeral"
    # per-run budget (demo defaults)
    run_max_steps: int = 40
    run_max_tokens: int = 150_000
    run_max_seconds: float = 360.0
    run_max_cost_usd: float = 0.05
    max_parallel_workers: int = 2
    max_parallel_llm_calls: int = 2
    # daily cap, UTC day
    daily_max_jobs: int = 2
    daily_max_tokens: int = 400_000
    min_token_fraction: float = 0.2  # tokens left today needed to admit a job
    # admission
    max_waiting: int = 3
    max_resumes: int = 2
    topic_min_chars: int = 5
    topic_max_chars: int = 300
    max_body_bytes: int = 4096
    max_streams: int = 20
    post_burst: int = 3
    post_refill_s: float = 300.0
    reads_per_minute: int = 60
    trust_proxy_hops: int = 1
    cors_origins: tuple[str, ...] = ()
    # SSE
    sse_poll_s: float = 1.0
    sse_heartbeat_s: float = 15.0
    # retention
    idempotency_ttl_s: float = 24 * 3600.0
    job_retention_s: float = 7 * 24 * 3600.0
    # test hooks (only honoured when test_hooks is True)
    test_hooks: bool = False
    crash_after: str = ""

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reporter.db"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    def multi_budget(self) -> MultiBudget:
        run = Budget(
            max_steps=self.run_max_steps,
            max_tokens=self.run_max_tokens,
            max_seconds=self.run_max_seconds,
            max_cost_usd=self.run_max_cost_usd,
            stop_fraction=MultiBudget().run.stop_fraction,
        )
        return MultiBudget(
            run=run,
            max_parallel_workers=self.max_parallel_workers,
            max_parallel_llm_calls=self.max_parallel_llm_calls,
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if environ is None else environ
        data_dir, storage = resolve_data_dir(env)
        defaults = cls()
        hooks = env.get("REPORTER_TEST_HOOKS") == "1"
        origins = tuple(o.strip() for o in env.get("CORS_ORIGINS", "").split(",") if o.strip())
        return cls(
            data_dir=data_dir,
            storage=storage,
            run_max_steps=_int(env, "RUN_MAX_STEPS", defaults.run_max_steps),
            run_max_tokens=_int(env, "RUN_MAX_TOKENS", defaults.run_max_tokens),
            run_max_seconds=_float(env, "RUN_MAX_SECONDS", defaults.run_max_seconds),
            run_max_cost_usd=_float(env, "RUN_MAX_COST_USD", defaults.run_max_cost_usd),
            max_parallel_workers=_int(env, "MAX_PARALLEL_WORKERS", defaults.max_parallel_workers),
            daily_max_jobs=_int(env, "DAILY_MAX_JOBS", defaults.daily_max_jobs),
            daily_max_tokens=_int(env, "DAILY_MAX_TOKENS", defaults.daily_max_tokens),
            max_waiting=_int(env, "MAX_WAITING", defaults.max_waiting),
            trust_proxy_hops=_int(env, "TRUST_PROXY_HOPS", defaults.trust_proxy_hops),
            cors_origins=origins,
            test_hooks=hooks,
            crash_after=env.get("REPORTER_CRASH_AFTER", "") if hooks else "",
        )
