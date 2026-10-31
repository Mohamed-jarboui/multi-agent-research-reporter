"""SQLite store: jobs, idempotency keys, checkpoints, the snapshot, events and the daily counters.

Standard library only. One write connection behind a lock (WAL mode, so readers never wait for
it); every read opens its own short-lived connection. A write transaction is only ever held for a
few statements and never across a model call.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from reporter.agent.checkpoint import Snapshot

TERMINAL = ("succeeded", "degraded", "failed", "cancelled")
# state -> the states it may be entered from; anything else is refused in SQL
ALLOWED_FROM: dict[str, tuple[str, ...]] = {
    "running": ("queued",),
    "succeeded": ("running",),
    "degraded": ("queued", "running"),
    "failed": ("running",),
    "cancelled": ("queued", "running"),
}
EVENT_TYPES = (
    "queued",
    "started",
    "plan",
    "worker_done",
    "critic_verdict",
    "retry",
    "written",
    "quota",
    "done",
    "failed",
)
TERMINAL_EVENTS = ("done", "failed")
ISO = "%Y-%m-%dT%H:%M:%SZ"

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY, state TEXT NOT NULL, reason TEXT, topic TEXT NOT NULL,
  body_hash TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
  resume_count INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
  crash_hook_fired INTEGER NOT NULL DEFAULT 0, result_json TEXT, usage_json TEXT);
CREATE TABLE IF NOT EXISTS idempotency(
  key_hash TEXT PRIMARY KEY, body_hash TEXT NOT NULL, job_id TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS checkpoints(
  job_id TEXT NOT NULL, key TEXT NOT NULL, output_json TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(job_id, key));
CREATE TABLE IF NOT EXISTS snapshot(
  job_id TEXT PRIMARY KEY, seq INTEGER NOT NULL, last_key TEXT NOT NULL, trace_json TEXT NOT NULL,
  meter_json TEXT NOT NULL, llm_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  job_id TEXT NOT NULL, seq INTEGER NOT NULL, type TEXT NOT NULL, data_json TEXT NOT NULL,
  created_at TEXT NOT NULL, PRIMARY KEY(job_id, seq));
CREATE TABLE IF NOT EXISTS daily_usage(
  day TEXT PRIMARY KEY, jobs_started INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0,
  llm_calls INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS quota_latch(
  id INTEGER PRIMARY KEY CHECK(id = 1), until TEXT NOT NULL, reason TEXT NOT NULL);
"""


class InvalidTransition(Exception):
    """A job tried to leave a state it cannot leave (e.g. a terminal one)."""


class Conflict(Exception):
    """The request does not fit the job's state (e.g. cancelling a finished job)."""


class SimulatedCrash(BaseException):
    """Raised by test hooks in place of a process death; not an Exception, so nothing in the agent
    swallows it."""


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(ISO)


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text, ISO).replace(tzinfo=UTC)


def utc_day(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%d")


def next_midnight(moment: datetime) -> datetime:
    day = moment.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return day + timedelta(days=1)


def new_job_id(moment: datetime) -> str:
    """rpt_<UTC timestamp>_<16 hex>: 64 random bits, so an id cannot be guessed."""
    return f"rpt_{moment.astimezone(UTC).strftime('%Y%m%dT%H%M%SZ')}_{secrets.token_hex(8)}"


@dataclass(frozen=True)
class Admission:
    kind: str  # created | replay | conflict | rejected
    job_id: str = ""
    state: str = ""
    code: str = ""  # rejection code: provider_quota | daily_cap | queue_full
    retry_after_s: int = 0


class JobStore:
    def __init__(self, path: Path, now: Callable[[], datetime] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now or (lambda: datetime.now(UTC))
        self._lock = threading.RLock()
        self._w = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None, timeout=5.0
        )
        self._w.execute("PRAGMA journal_mode=WAL")
        self._w.execute("PRAGMA busy_timeout=5000")
        self._w.execute("PRAGMA synchronous=NORMAL")
        self._w.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._w.close()

    # ------------------------------------------------------------------ plumbing

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._w.execute("BEGIN IMMEDIATE")
            try:
                yield self._w
            except BaseException:
                self._w.execute("ROLLBACK")
                raise
            self._w.execute("COMMIT")

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()

    def _append(self, c: sqlite3.Connection, job_id: str, events: list[tuple[str, dict]]) -> None:
        """Add events with the next seq numbers; the write lock makes the order total."""
        (seq,) = c.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM events WHERE job_id = ?", (job_id,)
        ).fetchone()
        stamp = iso(self._now())
        for kind, data in events:
            seq += 1
            c.execute(
                "INSERT INTO events(job_id, seq, type, data_json, created_at) VALUES(?,?,?,?,?)",
                (job_id, seq, kind, json.dumps(data), stamp),
            )

    # ------------------------------------------------------------------ admission

    def admit(
        self,
        *,
        topic: str,
        body_hash: str,
        key_hash: str | None,
        max_waiting: int,
        daily_max_jobs: int,
        daily_max_tokens: int,
        min_tokens: int,
        idempotency_ttl_s: float,
    ) -> Admission:
        """Everything POST /reports decides after validation, in one transaction (so two
        requests cannot both take the last place): replay, quota latch, daily cap, queue room,
        then the insert of the job, its key and its `queued` event."""
        now = self._now()
        with self._tx() as c:
            if key_hash:
                row = c.execute(
                    "SELECT body_hash, job_id, created_at FROM idempotency WHERE key_hash = ?",
                    (key_hash,),
                ).fetchone()
                if row is not None:
                    age = (now - parse_iso(row[2])).total_seconds()
                    if age <= idempotency_ttl_s:
                        state = c.execute(
                            "SELECT state FROM jobs WHERE id = ?", (row[1],)
                        ).fetchone()
                        kind = "replay" if row[0] == body_hash else "conflict"
                        return Admission(kind, row[1], state[0] if state else "")
                    c.execute("DELETE FROM idempotency WHERE key_hash = ?", (key_hash,))
            latch = c.execute("SELECT until FROM quota_latch WHERE id = 1").fetchone()
            if latch is not None and parse_iso(latch[0]) > now:
                wait = int((parse_iso(latch[0]) - now).total_seconds()) + 1
                return Admission("rejected", code="provider_quota", retry_after_s=wait)
            day = utc_day(now)
            used = c.execute(
                "SELECT jobs_started, tokens FROM daily_usage WHERE day = ?", (day,)
            ).fetchone() or (0, 0)
            waiting = c.execute("SELECT COUNT(*) FROM jobs WHERE state = 'queued'").fetchone()[0]
            until_midnight = int((next_midnight(now) - now).total_seconds()) + 1
            # a running job is already in jobs_started, so only waiting jobs are added
            if used[0] + waiting >= daily_max_jobs or daily_max_tokens - used[1] < min_tokens:
                return Admission("rejected", code="daily_cap", retry_after_s=until_midnight)
            if waiting >= max_waiting:
                return Admission("rejected", code="queue_full", retry_after_s=60)
            job_id = new_job_id(now)
            c.execute(
                "INSERT INTO jobs(id, state, topic, body_hash, created_at) VALUES(?,?,?,?,?)",
                (job_id, "queued", topic, body_hash, iso(now)),
            )
            if key_hash:
                c.execute(
                    "INSERT INTO idempotency(key_hash, body_hash, job_id, created_at) "
                    "VALUES(?,?,?,?)",
                    (key_hash, body_hash, job_id, iso(now)),
                )
            self._append(c, job_id, [("queued", {"position": waiting + 1})])
            return Admission("created", job_id, "queued")

    # ------------------------------------------------------------------ jobs

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._read() as c:
            row = c.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def queue_position(self, job_id: str) -> int:
        with self._read() as c:
            (n,) = c.execute(
                "SELECT COUNT(*) FROM jobs WHERE state = 'queued' AND rowid <= "
                "(SELECT rowid FROM jobs WHERE id = ?)",
                (job_id,),
            ).fetchone()
        return int(n)

    def counts(self) -> tuple[int, int]:
        """(running, waiting)."""
        with self._read() as c:
            rows = dict(
                c.execute(
                    "SELECT state, COUNT(*) FROM jobs WHERE state IN ('queued','running') "
                    "GROUP BY state"
                ).fetchall()
            )
        return int(rows.get("running", 0)), int(rows.get("queued", 0))

    def next_queued(self) -> dict[str, Any] | None:
        with self._read() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY rowid LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def running_jobs(self) -> list[dict[str, Any]]:
        with self._read() as c:
            rows = c.execute("SELECT * FROM jobs WHERE state = 'running' ORDER BY rowid").fetchall()
        return [dict(r) for r in rows]

    def start_job(self, job_id: str) -> None:
        """queued -> running; the daily job counter goes up and `started` is announced."""
        now = self._now()
        with self._tx() as c:
            changed = c.execute(
                "UPDATE jobs SET state = 'running', started_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (iso(now), job_id),
            ).rowcount
            if not changed:
                raise InvalidTransition(f"{job_id} is not queued")
            c.execute(
                "INSERT INTO daily_usage(day, jobs_started) VALUES(?, 1) "
                "ON CONFLICT(day) DO UPDATE SET jobs_started = jobs_started + 1",
                (utc_day(now),),
            )
            self._append(c, job_id, [("started", {"resumed": False, "from_key": None})])

    def mark_resumed(self, job_id: str) -> int:
        """A running job restarts after a crash: resume_count + 1 and a `started` event."""
        with self._tx() as c:
            changed = c.execute(
                "UPDATE jobs SET resume_count = resume_count + 1 "
                "WHERE id = ? AND state = 'running'",
                (job_id,),
            ).rowcount
            if not changed:
                raise InvalidTransition(f"{job_id} is not running")
            snap = c.execute("SELECT last_key FROM snapshot WHERE job_id = ?", (job_id,)).fetchone()
            (count,) = c.execute("SELECT resume_count FROM jobs WHERE id = ?", (job_id,)).fetchone()
            data = {"resumed": True, "from_key": snap[0] if snap else None}
            self._append(c, job_id, [("started", data)])
        return int(count)

    def finish(
        self,
        job_id: str,
        state: str,
        reason: str | None,
        *,
        result: dict | None = None,
        usage: dict | None = None,
        events: list[tuple[str, dict]] | None = None,
    ) -> None:
        """Move a job to a terminal state, drop its checkpoints, add the final events. A job that
        is already terminal cannot change: the UPDATE matches no row and this raises."""
        with self._tx() as c:
            self._finish(c, job_id, state, reason, result, usage, events or [])

    def _finish(self, c, job_id, state, reason, result, usage, events) -> None:
        allowed = ALLOWED_FROM[state]
        marks = ",".join("?" for _ in allowed)
        changed = c.execute(
            "UPDATE jobs SET state = ?, reason = ?, finished_at = ?, result_json = ?, "
            f"usage_json = ? WHERE id = ? AND state IN ({marks})",
            (
                state,
                reason,
                iso(self._now()),
                json.dumps(result) if result is not None else None,
                json.dumps(usage) if usage is not None else None,
                job_id,
                *allowed,
            ),
        ).rowcount
        if not changed:
            raise InvalidTransition(f"{job_id} cannot become {state}")
        c.execute("DELETE FROM checkpoints WHERE job_id = ?", (job_id,))
        c.execute("DELETE FROM snapshot WHERE job_id = ?", (job_id,))
        self._append(c, job_id, events)

    def request_cancel(self, job_id: str) -> str:
        """DELETE /reports/{id}: 'cancelled' for a queued job, 'requested' for a running one
        (it stops at the next node). Raises Conflict for a finished job, KeyError if unknown."""
        with self._tx() as c:
            row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row[0] == "queued":
                events = [("failed", {"state": "cancelled", "reason": "cancelled_by_user"})]
                self._finish(c, job_id, "cancelled", "cancelled_by_user", None, None, events)
                return "cancelled"
            if row[0] == "running":
                c.execute("UPDATE jobs SET cancel_requested = 1 WHERE id = ?", (job_id,))
                return "requested"
            raise Conflict(row[0])

    def cancel_requested(self, job_id: str) -> bool:
        with self._read() as c:
            row = c.execute("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return bool(row and row[0])

    def claim_crash_hook(self, job_id: str) -> bool:
        """True exactly once per job: the test hook may fire only the first time."""
        with self._tx() as c:
            return bool(
                c.execute(
                    "UPDATE jobs SET crash_hook_fired = 1 WHERE id = ? AND crash_hook_fired = 0",
                    (job_id,),
                ).rowcount
            )

    # ------------------------------------------------------------------ checkpoints

    def load_checkpoint(self, job_id: str, key: str) -> str | None:
        with self._read() as c:
            row = c.execute(
                "SELECT output_json FROM checkpoints WHERE job_id = ? AND key = ?", (job_id, key)
            ).fetchone()
        return row[0] if row else None

    def load_snapshot(self, job_id: str) -> Snapshot | None:
        with self._read() as c:
            row = c.execute(
                "SELECT last_key, trace_json, meter_json, llm_json FROM snapshot WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return Snapshot(row[1], json.loads(row[2]), json.loads(row[3]), row[0])

    def commit_checkpoint(
        self,
        job_id: str,
        key: str,
        output_json: str,
        snapshot: Callable[[], Snapshot],
        events: list[tuple[str, dict]],
    ) -> None:
        """Output, snapshot and events in one transaction. A job that is no longer running
        (cancelled, finished) saves nothing."""
        with self._tx() as c:
            row = c.execute("SELECT state FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None or row[0] != "running":
                return
            snap = snapshot()  # built under the lock, so a later checkpoint is never overwritten
            c.execute(
                "INSERT OR REPLACE INTO checkpoints(job_id, key, output_json, created_at) "
                "VALUES(?,?,?,?)",
                (job_id, key, output_json, iso(self._now())),
            )
            (seq,) = c.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM snapshot WHERE job_id = ?", (job_id,)
            ).fetchone()
            c.execute(
                "INSERT OR REPLACE INTO snapshot(job_id, seq, last_key, trace_json, meter_json, "
                "llm_json) VALUES(?,?,?,?,?,?)",
                (
                    job_id,
                    seq,
                    snap.last_key,
                    snap.trace_json,
                    json.dumps(snap.meter),
                    json.dumps(snap.llm),
                ),
            )
            self._append(c, job_id, events)

    # ------------------------------------------------------------------ events

    def events_after(self, job_id: str, after: int, limit: int = 200) -> list[dict[str, Any]]:
        with self._read() as c:
            rows = c.execute(
                "SELECT seq, type, data_json FROM events WHERE job_id = ? AND seq > ? "
                "ORDER BY seq LIMIT ?",
                (job_id, after, limit),
            ).fetchall()
        return [{"seq": r[0], "type": r[1], "data": json.loads(r[2])} for r in rows]

    def add_event(self, job_id: str, kind: str, data: dict) -> None:
        with self._tx() as c:
            self._append(c, job_id, [(kind, data)])

    # ------------------------------------------------------------------ daily usage, latch

    def usage_today(self) -> dict[str, int]:
        with self._read() as c:
            row = c.execute(
                "SELECT jobs_started, tokens, llm_calls FROM daily_usage WHERE day = ?",
                (utc_day(self._now()),),
            ).fetchone()
        return (
            {"jobs_started": row[0], "tokens": row[1], "llm_calls": row[2]}
            if row
            else {
                "jobs_started": 0,
                "tokens": 0,
                "llm_calls": 0,
            }
        )

    def add_usage(self, tokens: int) -> None:
        """Charge one model call to today's counters (called after every call)."""
        with self._tx() as c:
            c.execute(
                "INSERT INTO daily_usage(day, tokens, llm_calls) VALUES(?, ?, 1) "
                "ON CONFLICT(day) DO UPDATE SET tokens = tokens + excluded.tokens, "
                "llm_calls = llm_calls + 1",
                (utc_day(self._now()), tokens),
            )

    def set_latch(self, until: datetime, reason: str) -> None:
        with self._tx() as c:
            c.execute(
                "INSERT OR REPLACE INTO quota_latch(id, until, reason) VALUES(1, ?, ?)",
                (iso(until), reason),
            )

    def latch_until(self) -> datetime | None:
        with self._read() as c:
            row = c.execute("SELECT until FROM quota_latch WHERE id = 1").fetchone()
        if row is None:
            return None
        until = parse_iso(row[0])
        return until if until > self._now() else None

    # ------------------------------------------------------------------ retention

    def purge(self) -> None:
        now = self._now()
        old_keys = iso(now - timedelta(hours=24))
        old_jobs = iso(now - timedelta(days=7))
        with self._tx() as c:
            c.execute("DELETE FROM idempotency WHERE created_at < ?", (old_keys,))
            marks = ",".join("?" for _ in TERMINAL)
            ids = [
                r[0]
                for r in c.execute(
                    f"SELECT id FROM jobs WHERE state IN ({marks}) AND finished_at < ?",
                    (*TERMINAL, old_jobs),
                )
            ]
            for job_id in ids:
                for table in ("events", "checkpoints", "snapshot"):
                    c.execute(f"DELETE FROM {table} WHERE job_id = ?", (job_id,))
                c.execute("DELETE FROM idempotency WHERE job_id = ?", (job_id,))
                c.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


class JobCheckpoints:
    """The `CheckpointStore` of one job. After `freeze()` (a simulated crash) every call raises
    SimulatedCrash, so the threads of a dead run stop and write nothing, as in a real death."""

    def __init__(self, store: JobStore, job_id: str) -> None:
        self._store = store
        self._job_id = job_id
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def _check(self) -> None:
        if self._frozen:
            raise SimulatedCrash(self._job_id)

    def load(self, key: str) -> str | None:
        self._check()
        return self._store.load_checkpoint(self._job_id, key)

    def load_snapshot(self) -> Snapshot | None:
        return self._store.load_snapshot(self._job_id)

    def cancel_requested(self) -> bool:
        self._check()
        return self._store.cancel_requested(self._job_id)

    def commit(
        self,
        key: str,
        output_json: str,
        snapshot: Callable[[], Snapshot],
        events: list[tuple[str, dict[str, Any]]],
    ) -> None:
        self._check()
        self._store.commit_checkpoint(self._job_id, key, output_json, snapshot, events)
