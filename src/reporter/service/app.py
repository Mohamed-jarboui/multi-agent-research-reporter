"""The HTTP service: jobs with run ids, status, live progress over SSE, a quota page.

    POST   /reports               start a job (Idempotency-Key header optional)  -> 202 {id}
    GET    /reports/{id}          state, progress or result
    GET    /reports/{id}/events   server-sent events (Last-Event-ID or ?after=n resumes)
    DELETE /reports/{id}          cancel (a running job stops at the next node)
    GET    /health, /quota, /     liveness, today's counters, the demo page

Answers carry only fixed reason codes and texts: never a raw provider error, a key, an id of a
provider, an IP or an Idempotency-Key.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, ValidationError

from reporter.agent.llm_call import LLM
from reporter.service.config import SECRET_ENV_NAMES, VERSION, Settings
from reporter.service.quota import RateLimiter, client_ip
from reporter.service.runner import MESSAGES, HooksFactory, Runner
from reporter.service.sse import event_stream
from reporter.service.store import TERMINAL, Conflict, JobStore, iso, next_midnight
from reporter.service.telemetry import read_spans, setup_tracing
from reporter.tools import Tool

log = logging.getLogger("reporter.service")

RUN_ID = re.compile(r"^rpt_\d{8}T\d{6}Z_[0-9a-f]{16}$")
KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
STATIC = Path(__file__).parent / "static" / "index.html"

ERRORS: dict[str, str] = {
    "invalid_request": "The request is not valid.",
    "payload_too_large": "The request body is too large.",
    "rate_limited": "Too many requests; slow down.",
    "queue_full": "The queue is full; try again in a minute.",
    "idempotency_conflict": "This Idempotency-Key was already used with a different request.",
    "not_found": "No such report.",
    "already_finished": "The job has already finished.",
    "too_many_streams": "Too many open event streams.",
    **{k: v for k, v in MESSAGES.items() if k in ("provider_quota", "daily_cap")},
}


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str


def mask_secrets(text: str) -> str:
    """`text` with the value of each API key (read from the environment) replaced by ***."""
    for name in SECRET_ENV_NAMES:
        value = os.environ.get(name, "")
        if len(value) >= 4:
            text = text.replace(value, "***")
    return text


_mask_installed = False


def install_key_mask() -> None:
    """Mask API keys in every log record of the process, whichever logger made it. (A filter on
    a logger would miss records from its children; the record factory sees them all.)"""
    global _mask_installed
    if _mask_installed:
        return
    _mask_installed = True
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        record.msg, record.args = mask_secrets(record.getMessage()), ()
        return record

    logging.setLogRecordFactory(factory)


def error(status: int, code: str, headers: dict[str, str] | None = None) -> JSONResponse:
    body = {"error": {"code": code, "message": ERRORS.get(code, MESSAGES.get(code, code))}}
    return JSONResponse(body, status_code=status, headers=headers)


def normalise_topic(topic: str) -> str:
    """Stripped, with every run of whitespace collapsed to one space."""
    return " ".join(topic.split())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_app(
    settings: Settings,
    *,
    make_llm: Callable[[], LLM] | None = None,
    make_tools: Callable[[], dict[str, Tool]] | None = None,
    now: Callable[[], datetime] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    hooks_factory: HooksFactory | None = None,
    run_thread: bool = True,
    tracing: bool = True,
    agent_clock: Callable[[], float] = time.perf_counter,
    agent_sleep: Callable[[float], None] = time.sleep,
) -> FastAPI:
    clock_now = now or (lambda: datetime.now(UTC))
    if make_llm is None:
        from reporter.llm import build_default_llm

        make_llm = build_default_llm
    if make_tools is None:
        from reporter.tools import default_tools

        make_tools = default_tools

    install_key_mask()
    store = JobStore(settings.db_path, now=clock_now)
    runner = Runner(
        settings,
        store,
        make_llm=make_llm,
        make_tools=make_tools,
        now=clock_now,
        clock=agent_clock,
        sleep=agent_sleep,
        hooks_factory=hooks_factory,
    )
    posts = RateLimiter(settings.post_burst, settings.post_refill_s, monotonic)
    reads = RateLimiter(settings.reads_per_minute, 60.0 / settings.reads_per_minute, monotonic)
    streams = {"open": 0}
    streams_lock = threading.Lock()
    if tracing:
        setup_tracing(settings.runs_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if run_thread:
            runner.start()
        yield
        if run_thread:
            runner.stop()
        store.close()

    app = FastAPI(title="Research reporter", version=VERSION, lifespan=lifespan)
    app.state.settings, app.state.store, app.state.runner = settings, store, runner
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Idempotency-Key", "Last-Event-ID"],
        )

    def ip_of(request: Request) -> str:
        peer = request.client.host if request.client else None
        return client_ip(request.headers, peer, settings.trust_proxy_hops)

    def read_limited(request: Request) -> JSONResponse | None:
        ip = ip_of(request)
        if reads.allow(ip):
            return None
        return error(429, "rate_limited", {"Retry-After": str(max(reads.retry_after_s(ip), 1))})

    # ------------------------------------------------------------------ POST

    @app.post("/reports")
    async def create_report(request: Request) -> JSONResponse:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > settings.max_body_bytes:
            return error(413, "payload_too_large")
        body = b""
        async for chunk in request.stream():
            body += chunk
            if len(body) > settings.max_body_bytes:
                return error(413, "payload_too_large")
        try:
            parsed = ReportRequest.model_validate(json.loads(body))
        except (ValueError, ValidationError):  # bad JSON, unknown field, wrong type
            return error(422, "invalid_request")
        topic = normalise_topic(parsed.topic)
        if not settings.topic_min_chars <= len(topic) <= settings.topic_max_chars or any(
            unicodedata.category(ch) in ("Cc", "Cf") for ch in topic
        ):
            return error(422, "invalid_request")
        key = request.headers.get("idempotency-key")
        if key is not None and not KEY_PATTERN.match(key):
            return error(422, "invalid_request")

        ip = ip_of(request)
        if not posts.allow(ip):
            return error(429, "rate_limited", {"Retry-After": str(max(posts.retry_after_s(ip), 1))})
        admission = store.admit(
            topic=topic,
            body_hash=sha256(json.dumps({"topic": topic}, sort_keys=True)),
            key_hash=sha256(key) if key else None,
            max_waiting=settings.max_waiting,
            daily_max_jobs=settings.daily_max_jobs,
            daily_max_tokens=settings.daily_max_tokens,
            min_tokens=int(settings.min_token_fraction * settings.run_max_tokens),
            idempotency_ttl_s=settings.idempotency_ttl_s,
        )
        if admission.kind == "rejected":
            headers = {"Retry-After": str(admission.retry_after_s)}
            return error(429, admission.code, headers)
        if admission.kind == "conflict":
            return error(409, "idempotency_conflict")
        links = {
            "self": f"/reports/{admission.job_id}",
            "events": f"/reports/{admission.job_id}/events",
        }
        payload = {"id": admission.job_id, "state": admission.state, "links": links}
        if admission.kind == "replay":
            return JSONResponse(payload, status_code=200, headers={"Idempotent-Replayed": "true"})
        runner.wake()
        return JSONResponse(payload, status_code=202)

    # ------------------------------------------------------------------ GET, DELETE

    @app.get("/reports/{job_id}")
    def get_report(job_id: str, request: Request) -> JSONResponse:
        if (limited := read_limited(request)) is not None:
            return limited
        job = store.get_job(job_id) if RUN_ID.match(job_id) else None
        if job is None:
            return error(404, "not_found")
        return JSONResponse(public_job(job))

    @app.delete("/reports/{job_id}")
    def cancel_report(job_id: str, request: Request) -> JSONResponse:
        if (limited := read_limited(request)) is not None:
            return limited
        if not RUN_ID.match(job_id):
            return error(404, "not_found")
        try:
            outcome = store.request_cancel(job_id)
        except KeyError:
            return error(404, "not_found")
        except Conflict:
            return error(409, "already_finished")
        runner.wake()
        state = "cancelled" if outcome == "cancelled" else "running"
        return JSONResponse({"id": job_id, "state": state, "cancel_requested": True}, 202)

    @app.get("/reports/{job_id}/events")
    async def get_events(job_id: str, request: Request, after: str = ""):
        if (limited := read_limited(request)) is not None:
            return limited
        if not RUN_ID.match(job_id) or store.get_job(job_id) is None:
            return error(404, "not_found")
        header = request.headers.get("last-event-id", "")
        start = int(header) if header.isdigit() else int(after) if after.isdigit() else 0
        with streams_lock:
            if streams["open"] >= settings.max_streams:
                return error(429, "too_many_streams", {"Retry-After": "5"})
            streams["open"] += 1

        async def generate():
            try:
                async for part in event_stream(
                    store,
                    job_id,
                    start,
                    poll_s=settings.sse_poll_s,
                    heartbeat_s=settings.sse_heartbeat_s,
                    sleep=sleep,
                    monotonic=monotonic,
                ):
                    yield part
            finally:
                with streams_lock:
                    streams["open"] -= 1

        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        return StreamingResponse(generate(), media_type="text/event-stream", headers=headers)

    @app.get("/reports/{job_id}/trace")
    def get_trace(job_id: str, request: Request) -> JSONResponse:
        """The OpenTelemetry spans of a finished job (names, timings, tokens; no prompts)."""
        if (limited := read_limited(request)) is not None:
            return limited
        job = store.get_job(job_id) if RUN_ID.match(job_id) else None
        if job is None or job["state"] not in TERMINAL:
            return error(404, "not_found")
        try:
            spans = read_spans(settings.runs_dir, job_id)
        except OSError:
            return error(404, "not_found")
        return JSONResponse({"run_id": job_id, "spans": spans})

    # ------------------------------------------------------------------ small pages

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "storage": settings.storage, "version": VERSION}

    @app.get("/quota")
    def quota(request: Request) -> JSONResponse:
        if (limited := read_limited(request)) is not None:
            return limited
        moment = clock_now()
        usage = store.usage_today()
        running, waiting = store.counts()
        latch = store.latch_until()
        return JSONResponse(
            {
                "day": moment.astimezone(UTC).strftime("%Y-%m-%d"),
                "resets_at": iso(next_midnight(moment)),
                "jobs": {
                    "used": usage["jobs_started"],
                    "limit": settings.daily_max_jobs,
                    "left": max(settings.daily_max_jobs - usage["jobs_started"], 0),
                },
                "tokens": {
                    "used": usage["tokens"],
                    "limit": settings.daily_max_tokens,
                    "left": max(settings.daily_max_tokens - usage["tokens"], 0),
                },
                "queue": {"running": running, "waiting": waiting, "max": settings.max_waiting},
                "latch_until": iso(latch) if latch else None,
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return STATIC.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ public view

    def public_job(job: dict[str, Any]) -> dict[str, Any]:
        state = job["state"]
        out: dict[str, Any] = {
            "id": job["id"],
            "state": state,
            "reason": job["reason"],
            "message": MESSAGES.get(job["reason"] or ""),
            "topic": job["topic"],
            "created_at": job["created_at"],
            "started_at": job["started_at"],
            "finished_at": job["finished_at"],
            "resume_count": job["resume_count"],
        }
        if state == "queued":
            out["queue_position"] = store.queue_position(job["id"])
        elif state == "running":
            out["progress"] = progress(job["id"])
            snap = store.load_snapshot(job["id"])
            out["budget"] = {"used": snap.meter if snap else None}
        elif state in ("succeeded", "degraded"):
            if job["result_json"]:
                out["result"] = json.loads(job["result_json"])
            if job["usage_json"]:
                usage = json.loads(job["usage_json"])
                usage["trace"]["url"] = f"/reports/{job['id']}/trace"
                out["budget"], out["trace"] = usage["budget"], usage["trace"]
        return out

    def progress(job_id: str) -> dict[str, Any]:
        events = store.events_after(job_id, 0, 500)
        total = 0
        done: set[str] = set()
        for event in events:
            if event["type"] == "plan":
                total = len(event["data"]["sections"])
            elif event["type"] == "worker_done":
                done.add(event["data"]["section_id"])
        last = events[-1]["type"] if events else None
        return {"last_event": last, "sections_done": len(done), "sections_total": total}

    return app


def app_from_env() -> FastAPI:
    """The ASGI app of the container (`uvicorn reporter.service.app:app`)."""
    from reporter.cli import load_env_file

    load_env_file()
    return create_app(Settings.from_env())


def __getattr__(name: str) -> Any:
    """`reporter.service.app:app` builds the app on first use, not at import (tests import this
    module without wanting a database)."""
    if name == "app":
        value = app_from_env()
        globals()["app"] = value
        return value
    raise AttributeError(name)
