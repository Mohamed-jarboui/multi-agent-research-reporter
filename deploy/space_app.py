"""Entry point of the Hugging Face Gradio Space (pushed to the Space as `app.py`).

A genuine Gradio app (`gr.Blocks`, launched the way HF's template does) that also serves our
FastAPI service from the same server on port 7860:

- `/reports`, `/reports/{id}`, `/reports/{id}/events`, `/reports/{id}/trace`, `/health`,
  `/quota` are the service API, unchanged. They are registered first, so they win over the UI.
- `/` is the Gradio page. It talks to the service in-process (an httpx client wired straight to
  the ASGI app, no socket), so it goes through the same validation, rate limits, idempotency
  and daily cap as an outside client. Progress events are read from the job store.

The Space has `src/` next to this file, so the package is imported from there (no install
step). The app does not use a GPU, so it neither imports `spaces` nor defines a GPU function.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the Space layout is app.py + src/; in the repository this file is deploy/space_app.py
for candidate in (HERE / "src", HERE.parent / "src"):
    if (candidate / "reporter").is_dir():
        sys.path.insert(0, str(candidate))
        break

import gradio as gr  # noqa: E402
import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from reporter.service.app import app_from_env  # noqa: E402
from reporter.service.store import TERMINAL  # noqa: E402
from reporter.service.ui_text import event_line, quota_line, report_markdown, status_line  # noqa: E402

POLL_S = 3.0
MAX_WAIT_S = 900.0  # the UI stops watching after 15 minutes (the job itself has its own budget)


def client_key(request: gr.Request | None) -> str:
    """The visitor's address as the service's rate limiter should see it (HF proxy: last hop)."""
    if request is None:
        return "gradio"
    forwarded = request.headers.get("x-forwarded-for", "")
    parts = [p.strip() for p in forwarded.split(",") if p.strip()]
    if parts:
        return parts[-1]
    return request.client.host if request.client else "gradio"


def build_ui(api: FastAPI) -> gr.Blocks:
    """The demo page; every action goes through `api` (in-process)."""
    store = api.state.store

    def http(ip: str) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=api)
        headers = {"X-Forwarded-For": ip}
        return httpx.AsyncClient(transport=transport, base_url="http://service", headers=headers)

    async def show_quota() -> str:
        async with http("gradio") as client:
            res = await client.get("/quota")
        return quota_line(res.json()) if res.status_code == 200 else "Quota unavailable."

    async def start(topic: str, request: gr.Request):
        """Create the job, then stream (status, events, report) until it is finished."""
        ip = client_key(request)
        events: list[str] = []
        async with http(ip) as client:
            res = await client.post(
                "/reports",
                json={"topic": topic},
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            body = res.json()
            if res.status_code not in (200, 202):
                yield body["error"]["message"], "", ""
                return
            job_id, seen, waited = body["id"], 0, 0.0
            yield "queued", "", ""
            while waited <= MAX_WAIT_S:
                for event in store.events_after(job_id, seen, 200):
                    seen = event["seq"]
                    events.append(event_line(event))
                job = (await client.get(f"/reports/{job_id}")).json()
                if job.get("state") in TERMINAL:
                    quota = (await client.get("/quota")).json()
                    yield status_line(job), "\n".join(events), report_markdown(job, quota)
                    return
                yield status_line(job), "\n".join(events), ""
                await asyncio.sleep(POLL_S)
                waited += POLL_S
            yield "Still running; check back with GET /reports/" + job_id, "\n".join(events), ""

    with gr.Blocks(title="Multi-Agent Research Reporter") as demo:
        gr.Markdown(
            "# Multi-Agent Research Reporter\n"
            "Give a topic. A supervisor plans the report, workers research the sections in "
            "parallel, a critic checks them and a writer assembles a cited report. The free "
            "model quota allows only a few runs per day."
        )
        quota_box = gr.Markdown("Loading quota...")
        topic = gr.Textbox(label="Topic", placeholder="e.g. state of solid-state batteries")
        go = gr.Button("Start", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        log = gr.Textbox(label="Progress events", lines=8, interactive=False)
        report = gr.Markdown()
        go.click(start, topic, [status, log, report], concurrency_limit=4)
        topic.submit(start, topic, [status, log, report], concurrency_limit=4)
        demo.load(show_quota, None, quota_box)
    return demo


def build_app() -> FastAPI:
    """Our FastAPI service with the Gradio page mounted at `/` after its API routes."""
    api = app_from_env()
    # the static page of local runs would shadow the Gradio page at "/"
    api.router.routes = [r for r in api.router.routes if getattr(r, "path", None) != "/"]
    return gr.mount_gradio_app(api, build_ui(api), path="/")


app = build_app()

if __name__ == "__main__":
    # one worker on purpose: the queue, the rate limits and the quota latch live in this process
    uvicorn.run(app, host="0.0.0.0", port=7860, workers=1)
