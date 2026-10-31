"""Server-sent events: order, resume, late subscribers, heartbeat, close, disconnect."""

from __future__ import annotations

import asyncio
import json
import re

from reporter.service.sse import event_stream
from reporter.service.store import EVENT_TYPES
from svc import TOPIC, make_service


def parse(text: str) -> list[dict]:
    """Frames of an SSE body: [{id, event, data}]."""
    frames = []
    for block in text.split("\n\n"):
        fields = dict(re.findall(r"^(id|event|data): (.*)$", block, re.MULTILINE))
        if fields:
            frames.append(fields)
    return frames


def finished(tmp_path, **kw):
    svc = make_service(tmp_path, **kw)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.runner.drain()
    return svc, jid


def test_ids_run_from_1_in_order_and_the_stream_closes_after_done(tmp_path):
    svc, jid = finished(tmp_path)
    r = svc.client.get(f"/reports/{jid}/events")
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.text.startswith("retry: 3000\n\n")
    frames = parse(r.text)
    assert [int(f["id"]) for f in frames] == list(range(1, len(frames) + 1))
    types = [f["event"] for f in frames]
    assert set(types) <= set(EVENT_TYPES)
    assert types[:3] == ["queued", "started", "plan"]
    assert types.count("worker_done") == 3 and types[-2:] == ["written", "done"]
    assert types.index("critic_verdict") > max(i for i, t in enumerate(types) if t == "worker_done")


def test_last_event_id_header_and_after_param_resume(tmp_path):
    svc, jid = finished(tmp_path)
    everything = parse(svc.client.get(f"/reports/{jid}/events").text)
    n = len(everything)
    by_header = parse(svc.client.get(f"/reports/{jid}/events", headers={"Last-Event-ID": "4"}).text)
    assert [int(f["id"]) for f in by_header] == list(range(5, n + 1))
    by_param = parse(svc.client.get(f"/reports/{jid}/events?after={n - 1}").text)
    assert [f["event"] for f in by_param] == ["done"]
    assert parse(svc.client.get(f"/reports/{jid}/events?after=junk").text) == everything


def test_a_late_subscriber_gets_the_whole_history_from_the_database(tmp_path):
    svc, jid = finished(tmp_path)
    again = make_service(tmp_path)  # a "restarted" process on the same database
    frames = parse(again.client.get(f"/reports/{jid}/events").text)
    assert frames[0]["event"] == "queued" and frames[-1]["event"] == "done"


def test_the_event_data_holds_only_the_fixed_fields(tmp_path):
    svc, jid = finished(tmp_path)
    by_type = {f["event"]: f["data"] for f in parse(svc.client.get(f"/reports/{jid}/events").text)}
    assert '"sections":[{"id":"s1","title":"Part 1"}' in by_type["plan"]
    assert set(json.loads(by_type["worker_done"])) == {
        "section_id",
        "status",
        "pages",
        "findings",
    }
    assert by_type["done"] == '{"state":"succeeded","reason":null}'


def test_heartbeat_after_15_s_of_quiet_then_events_then_close(tmp_path):
    clock = {"t": 0.0, "sleeps": 0}
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]

    async def sleep(seconds: float) -> None:
        clock["t"] += seconds
        clock["sleeps"] += 1
        if clock["sleeps"] == 20:  # the job runs while the client is connected
            svc.runner.drain()

    async def collect() -> list[str]:
        stream = event_stream(
            svc.store,
            jid,
            0,
            poll_s=1.0,
            heartbeat_s=15.0,
            sleep=sleep,
            monotonic=lambda: clock["t"],
        )
        return [part async for part in stream]

    parts = asyncio.run(collect())
    heartbeats = [i for i, p in enumerate(parts) if p == ": hb\n\n"]
    assert len(heartbeats) == 1  # one at 15 s of fake quiet; the job ran at 20 s
    frames = parse("".join(parts))
    assert [f["event"] for f in frames][0] == "queued" and frames[-1]["event"] == "done"
    assert heartbeats[0] == 2  # after "retry:" and the queued frame, before anything else


def test_a_disconnect_does_not_stop_the_job(tmp_path):
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]

    async def read_two_then_leave() -> list[str]:
        stream = event_stream(
            svc.store,
            jid,
            0,
            poll_s=1.0,
            heartbeat_s=15.0,
            sleep=lambda s: asyncio.sleep(0),
            monotonic=lambda: 0.0,
        )
        got = [await stream.__anext__(), await stream.__anext__()]
        await stream.aclose()  # the client went away
        return got

    got = asyncio.run(read_two_then_leave())
    assert got[1].startswith("id: 1\nevent: queued")
    svc.runner.drain()
    assert svc.store.get_job(jid)["state"] == "succeeded"


def test_too_many_open_streams_get_429(tmp_path):
    svc, jid = finished(tmp_path, max_streams=0)
    r = svc.client.get(f"/reports/{jid}/events")
    assert r.status_code == 429 and r.json()["error"]["code"] == "too_many_streams"
