"""Crash and resume: finished workers are not run again; meter and trace continue; the resume
cap ends a job that keeps dying."""

from __future__ import annotations

import json

import pytest

from fakes import TeamLLM
from reporter.llm.router import FallbackLLM
from reporter.service.runner import CrashHook
from reporter.service.store import JobCheckpoints, SimulatedCrash
from svc import TOPIC, crash_after, make_service


def one_worker(tmp_path, hooks_factory=None, **kw):
    """Workers run one at a time, so a crash after s1 happens before s2 and s3 start."""
    return make_service(tmp_path, max_parallel_workers=1, hooks_factory=hooks_factory, **kw)


def titles_called(llm: TeamLLM) -> set[str]:
    return {title for (_, title) in llm.counts if title}


def test_crash_mid_fan_out_resumes_and_only_unfinished_workers_run(tmp_path):
    first = one_worker(tmp_path, crash_after("worker:s1"))
    jid = first.client.post("/reports", json=TOPIC).json()["id"]
    first.runner.drain()
    assert first.store.get_job(jid)["state"] == "running"  # the "process" died mid-run
    before = first.store.load_snapshot(jid)
    assert before is not None and before.last_key == "worker:s1"

    second = one_worker(tmp_path)  # a new process: new app, new router, same database
    second.runner.startup()
    assert second.store.get_job(jid)["resume_count"] == 1
    second.runner.drain()

    llm = second.llms[0]
    assert llm.counts[("plan", "")] == 0  # the supervisor was not asked again
    assert "Part 1" not in titles_called(llm)  # s1 is not recomputed
    assert {"Part 2", "Part 3"} <= titles_called(llm)

    job = second.client.get(f"/reports/{jid}").json()
    assert job["state"] == "succeeded" and job["resume_count"] == 1
    assert [s["id"] for s in job["result"]["sections"]] == ["s1", "s2", "s3"]
    # the report still cites s1's sources, which came from the checkpoint
    assert {s["id"] for s in job["result"]["sources"]} == {1, 2, 3, 4, 5, 6}

    events = second.store.events_after(jid, 0)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    types = [e["type"] for e in events]
    assert types.count("plan") == 1 and types.count("done") == 1
    done_ids = [e["data"]["section_id"] for e in events if e["type"] == "worker_done"]
    assert sorted(done_ids) == ["s1", "s2", "s3"]  # no duplicates
    started = [e["data"] for e in events if e["type"] == "started"]
    assert started == [
        {"resumed": False, "from_key": None},
        {"resumed": True, "from_key": "worker:s1"},
    ]

    # the meter continued from the snapshot, and the trace says it was resumed once
    assert job["budget"]["used"]["tokens"] >= before.meter["tokens"]
    assert job["budget"]["used"]["steps"] >= before.meter["steps"]
    trace = json.loads((second.settings.runs_dir / f"{jid}.json").read_text("utf-8"))
    assert len(trace["resumes"]) == 1 and trace["resumes"][0]["from_key"] == "worker:s1"
    assert len([w for w in trace["workers"] if w["section_id"] == "s1"]) == 1


def test_a_crash_with_parallel_workers_also_resumes_to_success(tmp_path):
    first = make_service(tmp_path, hooks_factory=crash_after("worker:s2"))
    jid = first.client.post("/reports", json=TOPIC).json()["id"]
    first.runner.drain()
    assert first.store.get_job(jid)["state"] == "running"
    second = make_service(tmp_path)
    second.runner.startup()
    second.runner.drain()
    assert second.store.get_job(jid)["state"] == "succeeded"
    assert "Part 2" not in titles_called(second.llms[0])


def test_the_job_ends_failed_after_two_resumes_that_also_crash(tmp_path):
    def always(job_id, checkpoints):
        class Die:
            def after_checkpoint(self, key):
                checkpoints.freeze()
                raise SimulatedCrash(key)

        return Die()

    svc = one_worker(tmp_path, always)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.runner.drain()  # crash 1
    for expected_count in (1, 2):  # restarts 1 and 2 resume, and die again
        svc = one_worker(tmp_path, always)
        svc.runner.startup()
        assert svc.store.get_job(jid)["resume_count"] == expected_count
        svc.runner.drain()
        assert svc.store.get_job(jid)["state"] == "running"
    final = one_worker(tmp_path)  # restart 3: the cap is reached
    final.runner.startup()
    final.runner.drain()
    job = final.client.get(f"/reports/{jid}").json()
    assert job["state"] == "failed" and job["reason"] == "crashed_repeatedly"
    assert "interrupted" in job["message"] and "result" not in job
    assert final.llms == []  # nothing ran
    last = final.store.events_after(jid, 0)[-1]
    assert last["type"] == "failed" and last["data"]["reason"] == "crashed_repeatedly"


def test_the_crash_hook_fires_once_per_job_even_when_called_again(tmp_path):
    svc = one_worker(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.store.start_job(jid)
    checkpoints = JobCheckpoints(svc.store, jid)
    hook = CrashHook(svc.store, jid, checkpoints, "worker:s1", mode="raise")
    hook.after_checkpoint("supervisor")  # another key: nothing
    with pytest.raises(SimulatedCrash):
        hook.after_checkpoint("worker:s1")
    hook.after_checkpoint("worker:s1")  # a restart would not die again
    with pytest.raises(SimulatedCrash):
        checkpoints.load("supervisor")  # the dead run's store is frozen


def test_startup_resumes_a_running_job_before_any_queued_job(tmp_path):
    svc = make_service(tmp_path)
    a = svc.client.post("/reports", json=TOPIC).json()["id"]
    b = svc.client.post("/reports", json={"topic": "second topic"}).json()["id"]
    svc.store.start_job(a)  # `a` was running when the process died; `b` was still queued
    svc.runner.startup()
    svc.runner.drain()
    assert svc.runner.processed == [a, b]
    assert svc.store.get_job(a)["state"] == svc.store.get_job(b)["state"] == "succeeded"
    assert svc.store.get_job(a)["resume_count"] == 1 and svc.store.get_job(b)["resume_count"] == 0


def test_a_crash_before_the_first_checkpoint_restarts_from_scratch(tmp_path):
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.store.start_job(jid)  # died before the supervisor finished: no snapshot at all
    svc.runner.startup()
    svc.runner.drain()
    assert svc.store.get_job(jid)["state"] == "succeeded"
    assert svc.llms[0].counts[("plan", "")] == 1


def test_checkpoints_are_deleted_when_the_job_ends(tmp_path):
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC).json()["id"]
    svc.runner.drain()
    with svc.store._read() as c:
        assert c.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert c.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0] == 0
    assert svc.store.get_job(jid)["state"] == "succeeded"


def test_purge_removes_old_keys_and_old_finished_jobs(tmp_path):
    svc = make_service(tmp_path)
    jid = svc.client.post("/reports", json=TOPIC, headers={"Idempotency-Key": "k" * 12}).json()[
        "id"
    ]
    svc.runner.drain()
    svc.now.advance(25 * 3600)
    svc.store.purge()
    assert svc.store.get_job(jid) is not None  # jobs live 7 days
    with svc.store._read() as c:
        assert c.execute("SELECT COUNT(*) FROM idempotency").fetchone()[0] == 0
    svc.now.advance(7 * 24 * 3600)
    svc.store.purge()
    assert svc.store.get_job(jid) is None and svc.store.events_after(jid, 0) == []


class Backend:
    """A provider that is a TeamLLM, so a real FallbackLLM keeps its per-provider summary."""

    name = "fake"

    def __init__(self) -> None:
        self.team = TeamLLM(3)

    def generate(self, prompt, **kwargs):
        return self.team.generate(prompt, **kwargs)


def test_the_router_summary_adds_up_over_the_segments_of_a_resumed_job(tmp_path):
    def router():
        return FallbackLLM([Backend()], max_attempts=1)

    first = one_worker(tmp_path, crash_after("worker:s1"), make_llm=router)
    jid = first.client.post("/reports", json=TOPIC).json()["id"]
    first.runner.drain()
    second = one_worker(tmp_path, make_llm=router)
    second.runner.startup()
    second.runner.drain()
    calls_1 = first.llms[0].summary()["calls"]["fake"]
    calls_2 = second.llms[0].summary()["calls"]["fake"]
    trace = json.loads((second.settings.runs_dir / f"{jid}.json").read_text("utf-8"))
    assert trace["llm"]["calls"]["fake"] == calls_1 + calls_2
    assert second.llms[0] is not first.llms[0]  # a new router per job segment
