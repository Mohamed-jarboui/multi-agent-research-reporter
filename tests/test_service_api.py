"""The HTTP service with fake models and tools: admission, idempotency, state machine, quota,
validation, redaction. No network."""

from __future__ import annotations

import logging
import re
import threading
import time

import pytest

from fakes import TeamLLM
from reporter.llm.base import ProviderError, RateLimitError
from reporter.llm.router import FallbackLLM
from reporter.service.config import Settings, resolve_data_dir
from reporter.service.quota import RateLimiter, client_ip
from reporter.service.store import InvalidTransition
from svc import KEY, TOPIC, make_service

RUN_ID = re.compile(r"^rpt_\d{8}T\d{6}Z_[0-9a-f]{16}$")


def post(svc, body=None, key=None, headers=None):
    h = dict(headers or {})
    if key:
        h["Idempotency-Key"] = key
    return svc.client.post("/reports", json=TOPIC if body is None else body, headers=h)


# ------------------------------------------------------------------ create and read


def test_post_returns_a_run_id_and_links(tmp_path):
    svc = make_service(tmp_path)
    r = post(svc)
    assert r.status_code == 202
    body = r.json()
    assert RUN_ID.match(body["id"]) and body["state"] == "queued"
    assert body["links"] == {
        "self": f"/reports/{body['id']}",
        "events": f"/reports/{body['id']}/events",
    }
    queued = svc.client.get(body["links"]["self"]).json()
    assert queued["state"] == "queued" and queued["queue_position"] == 1
    assert queued["message"] is None


def test_finished_job_has_result_budget_and_trace_summary(tmp_path):
    svc = make_service(tmp_path)
    jid = post(svc).json()["id"]
    svc.runner.drain()
    job = svc.client.get(f"/reports/{jid}").json()
    assert job["state"] == "succeeded" and job["reason"] is None
    assert job["resume_count"] == 0 and job["finished_at"]
    result = job["result"]
    assert result["report_markdown"].startswith("# Report")
    assert [s["id"] for s in result["sections"]] == ["s1", "s2", "s3"]
    assert result["writer_status"] == "ok"
    assert set(result["citation_check"]) == {
        "invalid_citations",
        "uncited_sources",
        "unfetched_urls",
    }
    assert job["budget"]["limit"]["max_steps"] == 40
    assert job["budget"]["used"]["tokens"] == job["trace"]["total_tokens"] > 0
    assert job["trace"]["cost_usd"] is None  # the fake model is unpriced: null, never 0
    assert job["trace"]["url"] == f"/reports/{jid}/trace"


def test_unknown_and_malformed_ids_are_404(tmp_path):
    svc = make_service(tmp_path)
    assert svc.client.get("/reports/nope").status_code == 404
    assert svc.client.get("/reports/rpt_20261031T120000Z_0123456789abcdef").status_code == 404
    assert svc.client.get("/reports/nope/events").status_code == 404
    assert svc.client.delete("/reports/nope").status_code == 404


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "body",
    [
        {"topic": "abc"},  # too short
        {"topic": "x" * 301},  # too long
        {"topic": "fine topic", "extra": 1},  # unknown field
        {"topic": 5},
        {},
        {"topic": "bad \x00 char"},
        {"topic": "     "},
    ],
)
def test_invalid_bodies_are_422(tmp_path, body):
    svc = make_service(tmp_path)
    r = post(svc, body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    assert svc.store.counts() == (0, 0)


def test_not_json_is_422_and_oversized_body_is_413(tmp_path):
    svc = make_service(tmp_path)
    assert svc.client.post("/reports", content=b"not json").status_code == 422
    big = svc.client.post("/reports", json={"topic": "a" * 5000})
    assert big.status_code == 413 and big.json()["error"]["code"] == "payload_too_large"


def test_a_topic_of_300_chars_is_accepted_and_whitespace_is_collapsed(tmp_path):
    svc = make_service(tmp_path)
    assert post(svc, {"topic": "a" * 300}).status_code == 202
    jid = post(svc, {"topic": "  two \n  words  "}).json()["id"]
    assert svc.store.get_job(jid)["topic"] == "two words"


def test_bad_idempotency_key_is_422(tmp_path):
    svc = make_service(tmp_path)
    assert post(svc, key="short").status_code == 422
    assert post(svc, key="has space in it!").status_code == 422


def test_cors_is_off_by_default_and_on_when_configured(tmp_path):
    off = make_service(tmp_path / "a")
    r = off.client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    on = make_service(tmp_path / "b", cors_origins=("https://ok.example",))
    r = on.client.get("/health", headers={"Origin": "https://ok.example"})
    assert r.headers["access-control-allow-origin"] == "https://ok.example"


# ------------------------------------------------------------------ idempotency


def test_same_key_and_body_returns_the_same_job(tmp_path):
    svc = make_service(tmp_path)
    first, second = post(svc, key=KEY), post(svc, key=KEY)
    assert first.status_code == 202 and second.status_code == 200
    assert second.json()["id"] == first.json()["id"]
    assert second.headers["Idempotent-Replayed"] == "true"
    assert svc.store.counts() == (0, 1)


def test_same_key_with_another_body_is_409(tmp_path):
    svc = make_service(tmp_path)
    post(svc, key=KEY)
    other = post(svc, {"topic": "a different topic"}, key=KEY)
    assert other.status_code == 409 and other.json()["error"]["code"] == "idempotency_conflict"
    assert svc.store.counts() == (0, 1)


def test_whitespace_differences_do_not_change_the_body_hash(tmp_path):
    svc = make_service(tmp_path)
    a = post(svc, {"topic": "solid state batteries"}, key=KEY)
    b = post(svc, {"topic": "  solid   state batteries "}, key=KEY)
    assert b.status_code == 200 and b.json()["id"] == a.json()["id"]


def test_no_key_always_makes_a_new_job(tmp_path):
    svc = make_service(tmp_path)
    assert post(svc).json()["id"] != post(svc).json()["id"]
    assert svc.store.counts() == (0, 2)


def test_a_key_expires_after_24_hours(tmp_path):
    svc = make_service(tmp_path)
    first = post(svc, key=KEY).json()["id"]
    svc.now.advance(25 * 3600)
    again = post(svc, key=KEY)
    assert again.status_code == 202 and again.json()["id"] != first


def test_keys_are_stored_hashed(tmp_path):
    svc = make_service(tmp_path)
    post(svc, key=KEY)
    with svc.store._read() as c:
        stored = [r[0] for r in c.execute("SELECT key_hash FROM idempotency")]
    assert stored and KEY not in stored[0] and len(stored[0]) == 64


def test_two_threads_with_one_key_make_one_job(tmp_path):
    svc = make_service(tmp_path)
    barrier = threading.Barrier(2)
    kinds: list[str] = []

    def go():
        barrier.wait()
        adm = svc.store.admit(
            topic="t topic",
            body_hash="h",
            key_hash="k" * 64,
            max_waiting=3,
            daily_max_jobs=9,
            daily_max_tokens=10**9,
            min_tokens=0,
            idempotency_ttl_s=86400,
        )
        kinds.append(adm.kind)

    threads = [threading.Thread(target=go) for _ in range(2)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(kinds) == ["created", "replay"] and svc.store.counts() == (0, 1)


def test_a_replay_still_works_when_the_queue_is_full(tmp_path):
    svc = make_service(tmp_path, max_waiting=1)
    first = post(svc, key=KEY)
    assert post(svc).status_code == 429
    again = post(svc, key=KEY)
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"]


# ------------------------------------------------------------------ queue, cap, rate limit


def test_fourth_waiting_job_gets_429_queue_full(tmp_path):
    svc = make_service(tmp_path)
    assert [post(svc).status_code for _ in range(3)] == [202, 202, 202]
    full = post(svc)
    assert full.status_code == 429 and full.json()["error"]["code"] == "queue_full"
    assert "retry-after" in full.headers
    svc.runner.step()  # one job leaves the queue
    assert post(svc).status_code == 202


def test_daily_job_cap_rejects_until_midnight_utc(tmp_path):
    svc = make_service(tmp_path, daily_max_jobs=2)  # clock: 2026-10-31 12:00 UTC
    post(svc)
    post(svc)
    svc.runner.drain()
    capped = post(svc)
    assert capped.status_code == 429 and capped.json()["error"]["code"] == "daily_cap"
    assert int(capped.headers["retry-after"]) == 12 * 3600 + 1
    svc.now.advance(12 * 3600 + 1)  # past 00:00 UTC: the counters start again
    assert post(svc).status_code == 202
    assert svc.client.get("/quota").json()["jobs"] == {"used": 0, "limit": 2, "left": 2}


def test_daily_token_cap(tmp_path):
    svc = make_service(tmp_path, run_max_tokens=10_000, daily_max_tokens=10_000)
    svc.store.add_usage(9_500)  # 500 left, but a job needs 20% of 10 000
    capped = post(svc)
    assert capped.status_code == 429 and capped.json()["error"]["code"] == "daily_cap"
    svc.now.advance(13 * 3600)
    assert post(svc).status_code == 202


def test_a_queued_job_that_starts_after_the_cap_ends_degraded_without_a_model_call(tmp_path):
    svc = make_service(tmp_path, run_max_tokens=10_000, daily_max_tokens=10_000)
    jid = post(svc).json()["id"]
    svc.store.add_usage(9_500)  # another run used the day's tokens while this one waited
    svc.runner.drain()
    job = svc.client.get(f"/reports/{jid}").json()
    assert job["state"] == "degraded" and job["reason"] == "daily_cap"
    assert job["message"]
    kinds = [e["type"] for e in svc.store.events_after(jid, 0)]
    assert kinds == ["queued", "quota", "done"]
    assert svc.llms == []  # no router was even built


def test_per_ip_token_bucket_refills_with_time(tmp_path):
    svc = make_service(tmp_path, post_burst=2, max_waiting=10)
    ip = {"X-Forwarded-For": "9.9.9.9"}
    assert [post(svc, headers=ip).status_code for _ in range(2)] == [202, 202]
    limited = post(svc, headers=ip)
    assert limited.status_code == 429 and limited.json()["error"]["code"] == "rate_limited"
    assert post(svc, headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 202  # another client
    svc.now.advance(301)  # one token every 300 s
    assert post(svc, headers=ip).status_code == 202


def test_a_forged_left_entry_of_x_forwarded_for_does_not_help(tmp_path):
    svc = make_service(tmp_path, post_burst=1)
    assert post(svc, headers={"X-Forwarded-For": "1.1.1.1, 7.7.7.7"}).status_code == 202
    again = post(svc, headers={"X-Forwarded-For": "2.2.2.2, 7.7.7.7"})  # same proxy-seen client
    assert again.status_code == 429


def test_client_ip_rule_and_bucket_unit():
    assert client_ip({"x-forwarded-for": "a, b, c"}, "peer", 1) == "c"
    assert client_ip({"x-forwarded-for": "a, b, c"}, "peer", 2) == "b"
    assert client_ip({}, "peer", 1) == "peer"
    assert client_ip({"x-forwarded-for": "a"}, "peer", 0) == "peer"
    now = [0.0]
    bucket = RateLimiter(1, 10.0, lambda: now[0], max_clients=2)
    assert bucket.allow("x") and not bucket.allow("x")
    assert bucket.retry_after_s("x") == 11
    bucket.allow("y")
    bucket.allow("z")  # "x" is the least recently used: dropped, so it starts full again
    assert bucket.allow("x")


# ------------------------------------------------------------------ state machine


def test_a_terminal_job_cannot_change(tmp_path):
    svc = make_service(tmp_path)
    jid = post(svc).json()["id"]
    svc.runner.drain()
    for target in ("running", "succeeded", "failed", "cancelled", "degraded"):
        with pytest.raises(InvalidTransition):
            if target == "running":
                svc.store.start_job(jid)
            else:
                svc.store.finish(jid, target, None)
    assert svc.store.get_job(jid)["state"] == "succeeded"


def test_forbidden_first_moves_are_refused(tmp_path):
    svc = make_service(tmp_path)
    jid = post(svc).json()["id"]
    with pytest.raises(InvalidTransition):
        svc.store.finish(jid, "succeeded", None)  # queued -> succeeded is not allowed
    with pytest.raises(InvalidTransition):
        svc.store.mark_resumed(jid)  # only a running job can resume
    svc.store.start_job(jid)
    with pytest.raises(InvalidTransition):
        svc.store.start_job(jid)  # running -> running through start is not allowed
    svc.store.finish(jid, "failed", "internal_error")
    assert svc.store.get_job(jid)["state"] == "failed"


def test_delete_cancels_a_queued_job_now(tmp_path):
    svc = make_service(tmp_path)
    jid = post(svc).json()["id"]
    r = svc.client.delete(f"/reports/{jid}")
    assert r.status_code == 202 and r.json()["state"] == "cancelled"
    job = svc.client.get(f"/reports/{jid}").json()
    assert job["state"] == "cancelled" and job["reason"] == "cancelled_by_user"
    assert "result" not in job
    svc.runner.drain()
    assert svc.llms == []  # it never ran


def test_delete_on_a_terminal_job_is_409(tmp_path):
    svc = make_service(tmp_path)
    jid = post(svc).json()["id"]
    svc.runner.drain()
    r = svc.client.delete(f"/reports/{jid}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "already_finished"


def test_delete_on_a_running_job_stops_it_at_the_next_node(tmp_path):
    holder: dict[str, str] = {}

    def on_call(kind, title, nth):
        if kind == "plan":  # the supervisor is running: the client cancels now
            assert svc.client.delete(f"/reports/{holder['id']}").status_code == 202
        return None

    svc = make_service(tmp_path, make_llm=lambda: TeamLLM(3, on_call=on_call))
    holder["id"] = post(svc).json()["id"]
    svc.runner.drain()
    job = svc.client.get(f"/reports/{holder['id']}").json()
    assert job["state"] == "cancelled" and job["reason"] == "cancelled_by_user"
    kinds = [e["type"] for e in svc.store.events_after(holder["id"], 0)]
    assert kinds == ["queued", "started", "plan", "failed"]  # no worker started


# ------------------------------------------------------------------ provider quota


class QuotaProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate(self, prompt, **kwargs):
        error = RateLimitError("429 quota for key sk-LEAK", provider=self.name, status=429)
        error.limit = "per_day_requests"
        raise error


def quota_llm():
    return FallbackLLM(
        [QuotaProvider("gemini"), QuotaProvider("groq")], max_attempts=1, sleep=lambda s: None
    )


def test_every_provider_out_of_daily_quota_degrades_and_latches(tmp_path):
    svc = make_service(tmp_path, make_llm=quota_llm)
    jid = post(svc).json()["id"]
    svc.runner.drain()
    job = svc.client.get(f"/reports/{jid}").json()
    assert job["state"] == "degraded" and job["reason"] == "provider_quota"
    assert "quota" in job["message"]
    events = svc.store.events_after(jid, 0)
    quota = next(e for e in events if e["type"] == "quota")
    assert quota["data"]["kind"] == "provider_limit"
    assert quota["data"]["limit_type"] == "per_day_requests"
    assert events[-1]["type"] == "done"
    assert svc.store.latch_until() is not None
    blocked = post(svc)
    assert blocked.status_code == 429 and blocked.json()["error"]["code"] == "provider_quota"
    assert svc.client.get("/quota").json()["latch_until"]
    svc.now.advance(3601)  # no delay was given by the provider: the latch is one hour
    assert post(svc).status_code == 202


# ------------------------------------------------------------------ redaction


def test_provider_errors_and_keys_never_reach_answers_events_logs_or_traces(
    tmp_path, monkeypatch, caplog
):
    secret = "sk-FAKEKEY-0123456789"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    class Boom:
        def generate(self, prompt, **kwargs):
            raise ProviderError(f"upstream said: bad key {secret} org_ABC123")

    svc = make_service(tmp_path, make_llm=Boom)
    with caplog.at_level(logging.DEBUG, logger="reporter"):
        logging.getLogger("reporter.service").warning("leaky line with %s", secret)
        jid = post(svc).json()["id"]
        svc.runner.drain()
        job = svc.client.get(f"/reports/{jid}")
        events = svc.client.get(f"/reports/{jid}/events")
    everything = job.text + events.text + caplog.text
    assert secret not in everything and "org_ABC123" not in everything
    assert "***" in caplog.text  # the filter masked the one line that did contain the key
    assert job.json()["state"] == "degraded"


def test_an_internal_error_is_failed_with_a_fixed_message(tmp_path, monkeypatch, caplog):
    secret = "sk-FAKEKEY-9876543210"
    monkeypatch.setenv("GROQ_API_KEY", secret)

    class Bug:
        def generate(self, prompt, **kwargs):
            raise RuntimeError(f"bug with {secret}")

    svc = make_service(tmp_path, make_llm=Bug)
    jid = post(svc).json()["id"]
    with caplog.at_level(logging.DEBUG, logger="reporter"):
        svc.runner.drain()
    job = svc.client.get(f"/reports/{jid}")
    assert job.json()["state"] == "failed" and job.json()["reason"] == "internal_error"
    assert "result" not in job.json()
    assert secret not in job.text + caplog.text
    last = svc.store.events_after(jid, 0)[-1]
    assert last["type"] == "failed" and last["data"] == {
        "state": "failed",
        "reason": "internal_error",
    }


# ------------------------------------------------------------------ health, quota, storage


def test_health_and_quota(tmp_path):
    svc = make_service(tmp_path, daily_max_jobs=2)
    health = svc.client.get("/health").json()
    assert health == {"status": "ok", "storage": "persistent", "version": health["version"]}
    post(svc)
    q = svc.client.get("/quota").json()
    assert q["day"] == "2026-10-31" and q["resets_at"] == "2026-11-01T00:00:00Z"
    assert q["queue"] == {"running": 0, "waiting": 1, "max": 3}
    assert q["jobs"]["limit"] == 2 and q["tokens"]["limit"] > 0 and q["latch_until"] is None
    svc.runner.drain()
    q = svc.client.get("/quota").json()
    assert q["jobs"]["used"] == 1 and q["tokens"]["used"] > 0


def test_home_page_is_served(tmp_path):
    svc = make_service(tmp_path)
    r = svc.client.get("/")
    assert r.status_code == 200 and "EventSource" in r.text


def test_without_persistent_storage_the_service_says_ephemeral(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="reporter"):
        folder, storage = resolve_data_dir({}, tmp_path / "no-data", tmp_path / "var")
    assert (folder, storage) == (tmp_path / "var", "ephemeral")
    assert "no persistent storage" in caplog.text
    (tmp_path / "data").mkdir()
    assert resolve_data_dir({}, tmp_path / "data", tmp_path / "var")[1] == "persistent"
    assert resolve_data_dir({"REPORTER_DATA_DIR": "/x"}, tmp_path, tmp_path)[1] == "persistent"
    svc = make_service(tmp_path / "e", storage="ephemeral")
    assert svc.client.get("/health").json()["storage"] == "ephemeral"


def test_crash_hook_setting_needs_the_test_hooks_switch():
    assert Settings.from_env({"REPORTER_CRASH_AFTER": "worker:s1"}).crash_after == ""
    on = Settings.from_env({"REPORTER_CRASH_AFTER": "worker:s1", "REPORTER_TEST_HOOKS": "1"})
    assert on.crash_after == "worker:s1" and on.test_hooks


def test_the_background_thread_picks_up_and_finishes_a_job(tmp_path):
    svc = make_service(tmp_path, run_thread=True)
    with svc.client:  # runs the lifespan: starts and stops the runner thread
        jid = post(svc).json()["id"]
        for _ in range(200):
            state = svc.client.get(f"/reports/{jid}").json()["state"]
            if state in ("succeeded", "degraded", "failed"):
                break
            time.sleep(0.05)
        assert state == "succeeded"
