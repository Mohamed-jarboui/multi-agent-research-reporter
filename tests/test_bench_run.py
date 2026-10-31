"""Benchmark loop: topics, resume, quota stop, not_run accounting, failure classification."""

from __future__ import annotations

import json

import pytest

from bench_fakes import GOOD_REPORT, AgentFactory, ScoringLLM, make_trace, topic
from reporter.bench.records import RunRecord, load_records
from reporter.bench.runner import (
    EXIT_CONFIG,
    EXIT_OK,
    EXIT_QUOTA,
    BenchConfig,
    Benchmark,
    classify,
    daily_exhausted,
    dominant_limit,
    plan_runs,
)
from reporter.bench.topics import TopicsError, load_topics, parse_topics
from reporter.llm.base import ProviderError
from reporter.llm.ping import PingResult

DAILY = {"gemini": {"per_day_requests": 3}, "groq": {"per_day_tokens": 2}}


def line(topic_id="a", points=3, **extra):
    data = {
        "id": topic_id,
        "topic": "t",
        "domain": "tech",
        "must_cover": [f"p{i}" for i in range(points)],
    }
    return json.dumps(data | extra)


def test_shipped_topics_file_is_valid():
    topics = load_topics()
    assert len(topics) == 12
    assert {"tech", "science", "policy", "business"} <= {t.domain for t in topics}


def test_topics_validation():
    assert len(parse_topics(line("a") + "\n" + line("b"), expected=2)) == 2
    with pytest.raises(TopicsError, match="duplicate"):
        parse_topics(line("a") + "\n" + line("a"), expected=None)
    with pytest.raises(TopicsError, match="3 to 5"):
        parse_topics(line("a", points=2), expected=None)
    with pytest.raises(TopicsError, match="3 to 5"):
        parse_topics(line("a", points=6), expected=None)
    with pytest.raises(TopicsError, match="expected 12"):
        parse_topics(line("a"))
    with pytest.raises(TopicsError, match="line 1"):
        parse_topics("not json", expected=None)


def test_plan_interleaves_systems_per_topic_and_alternates_first():
    items = plan_runs([topic("a"), topic("b")], ("baseline", "multi"), 2)
    assert [i.topic.id for i in items] == ["a"] * 4 + ["b"] * 4
    for start in range(0, 8, 2):  # each consecutive pair is one baseline and one multi
        assert {items[start].system, items[start + 1].system} == {"baseline", "multi"}
    assert items[0].system != items[2].system  # the first system alternates


def config(tmp_path, topics, **kw):
    return BenchConfig(
        topics=topics,
        records_dir=tmp_path / "rec",
        judgements_dir=tmp_path / "judge",
        traces_dir=tmp_path / "traces",
        **kw,
    )


def ok_script(system, n):
    return make_trace(system)


def test_resume_skips_finished_runs(tmp_path):
    topics = [topic("a"), topic("b")]
    first = AgentFactory(ok_script)
    out = Benchmark(config(tmp_path, topics, max_runs=3), first).run()
    assert (out.exit_code, out.executed) == (EXIT_OK, 3)
    assert len(load_records(tmp_path / "rec")) == 3

    second = AgentFactory(ok_script)
    out = Benchmark(config(tmp_path, topics), second).run()
    assert out.executed == 5 and len(second.calls) == 5  # 8 planned minus 3 finished
    assert len(load_records(tmp_path / "rec")) == 8

    third = AgentFactory(ok_script)
    assert Benchmark(config(tmp_path, topics), third).run().executed == 0
    assert third.calls == []


def test_dry_run_calls_nothing(tmp_path):
    factory = AgentFactory(ok_script)
    out = Benchmark(config(tmp_path, [topic("a")]), factory, log=lambda _: None).run(dry_run=True)
    assert out.exit_code == EXIT_OK and factory.calls == []
    assert not (tmp_path / "rec").exists()


def test_systems_and_runs_per_topic_options(tmp_path):
    factory = AgentFactory(ok_script)
    cfg = config(tmp_path, [topic("a")], systems=("multi",), runs_per_topic=1)
    Benchmark(cfg, factory).run()
    assert factory.calls == ["multi"]


def test_quota_stop_marks_remaining_runs_not_run(tmp_path):
    def script(system, n):
        return make_trace(system, ok=False, limits=DAILY) if n == 2 else make_trace(system)

    topics = [topic("a"), topic("b")]
    out = Benchmark(config(tmp_path, topics), AgentFactory(script)).run()
    assert out.exit_code == EXIT_QUOTA
    assert "per-day quota" in out.message
    assert (out.executed, out.not_run) == (2, 6)
    records = load_records(tmp_path / "rec")
    statuses = [r.status for r in records.values()]
    assert statuses.count("success") == 1 and statuses.count("failed") == 1
    assert statuses.count("not_run") == 6
    failed = next(r for r in records.values() if r.status == "failed")
    assert failed.limit == "per_day_requests"
    assert all("per-day quota" in r.reason for r in records.values() if r.status == "not_run")

    # later: not_run runs are redone, the finished ones are kept
    factory = AgentFactory(ok_script)
    out = Benchmark(config(tmp_path, topics), factory).run()
    assert out.exit_code == EXIT_OK and len(factory.calls) == 6


def test_one_provider_with_daily_limit_does_not_stop(tmp_path):
    limits = {"gemini": {"per_day_requests": 3}}
    factory = AgentFactory(lambda s, n: make_trace(s, limits=limits))
    out = Benchmark(config(tmp_path, [topic("a")]), factory).run()
    assert out.exit_code == EXIT_OK and out.executed == 4


def test_consecutive_failures_stop(tmp_path):
    factory = AgentFactory(lambda s, n: make_trace(s, ok=False))
    out = Benchmark(
        config(tmp_path, [topic("a"), topic("b")], max_consecutive_failures=2), factory
    ).run()
    assert out.exit_code == EXIT_QUOTA and out.executed == 2 and out.not_run == 6


def test_exception_is_a_failed_run_not_a_crash(tmp_path):
    factory = AgentFactory(lambda s, n: RuntimeError("boom") if n == 1 else make_trace(s))
    Benchmark(config(tmp_path, [topic("a")], runs_per_topic=1), factory).run()
    records = list(load_records(tmp_path / "rec").values())
    failed = [r for r in records if r.status == "failed"]
    assert len(failed) == 1 and "RuntimeError: boom" in failed[0].reason


def ping(name, ok=True, limit=None):
    if ok:
        return PingResult(name, "ok")
    return PingResult(name, "failed", "RateLimitError 429", limit)


def test_preflight_stops_when_every_provider_has_a_daily_limit(tmp_path):
    results = [ping("gemini", False, "per_day_requests"), ping("groq", False, "per_day_tokens")]
    factory = AgentFactory(ok_script)
    bench = Benchmark(config(tmp_path, [topic("a")]), factory, ping_fn=lambda: results)
    out = bench.run()
    assert out.exit_code == EXIT_QUOTA and factory.calls == [] and out.not_run == 4
    assert {r.status for r in load_records(tmp_path / "rec").values()} == {"not_run"}


def test_preflight_continues_with_one_provider_or_minute_limits(tmp_path):
    for results in (
        [ping("gemini", False, "per_day_requests"), ping("groq")],
        [ping("gemini", False, "per_minute_tokens"), ping("groq", False, "per_day_tokens")],
    ):
        factory = AgentFactory(ok_script)
        bench = Benchmark(
            config(tmp_path / str(len(results)) / str(id(results)), [topic("a")], max_runs=1),
            factory,
            ping_fn=lambda results=results: results,
        )
        assert bench.run().exit_code == EXIT_OK and len(factory.calls) == 1


def test_preflight_without_any_provider_is_a_config_error(tmp_path):
    results = [PingResult("gemini", "not configured"), PingResult("groq", "not configured")]
    bench = Benchmark(
        config(tmp_path, [topic("a")]), AgentFactory(ok_script), ping_fn=lambda: results
    )
    assert bench.run().exit_code == EXIT_CONFIG


def test_classification_of_failures_and_limits():
    assert classify("baseline", make_trace("baseline"), "")[0] == "success"
    assert classify("multi", make_trace("multi"), "")[0] == "success"
    assert classify("baseline", None, "ValueError: x")[0] == "failed"
    stub = make_trace("baseline", ok=False)
    assert classify("baseline", stub, "")[0] == "failed"
    assert classify("multi", stub, "")[0] == "failed"
    fallback = make_trace("multi")
    fallback.writer = {"status": "fallback", "reason": "x"}
    assert classify("multi", fallback, "")[0] == "failed"  # a digest is not a written report
    no_pages = make_trace("baseline")
    no_pages.sources = []
    assert classify("baseline", no_pages, "")[0] == "failed"
    summary = {"limits": {"gemini": {"per_minute_tokens": 5}, "groq": {"per_day_tokens": 1}}}
    assert dominant_limit(summary) == "per_day_tokens"
    assert daily_exhausted(summary) == {"groq"}
    assert dominant_limit({}) is None


def test_failed_run_keeps_its_cost_and_reason(tmp_path):
    factory = AgentFactory(lambda s, n: make_trace(s, ok=False, tokens=(1000, 500)))
    Benchmark(config(tmp_path, [topic("a")], runs_per_topic=1), factory).run()
    for record in load_records(tmp_path / "rec").values():
        assert record.status == "failed" and record.tokens == 3000 and record.reason


def test_runs_are_scored_and_pairs_judged(tmp_path):
    quote = "Small models run on phones using quantization"
    llm = ScoringLLM(quotes={1: quote, 2: "this sentence is not in the report at all"})
    bench = Benchmark(
        config(tmp_path, [topic("a")], runs_per_topic=1), AgentFactory(ok_script), llm
    )
    bench.run()
    records = load_records(tmp_path / "rec")
    for record in records.values():
        assert record.coverage and record.coverage.total == 3
        assert record.coverage.covered == 1  # the invented quote does not count
        assert record.citations and record.citations.valid == 2 and record.citations.clean
    judged = list((tmp_path / "judge").glob("*.json"))
    assert len(judged) == 1
    assert json.loads(judged[0].read_text())["judge_model"] == "judge-model"


def test_scoring_failure_is_recorded_and_retried_next_time(tmp_path):
    class Down:
        def generate(self, prompt, **kwargs):
            raise ProviderError("down")

    cfg = config(tmp_path, [topic("a")], runs_per_topic=1, systems=("baseline",))
    Benchmark(cfg, AgentFactory(ok_script), Down(), log=lambda _: None).run()
    record = next(iter(load_records(tmp_path / "rec").values()))
    assert record.status == "success" and record.coverage is None
    assert record.scoring_error.startswith("coverage")

    llm = ScoringLLM(quotes={1: "Small models run on phones using quantization"})
    factory = AgentFactory(ok_script)
    Benchmark(cfg, factory, llm, log=lambda _: None).run()
    assert factory.calls == []  # not re-run
    assert next(iter(load_records(tmp_path / "rec").values())).coverage is not None


def test_record_roundtrip_and_report_kept(tmp_path):
    Benchmark(config(tmp_path, [topic("a")], runs_per_topic=1), AgentFactory(ok_script)).run()
    record = next(iter(load_records(tmp_path / "rec").values()))
    assert isinstance(record, RunRecord) and record.report == GOOD_REPORT
    assert record.words > 5 and record.tokens == 300 and record.latency_s == 10.0


def test_cli_bench_dispatch(tmp_path, capsys):
    from reporter import cli

    assert cli.main(["bench"]) == 2
    args = [
        "bench",
        "run",
        "--dry-run",
        "--records-dir",
        str(tmp_path),
        "--topics",
        "carbon-capture",
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "planned 4 runs" in out and "would run carbon-capture" in out
    assert cli.main(["bench", "run", "--dry-run", "--topics", "nope"]) == 2
