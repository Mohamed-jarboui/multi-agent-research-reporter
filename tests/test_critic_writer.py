"""R2: deterministic checks, the critic, the retry cap, best-attempt choice and the writer."""

import json
from datetime import date

import pytest

from fakes import TeamLLM, critic_json, team_tools
from reporter import cli
from reporter.agent import MultiAgent
from reporter.agent.checks import (
    check_citations,
    clean_text,
    deterministic_reviews,
    fetched_urls,
    plain_reason,
)
from reporter.agent.critic import attempt_rank, parse_review, retry_candidates
from reporter.agent.parsing import citation_numbers
from reporter.agent.schemas import Finding, Review, SectionBrief, SectionOut, SectionReview
from reporter.budget import Budget, MultiBudget
from reporter.llm.base import ProviderError
from reporter.trace import SourceRecord, StepRecord, ToolCallRecord, Trace

TODAY = date(2026, 10, 30)


def run(llm, tmp_path, budget=None, **kw):
    agent = MultiAgent(llm, team_tools(), budget, today=TODAY, runs_dir=tmp_path / "runs", **kw)
    return agent.run("Topic")


class RecordingLLM(TeamLLM):
    """TeamLLM that keeps the prompts."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.critic_prompts: list[str] = []
        self.all_prompts: list[str] = []

    def generate(self, prompt, **kwargs):
        self.all_prompts.append(prompt)
        if "critic" in (kwargs.get("system") or ""):
            self.critic_prompts.append(prompt)
        return super().generate(prompt, **kwargs)


def sources(n: int) -> list[SourceRecord]:
    return [
        SourceRecord(id=i, url=f"https://x{i}.example/p", title=f"X{i}") for i in range(1, n + 1)
    ]


# ------------------------------------------------------------ deterministic checks


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A [5].", [5]),
        ("A [5, 6].", [5, 6]),
        ("A [5][6].", [5, 6]),
        ("A [5-7].", [5, 6, 7]),
        ("A [5,7-9] and [1].", [5, 7, 8, 9, 1]),
        ("A [3–5].", [3, 4, 5]),
        ("A [ 2 ; 4 ] [2].", [2, 4]),
        ("A [7-5].", [7, 5]),  # backwards range: not expanded, ends are kept
        ("A [link](http://x) and [abc].", []),
    ],
)
def test_citation_numbers_parses_every_form(text, expected):
    assert citation_numbers(text) == expected


def test_check_citations_reports_invalid_uncited_and_unfetched():
    text = "Fact [1, 2]. Other [4-6]. See https://rogue.example/page."
    check = check_citations(text, sources(5), ["https://x1.example/p", "https://x2.example/p/"])

    assert check.invalid_citations == [6]
    assert check.uncited_sources == [3]
    assert check.unfetched_urls == [
        "https://x4.example/p",
        "https://x5.example/p",
        "https://rogue.example/page",
    ]
    assert not check.ok


def test_check_citations_ok_when_everything_matches():
    srcs = sources(2)
    check = check_citations("A [1][2]. B [1-2].", srcs, [s.url for s in srcs])
    assert check.ok and check.cited == [1, 2]


def test_fetched_urls_come_from_successful_fetch_calls():
    trace = Trace(run_id="r", system="multi", topic="t", started_at="now")
    for name, ok, url in [
        ("fetch_page", True, "https://a.example"),
        ("fetch_page", False, "https://b.example"),
        ("web_search", True, ""),
    ]:
        step = StepRecord(index=0, kind="think")
        step.tool_call = ToolCallRecord(
            name=name, input={"url": url}, ok=ok, latency_s=0, observation_chars=0
        )
        trace.add_step(step)
    assert fetched_urls(trace, ["https://c.example"]) == {"https://a.example", "https://c.example"}


def section(sid: str, *, pages: int, findings: int = 1) -> SectionOut:
    brief = SectionBrief(id=sid, title=f"T{sid}", goal="g", scope=["s"])
    return SectionOut(
        brief=brief,
        status="complete" if findings else "incomplete",
        confidence="low",
        findings=[Finding(claim="c", source_ids=[1])] * findings,
        n_pages=pages,
    )


def test_section_without_pages_or_findings_fails_without_a_model():
    from reporter.agent.schemas import MergedResearch

    merged = MergedResearch(sections=[section("s1", pages=0, findings=0), section("s2", pages=2)])
    reviews = deterministic_reviews(merged)
    assert list(reviews) == ["s1"] and reviews["s1"].verdict == "fail"
    assert reviews["s1"].issues[0].origin == "deterministic"


# ----------------------------------------------------------------- critic parsing


def test_parse_review_fills_missing_sections_and_drops_bad_entries():
    reply = json.dumps(
        {
            "sections": [
                {"id": "s1", "verdict": "weak", "issues": []},
                {"id": "s9", "verdict": "pass"},
                {"id": "s2", "verdict": "maybe"},
            ],
            "contradictions": [{"sections": ["s1", "s7"], "issue": "x vs y"}, {"issue": ""}],
            "coverage_gaps": ["gap", ""],
        }
    )
    reviews, contradictions, gaps = parse_review(reply, ["s1", "s2", "s3"])

    assert reviews["s1"].verdict == "weak" and reviews["s1"].issues  # detail added
    assert reviews["s2"].verdict == "unreviewed" and reviews["s3"].verdict == "unreviewed"
    assert contradictions[0].section_ids == ["s1"] and gaps == ["gap"]
    with pytest.raises(ValueError):
        parse_review('{"sections": "no"}', ["s1"])


def test_attempt_rank_orders_verdict_then_pages_then_findings():
    assert attempt_rank("pass", section("s1", pages=1)) > attempt_rank(
        "weak", section("s1", pages=4)
    )
    assert attempt_rank("weak", section("s1", pages=3)) > attempt_rank(
        "weak", section("s1", pages=2)
    )
    assert attempt_rank("weak", section("s1", pages=2, findings=3)) > attempt_rank(
        "weak", section("s1", pages=2, findings=1)
    )
    assert attempt_rank("unreviewed", section("s1", pages=1)) > attempt_rank(
        "weak", section("s1", pages=1)
    )


def test_retry_candidates_skip_retried_sections_and_cap_the_count():
    review = Review(
        sections={
            f"s{i}": SectionReview(section_id=f"s{i}", verdict=v)
            for i, v in enumerate(["weak", "fail", "pass", "weak", "weak"], 1)
        }
    )
    assert retry_candidates(review, set()) == ["s2", "s1", "s4"]  # fail first, then by id, max 3
    assert retry_candidates(review, {"s2", "s1", "s4"}) == ["s5"]
    assert retry_candidates(review, {"s1", "s2", "s4", "s5"}) == []


# ------------------------------------------------------------------- full graph


def test_full_run_checks_critiques_and_writes_a_cited_report(tmp_path):
    llm = TeamLLM(3)
    result = run(llm, tmp_path)

    trace = result.trace
    assert llm.counts[("critic", "")] == 1 and llm.counts[("writer", "")] == 1
    assert trace.critic["status"] == "ok" and set(trace.critic["before"].values()) == {"pass"}
    assert trace.retries == [] and trace.writer == {"status": "ok"}
    assert result.report.startswith("# Report") and result.report.count("## Sources") == 1
    assert "1. [S1-1](https://s1-1.example/p)" in result.report
    assert trace.checks["stage"] == "report"
    assert trace.checks["invalid_citations"] == [] and trace.checks["unfetched_urls"] == []
    assert trace.checks["uncited_sources"] == [5, 6]  # the fake writer cites [1][2] and [3-4]
    assert result.digest.startswith("# Topic") and trace.report == result.report
    assert {s.kind for s in trace.steps} >= {"critic", "write"}


def test_a_weak_section_is_retried_once_and_the_better_attempt_is_kept(tmp_path):
    def on_call(kind, title, nth):
        if kind == "critic" and nth == 1:
            return critic_json(["s1", "s2", "s3"], {"s1": "weak"})
        # the retry is graded `pass` by the default reply

    llm = RecordingLLM(3, on_call)
    result = run(llm, tmp_path)

    trace = result.trace
    assert llm.counts[("final", "Part 1")] == 2 and llm.counts[("final", "Part 2")] == 1
    assert [w.phase for w in trace.workers if w.section_id == "s1"] == ["research", "retry"]
    (record,) = trace.retries
    assert record["section_id"] == "s1" and record["kept"] == "retry"
    assert record["before"]["verdict"] == "weak" and record["after"]["verdict"] == "pass"
    assert trace.critic["before"]["s1"] == "weak" and trace.critic["after"] == {"s1": "pass"}
    assert trace.critic["final"] == {"s1": "pass", "s2": "pass", "s3": "pass"}
    assert any(
        "s1 is thin" in p and "Part 1" in p for p in llm.all_prompts
    )  # issues reached the worker
    assert trace.checks["invalid_citations"] == []
    assert [s.id for s in trace.sources] == list(range(1, 7))


def test_retry_cap_a_section_goes_back_once_even_if_it_stays_weak(tmp_path):
    def on_call(kind, title, nth):
        if kind == "critic":
            ids = ["s1", "s2", "s3"] if nth == 1 else ["s1"]
            return critic_json(ids, {"s1": "weak"})

    llm = TeamLLM(3, on_call)
    result = run(llm, tmp_path)

    trace = result.trace
    assert llm.counts[("critic", "")] == 2  # one review, one re-review of the retried section
    assert llm.counts[("final", "Part 1")] == 2  # first research + exactly one retry
    assert [w.phase for w in trace.workers if w.section_id == "s1"] == ["research", "retry"]
    (record,) = trace.retries
    assert record["kept"] == "original"  # no strict improvement: the first attempt stays
    assert trace.critic["final"]["s1"] == "weak"
    assert "Weak:" in result.report and "Part 1" in result.report  # the report says so


def test_a_section_with_no_pages_is_flagged_without_the_critic_and_named_in_the_report(tmp_path):
    def on_call(kind, title, nth):
        if kind == "think" and title == "Part 2":
            return ProviderError("down")

    llm = RecordingLLM(3, on_call)
    result = run(llm, tmp_path)

    trace = result.trace
    assert trace.critic["before"]["s2"] == "fail"
    assert trace.critic["issues"]["s2"][0]["origin"] == "deterministic"
    assert all("## s2:" not in p for p in llm.critic_prompts)  # never sent to the model
    assert len(trace.retries) == 1 and trace.retries[0]["kept"] == "original"
    assert len([w for w in trace.workers if w.section_id == "s2" and w.phase == "retry"]) == 1
    assert 'Missing: "Part 2"' in result.report
    assert trace.checks["research"] == {"s2": trace.checks["research"]["s2"]}


def test_nothing_usable_gives_an_honest_short_report_and_no_writer_call(tmp_path):
    def on_call(kind, title, nth):
        if kind == "think":
            return ProviderError("down")

    llm = TeamLLM(3, on_call)
    result = run(llm, tmp_path)

    assert llm.counts[("writer", "")] == 0 and llm.counts[("critic", "")] == 0
    assert result.trace.writer["status"] == "no_report"
    assert "No report could be written" in result.report and "## Sources" not in result.report
    assert result.report.count("Missing:") == 3


def test_critic_failure_keeps_the_deterministic_checks_and_the_run_goes_on(tmp_path):
    llm = TeamLLM(3, lambda kind, title, nth: ProviderError("429") if kind == "critic" else None)
    result = run(llm, tmp_path)

    trace = result.trace
    assert trace.critic["status"] == "failed"
    assert any(n.startswith("critic_failed") for n in trace.notes)
    assert set(trace.critic["before"].values()) == {"unreviewed"} and trace.retries == []
    assert trace.writer == {"status": "ok"}
    assert "were not independently checked" in result.report


def test_an_unparseable_critic_reply_is_retried_once_then_given_up(tmp_path):
    llm = TeamLLM(3, lambda kind, title, nth: "not json" if kind == "critic" else None)
    result = run(llm, tmp_path)

    assert llm.counts[("critic", "")] == 2
    assert result.trace.critic["status"] == "failed"


def test_writer_failure_falls_back_to_the_digest_with_a_warning(tmp_path):
    llm = TeamLLM(3, lambda kind, title, nth: ProviderError("down") if kind == "writer" else None)
    result = run(llm, tmp_path)

    assert result.report.startswith("> Warning: the writer step failed")
    assert "## Part 1" in result.report and "## Sources" in result.report
    assert result.trace.writer["status"] == "fallback"
    assert any(n.startswith("writer_failed") for n in result.trace.notes)
    assert result.trace.checks["stage"] == "digest"


def test_an_empty_writer_reply_is_retried_then_falls_back(tmp_path):
    llm = TeamLLM(3, lambda kind, title, nth: "" if kind == "writer" else None)
    result = run(llm, tmp_path)

    assert llm.counts[("writer", "")] == 2
    assert result.trace.writer["status"] == "fallback"


def test_the_model_never_writes_the_sources_and_bad_citations_are_recorded(tmp_path):
    bad = (
        "# R\n\nClaim [1, 99]. Also [2-3]. See https://rogue.example/x.\n\n"
        "## Sources\n\n1. [Invented](https://invented.example)\n"
    )
    llm = TeamLLM(3, lambda kind, title, nth: bad if kind == "writer" else None)
    result = run(llm, tmp_path)

    assert "invented.example" not in result.report
    assert result.report.count("## Sources") == 1
    checks = result.trace.checks
    assert checks["invalid_citations"] == [99]
    assert checks["unfetched_urls"] == ["https://rogue.example/x"]
    assert checks["uncited_sources"] == [4, 5, 6]


def test_no_critic_no_writer_is_the_r1_behaviour(tmp_path):
    llm = TeamLLM(3)
    result = run(llm, tmp_path, critic=False, writer=False)

    assert llm.counts[("critic", "")] == 0 and llm.counts[("writer", "")] == 0
    assert result.report == result.digest and result.trace.retries == []
    assert result.trace.writer == {"status": "skipped"}


def test_research_stops_at_70_percent_and_retries_are_skipped_when_budget_is_short(tmp_path):
    assert MultiBudget().run.stop_fraction == 0.7
    budget = MultiBudget(run=Budget(max_steps=36, max_tokens=1_700, stop_fraction=0.7))

    def on_call(kind, title, nth):
        if kind == "critic":
            return critic_json(["s1", "s2", "s3"], {"s1": "weak"})

    result = run(TeamLLM(3, on_call), tmp_path, budget)

    trace = result.trace
    assert trace.critic["before"]["s1"] == "weak"
    assert any(w.stop_reason == "budget_tokens" for w in trace.workers)
    assert trace.retries == [] and "retry_skipped" in trace.critic


# ------------------------------------------------------------------------- CLI


def test_cli_flags_and_the_one_line_summary(tmp_path, monkeypatch, capsys):
    args = cli.build_parser().parse_args(["multi", "T", "--no-critic", "--no-writer"])
    assert args.no_critic and args.no_writer

    monkeypatch.setattr("reporter.llm.build_default_llm", lambda: TeamLLM(3))
    monkeypatch.setattr("reporter.tools.default_tools", team_tools)
    monkeypatch.chdir(tmp_path)

    assert cli.main(["multi", "Topic"]) == 0

    captured = capsys.readouterr()
    assert captured.out.startswith("# Report") and "## Sources" in captured.out
    assert "sections ok/weak/missing: 3/0/0" in captured.err
    assert "retries: 0 (kept 0)" in captured.err and "writer: ok" in captured.err


# ---------------------------------------------------- retry gate, sanitising, unverified


def weak_first(kind, title, nth):
    if kind == "critic" and nth == 1:
        return critic_json(["s1", "s2", "s3"], {"s1": "weak"})


def test_a_weak_section_is_retried_after_research_used_its_share_of_steps(tmp_path):
    # 12 run steps: research stops at 70% (8.4 steps) and has used more than that when the
    # critic runs; the retry gate looks at what one retry needs, not at the 70% line.
    budget = MultiBudget(run=Budget(max_steps=12, stop_fraction=0.7))
    llm = TeamLLM(3, weak_first)

    result = run(llm, tmp_path, budget)

    assert result.trace.critic["before"]["s1"] == "weak"
    assert "retry_skipped" not in result.trace.critic
    assert [w.phase for w in result.trace.workers if w.section_id == "s1"] == ["research", "retry"]
    assert len(result.trace.retries) == 1


def test_default_budget_leaves_room_for_the_retry_with_five_sections(tmp_path):
    def on_call(kind, title, nth):
        if kind == "critic" and nth == 1:
            return critic_json([f"s{i}" for i in range(1, 6)], {"s2": "weak"})

    result = run(TeamLLM(5, on_call), tmp_path)  # default MultiBudget

    assert result.trace.totals.react_steps >= 15
    assert [r["section_id"] for r in result.trace.retries] == ["s2"]


def test_plain_reason_and_clean_text_never_pass_raw_detail():
    raw = (
        'AllProvidersFailedError: last error: RateLimitError("Error code: 429 org_REDACTED '
        'https://console.groq.com/settings/billing")'
    )
    assert plain_reason(raw) == "rate limited"
    assert plain_reason(None, "budget_steps") == "step budget reached"
    assert plain_reason("worker_time") == "time limit reached"
    assert plain_reason("[model_output_unparseable]") == "model output unusable"
    assert plain_reason("weird") == "research did not complete"
    assert clean_text("see https://a.example/x [budget_steps] now.") == "see now"


def test_provider_error_text_never_reaches_the_report(tmp_path):
    secret = (
        "AllProvidersFailedError: Rate limit for organization `org_REDACTED` "
        "https://console.groq.com/settings/billing [budget_steps]"
    )

    def on_call(kind, title, nth):
        if kind == "think" and title == "Part 2":
            return ProviderError(secret)
        if kind == "writer":
            return ProviderError(secret)

    for writer in (True, False):
        result = run(TeamLLM(3, on_call), tmp_path, writer=writer)
        for bad in ("org_REDACTED", "console.groq", "AllProviders", "[budget", "Error code"):
            assert bad not in result.report
        assert "rate limited" not in result.report  # no fetch returned 403 or 429
        assert "no page could be fetched" in result.report
    assert "org_REDACTED" in json.dumps(result.trace.workers[0].model_dump()) or any(
        "org_REDACTED" in (w.error or "") for w in result.trace.workers
    )  # the detail is kept in the trace


def critic_with_issue(evidence: str):
    def on_call(kind, title, nth):
        if kind == "critic" and nth == 1:
            reply = json.loads(critic_json(["s1", "s2", "s3"]))
            reply["sections"][0] = {
                "id": "s1",
                "verdict": "pass",  # no retry: the issues are kept as they are
                "issues": [
                    {"rubric": 2, "issue": "not in the source", "evidence": evidence},
                    {"rubric": 4, "issue": "thin scope", "evidence": ""},
                ],
            }
            return json.dumps(reply)

    return on_call


def test_a_claim_flagged_with_quoted_evidence_is_omitted_and_listed(tmp_path):
    quote = 'F1: the page says "about 30 percent", not 40'
    llm = RecordingLLM(3, critic_with_issue(quote))
    result = run(llm, tmp_path)

    writer_prompt = next(p for p in llm.all_prompts if "Findings by section" in p)
    assert "Part 1 fact one" not in writer_prompt and "Part 1 fact two" in writer_prompt
    assert "Unverified, not used in the text: Part 1 fact one" in result.report
    assert "## Unverified claims" not in result.report
    assert (
        'Critic issues: "Part 1" has 2 issue(s); 1 claim(s) omitted (evidence quoted), '
        "0 listed as unverified." in result.report
    )


def test_a_claim_flagged_without_evidence_is_kept_but_labelled_unverified(tmp_path):
    llm = RecordingLLM(3, critic_with_issue("F1"))
    result = run(llm, tmp_path)

    writer_prompt = next(p for p in llm.all_prompts if "Findings by section" in p)
    assert "Part 1 fact one" not in writer_prompt  # the model never states it as a fact
    assert "## Unverified claims" in result.report
    assert "- (unverified) Part 1 fact one [1]" in result.report
    assert "1 listed as unverified" in result.report
    assert result.trace.checks["invalid_citations"] == []


# ------------------------------------------------------------- writer output rules

DIRTY_REPLY = """# R

Intro with a fact [1]. Note [Critic] and [budget_steps] here.

## Part 1

- Cited bullet [1]
- Uncited bullet in a good section

## Part 2

- Made-up bullet one
- Made-up bullet two
- Made-up bullet three

---
*All factual statements are sourced from the provided findings and citations.*
"""


def part2_down(kind, title, nth):
    if kind == "think" and title == "Part 2":
        return ProviderError("down")
    if kind == "writer":
        return DIRTY_REPLY


def test_writer_output_is_sanitised_in_code(tmp_path):
    result = run(TeamLLM(3, part2_down), tmp_path)

    report = result.report
    assert "[Critic]" not in report and "[budget_steps]" not in report
    assert "Made-up bullet" not in report and "## Part 2" not in report  # missing: notes only
    assert "Uncited bullet in a good section" not in report and "- Cited bullet [1]" in report
    assert "All factual statements" not in report and "\n---\n" not in report
    assert 'Missing: "Part 2"' in report
    assert result.trace.writer == {
        "status": "ok",
        "stray_tokens": 2,
        "assurance_lines": 1,
        "dropped_sections": 1,
        "uncited_bullets": 1,
    }


def test_the_report_ends_with_a_code_built_claim_count(tmp_path):
    result = run(TeamLLM(3, part2_down), tmp_path)

    # 2 sections with 2 findings each are verified; the third is missing
    assert "**Claim check:** 4 claims cited, 0 claims left out or marked unverified, " in (
        result.report
    )
    assert "1 sections missing or weak." in result.report


def test_claim_check_counts_unverified_and_weak_sections(tmp_path):
    result = run(RecordingLLM(3, critic_with_issue("F1")), tmp_path)

    assert "5 claims cited, 1 claims left out or marked unverified, 0 sections" in result.report


def test_sanitiser_keeps_non_section_headings_and_cited_text():
    from reporter.agent.schemas import MergedResearch
    from reporter.agent.writer import sanitize_body

    merged = MergedResearch(sections=[section("s1", pages=2), section("s2", pages=0, findings=0)])
    text = (
        "# T\n\n## Limits\n\nSources disagree [1].\n\n## Ts2\n\n- nothing\n\n## Ts1\n\n- ok [1]\n"
    )
    out, counts = sanitize_body(text, merged, Review())

    assert "## Limits" in out and "## Ts1" in out and "## Ts2" not in out
    assert counts["dropped_sections"] == 1 and counts["uncited_bullets"] == 0


# ------------------------------------------------------------- coverage-note reasons


def reason_section(**kw) -> SectionOut:
    base = section("s1", pages=0, findings=0)
    return base.model_copy(update=kw)


def test_reason_pages_fetched_but_no_findings_is_not_called_rate_limited():
    from reporter.agent.checks import section_reason

    err = 'RateLimitError("Error code: 429") model call failed'
    reason = section_reason(reason_section(n_pages=2, error=err))

    assert "pages were fetched but no findings could be extracted" in reason
    assert "rate" not in reason and "429" not in reason


def test_reason_no_pages_and_blocked_fetches_and_budget():
    from reporter.agent.checks import section_reason

    assert section_reason(reason_section(error="no page was fetched (stopped: max_steps)")) == (
        "no page could be fetched"
    )
    blocked = section_reason(reason_section(blocked_fetches=2))
    assert "refused" in blocked and "403" in blocked
    assert "step budget" in section_reason(reason_section(stop_reason="budget_steps"))
    assert "time budget" in section_reason(reason_section(n_pages=1, stop_reason="worker_time"))


def test_a_real_403_or_429_fetch_is_counted_on_the_section():
    from reporter.agent.multi import merge_results
    from reporter.agent.schemas import WorkerResult

    brief = SectionBrief(id="s1", title="T", goal="g", scope=["s"])
    result = WorkerResult(
        section_id="s1",
        index=0,
        status="incomplete",
        failed_fetches=[
            {"url": "https://a.example", "reason": "HTTP 429"},
            {"url": "https://b.example", "reason": "HTTP 404"},
            {"url": "https://c.example", "reason": "HTTP 403"},
        ],
    )
    merged = merge_results([brief], [result])

    assert merged.sections[0].blocked_fetches == 2


def test_coverage_notes_use_plain_words_without_bracket_tags(tmp_path):
    import re

    result = run(TeamLLM(3, part2_down), tmp_path)

    notes = result.report.split("## Coverage notes")[1].split("## Sources")[0]
    assert "no page could be fetched" in notes
    assert not re.search(r"\[[A-Za-z_]+\]", notes)
