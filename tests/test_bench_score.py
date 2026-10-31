"""Benchmark scoring, pairs, judge agreement and aggregation."""

from __future__ import annotations

import json

import pytest

from bench_fakes import GOOD_REPORT, URL1, ScoringLLM, make_trace, topic
from reporter.bench.aggregate import aggregate, latency_stats, percentile
from reporter.bench.pairs import (
    choose_pairs,
    judge_pairs,
    load_human_ratings,
    make_pairs,
)
from reporter.bench.records import CitationScore, CoverageScore, RunRecord
from reporter.bench.runner import PlanItem, build_record
from reporter.bench.scoring import (
    Judgement,
    LLMJsonError,
    Verdict,
    baseline_is_a,
    judge_ordered,
    judge_run_pair,
    normalize_text,
    quote_in_text,
    score_citations,
    score_coverage,
)

# ----------------------------------------------------------------------------- scoring


def test_quote_must_occur_in_the_report_after_normalising():
    text = "Models are “quantized” to **4-bit**\nprecision on phones."
    assert quote_in_text('are "quantized" to 4‑bit precision', text)
    assert not quote_in_text("a sentence that is not there at all", text)
    assert not quote_in_text("short", "short text")  # under 20 characters never counts
    assert normalize_text("A  B\n C") == "a b c"


def test_coverage_counts_only_verified_quotes():
    real = "Small models run on phones using quantization"
    llm = ScoringLLM(quotes={1: real, 2: "an invented sentence about nothing"})
    score = score_coverage(llm, topic(points=4), GOOD_REPORT)
    assert (score.covered, score.total) == (1, 4)
    by_point = {p["point"]: p for p in score.points}
    assert by_point["point 1"]["covered"] and by_point["point 1"]["quote"] == real
    assert by_point["point 2"]["claimed"] and not by_point["point 2"]["covered"]
    assert not by_point["point 3"]["claimed"]
    assert len(llm.prompts) == 1  # one call for all points


def test_coverage_bad_json_raises_after_one_retry():
    class Bad:
        calls = 0

        def generate(self, prompt, **kwargs):
            Bad.calls += 1
            from reporter.llm.base import Generation

            return Generation("not json", "m", 1, 1)

    with pytest.raises(LLMJsonError):
        score_coverage(Bad(), topic(), GOOD_REPORT)
    assert Bad.calls == 2


def test_citation_score_flags_invalid_uncited_and_unfetched():
    clean = score_citations(GOOD_REPORT, make_trace("baseline"))
    assert (clean.cited, clean.valid, clean.clean) == (2, 2, True)

    bad = GOOD_REPORT.replace("[2]", "[7]")  # 7 does not exist, source 2 is never cited
    score = score_citations(bad, make_trace("baseline", report=bad))
    assert score.invalid_citations == [7] and score.uncited_sources == [2]
    assert (score.cited, score.valid, score.clean) == (2, 1, False)

    unfetched = score_citations(GOOD_REPORT, make_trace("baseline", fetched=(URL1,)))
    assert (unfetched.cited, unfetched.valid) == (2, 1) and unfetched.unfetched_urls


# ------------------------------------------------------------------------------ judge


def reports():
    return {"baseline": "baseline text " * 5, "multi": "multi text " * 20}


def test_judge_order_is_seeded_and_identity_free():
    orders = {baseline_is_a(7, "t1", k) for k in range(1, 40)}
    assert orders == {True, False}  # both orders occur
    assert baseline_is_a(7, "t1", 1) == baseline_is_a(7, "t1", 1)
    llm = ScoringLLM(winner="A")
    j = judge_run_pair(llm, topic(), 1, reports(), {"fake-model"}, seed=7)
    prompt = llm.prompts[0]
    assert "baseline" not in prompt.split("=== Report A ===")[0].lower().replace("topic", "")
    assert j.winner_system == j.a_system and j.seed == 7
    assert j.words == {"baseline": 10, "multi": 40}  # recorded to inspect length bias
    assert "IGNORE LENGTH" in prompt
    assert j.judge_model == "judge-model" and j.same_model_as_writer is False
    same = judge_run_pair(
        ScoringLLM(model="fake-model"), topic(), 1, reports(), {"fake-model"}, seed=7
    )
    assert same.same_model_as_writer is True


def test_judge_verdict_parsing_and_tie():
    assert judge_ordered(ScoringLLM(winner="tie"), "t", "a", "b").winner == "tie"
    assert judge_ordered(ScoringLLM(winner="b"), "t", "a", "b").winner == "B"
    with pytest.raises(LLMJsonError):
        judge_ordered(ScoringLLM(winner="C"), "t", "a", "b")


def test_swap_check_turns_position_bias_into_a_tie():
    biased = ScoringLLM(winner="A")  # always picks the first position
    v = judge_ordered(biased, "t", "x", "y", swap_check=True)
    assert v.swap_checked and v.consistent is False and v.winner == "tie"
    steady = ScoringLLM(winner=lambda p: "A" if p.index("good") < p.index("=== Report B") else "B")
    v = judge_ordered(steady, "t", "good", "bad", swap_check=True)
    assert v.consistent is True and v.winner == "A"


# ------------------------------------------------------------------------------ pairs


ORDINAL = {"baseline": "first", "multi": "second"}


ORDINAL = {"baseline": "first", "multi": "second"}


def record(topic_id, system, k, status="success", report=None, **kw):
    report = (
        report
        or f"# {topic_id}\n\nBody of the {ORDINAL[system]} report {k}.\n\n## Sources\n\n1. x\n"
    )
    return RunRecord(
        topic_id=topic_id, system=system, run_index=k, status=status,
        report=report if status == "success" else "", words=20, tokens=1000, latency_s=10.0,
        models=["fake-model"], **kw,
    )  # fmt: skip


def grid(topic_ids, runs=2, fail=()):
    out = {}
    for t in topic_ids:
        for system in ("baseline", "multi"):
            for k in range(1, runs + 1):
                status = "failed" if (t, system, k) in fail else "success"
                out[(t, system, k)] = record(t, system, k, status)
    return out


def test_pairs_hide_identity_seed_the_order_and_write_the_key(tmp_path):
    topics = {t: topic(t) for t in "abcdefgh"}
    records = grid("abcdefgh", runs=1, fail={("c", "multi", 1)})
    specs = make_pairs(records, topics, 5, 6, tmp_path / "p.md", tmp_path / "key.json")
    assert len(specs) == 6 and "c" not in {s.topic_id for s in specs}  # pair needs both systems
    md = (tmp_path / "p.md").read_text(encoding="utf-8")
    assert "baseline" not in md.lower() and "multi" not in md.lower()
    assert md.count("## Report A") == 6 and md.count("## Report B") == 6
    key = json.loads((tmp_path / "key.json").read_text())
    assert key["seed"] == 5 and len(key["pairs"]) == 6
    assert "baseline" not in md and all(
        {p["a_system"], p["b_system"]} == {"baseline", "multi"} for p in key["pairs"]
    )
    assert choose_pairs(records, 6, 5) == specs  # same seed, same pairs and order
    assert choose_pairs(records, 6, 6) != specs


def test_pairs_with_fewer_than_six_valid_say_so(tmp_path):
    topics = {t: topic(t) for t in "ab"}
    specs = make_pairs(grid("ab", runs=1), topics, 0, 6, tmp_path / "p.md", tmp_path / "k.json")
    assert len(specs) == 2
    assert "Only 2 valid pairs exist" in (tmp_path / "p.md").read_text(encoding="utf-8")


def test_second_runs_fill_in_after_distinct_topics():
    specs = choose_pairs(grid("ab", runs=2), 4, 1)
    assert len(specs) == 4 and {s.topic_id for s in specs[:2]} == {"a", "b"}


def test_judge_pairs_without_human_file_is_not_calibrated():
    recs = grid("ab", runs=1)
    topics = {t: topic(t) for t in "ab"}
    specs = choose_pairs(recs, 6, 0)
    out = judge_pairs(ScoringLLM(winner="A"), specs, recs, topics, None)
    assert out["calibrated"] is False and "not calibrated" in out["message"]
    assert out["n_judged"] == 2 and out["pairs"][0]["judge_winner_system"] == specs[0].a_system


def test_judge_pairs_agreement_with_n(tmp_path):
    recs = grid("abcd", runs=1)
    topics = {t: topic(t) for t in "abcd"}
    specs = choose_pairs(recs, 6, 0)
    human_file = tmp_path / "h.jsonl"
    human_file.write_text(
        '{"pair": 1, "winner": "A", "why": "x"}\n{"pair": 2, "winner": "B"}\n'
        '{"pair": 3, "winner": "tie"}\n',
        encoding="utf-8",
    )
    human = load_human_ratings(human_file)
    assert human == {1: "A", 2: "B", 3: "tie"}
    out = judge_pairs(ScoringLLM(winner="A"), specs, recs, topics, human)
    assert out["calibrated"] and out["n_rated"] == 3
    assert (
        out["agreement"]["value"] == pytest.approx(1 / 3, abs=1e-3) and out["agreement"]["n"] == 3
    )
    assert out["agreement_on_decisive_human_ratings"]["n"] == 2
    assert out["pairs"][3].get("human_winner") is None  # unrated pair has no agreement


def test_bad_human_line_is_an_error(tmp_path):
    f = tmp_path / "h.jsonl"
    f.write_text('{"pair": 1, "winner": "C"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_human_ratings(f)


# -------------------------------------------------------------------------- aggregation


def test_percentiles_and_small_n_flag():
    assert percentile([], 50) is None
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)
    small = latency_stats([1.0, 2.0, 3.0])
    assert (
        small["n"] == 3 and small["p95_meaningful"] is False and "not meaningful" in small["note"]
    )
    assert latency_stats([float(i) for i in range(20)])["p95_meaningful"] is True


def judgement(topic_id, k, winner, baseline_words=100, multi_words=300):
    return Judgement(
        topic_id=topic_id, run_index=k, seed=0, a_system="baseline", b_system="multi",
        winner_system=winner, verdict=Verdict(winner="A"),
        words={"baseline": baseline_words, "multi": multi_words}, judge_model="j",
        same_model_as_writer=False,
    )  # fmt: skip


def scored(r, covered, total, cited=4, valid=4, clean=True):
    r.coverage = CoverageScore(model="m", points=[], covered=covered, total=total)
    r.citations = CitationScore(cited=cited, valid=valid, clean=clean)
    return r


def test_aggregation_math_keeps_failed_runs_in_denominators():
    topics = [topic("a"), topic("b")]
    recs = {}
    # baseline: a r1 ok (cost 1, 10 s), a r2 failed (3, 40 s), b r1 ok (2, 20 s), b r2 not run
    for key, status, cost, lat in [
        (("a", "baseline", 1), "success", 1.0, 10.0),
        (("a", "baseline", 2), "failed", 3.0, 40.0),
        (("b", "baseline", 1), "success", 2.0, 20.0),
    ]:
        r = record(
            key[0],
            "baseline",
            key[2],
            status,
            limit="per_day_tokens" if status == "failed" else None,
        )
        r.cost_usd, r.latency_s, r.reason = cost, lat, "writer no_report: x"
        recs[key] = scored(r, 2, 4) if status == "success" else r
    recs[("a", "multi", 1)] = scored(record("a", "multi", 1), 3, 4, cited=5, valid=4, clean=False)
    out = aggregate(
        topics, recs,
        [judgement("a", 1, "multi"), judgement("a", 2, "baseline"), judgement("b", 1, "tie")],
        runs_per_topic=2, calibration_path=None,
    )  # fmt: skip
    base = out["systems"]["baseline"]
    assert base["runs"] == {"planned": 4, "attempted": 3, "succeeded": 2, "failed": 1, "not_run": 1}
    assert base["failure_rate"]["value"] == pytest.approx(1 / 3, abs=1e-3)
    assert base["failure_rate"]["n"] == 3
    assert base["failure_limits"] == {"per_day_tokens": 1}
    assert base["cost_usd"]["total_attempted_runs"] == 6.0
    assert base["cost_usd"]["per_successful_report"] == 3.0  # (1 + 3 + 2) / 2 successes
    assert base["tokens"]["per_successful_report"] == 1500.0  # 3 attempts x 1000 / 2
    assert base["coverage"]["rate"]["value"] == 0.5 and base["coverage"]["rate"]["n"] == 8
    assert base["latency_s"]["successful_runs"]["n"] == 2
    assert base["latency_s"]["successful_runs"]["p50"] == 15.0
    assert base["latency_s"]["all_attempted_runs"]["n"] == 3
    assert base["latency_s"]["successful_runs"]["p95_meaningful"] is False
    multi = out["systems"]["multi"]
    assert multi["runs"]["attempted"] == 1 and multi["runs"]["not_run"] == 3
    assert multi["citations"]["valid_rate"]["value"] == 0.8
    assert multi["citations"]["reports_with_no_problem"]["value"] == 0.0
    # judge: only pairs with both systems successful count; (a, r2) has no multi record at all
    # and (b, r1) has no multi record, so only (a, r1) remains
    assert out["judge"]["pairs"] == 1 and out["judge"]["multi_wins"] == 1


def test_judge_win_rate_with_ties():
    topics = [topic("a"), topic("b"), topic("c"), topic("d")]
    recs = grid("abcd", runs=1)
    judged = [
        judgement("a", 1, "multi"),
        judgement("b", 1, "multi"),
        judgement("c", 1, "baseline"),
        judgement("d", 1, "tie"),
    ]
    j = aggregate(topics, recs, judged, 1, None)["judge"]
    assert (j["pairs"], j["multi_wins"], j["baseline_wins"], j["ties"]) == (4, 2, 1, 1)
    assert j["multi_win_rate_all_pairs"]["value"] == 0.5
    assert j["multi_win_rate_excluding_ties"]["value"] == pytest.approx(2 / 3, abs=1e-3)
    assert j["multi_win_rate_excluding_ties"]["n"] == 3
    assert j["longer_report_won"] == {"value": 0.6667, "numerator": 2, "n": 3}  # multi is longer


def test_not_measured_lists_gaps_and_prices_never_invented():
    topics = [topic("a")]
    recs = {
        ("a", "baseline", 1): record("a", "baseline", 1, tokens_by_model={"fake-model": [10, 5]})
    }
    out = aggregate(topics, recs, [], 1, None)
    text = " ".join(out["not_measured"])
    assert "list-price-equivalent" in text and "judge agreement" in text
    assert "list_price_equivalent_usd" not in out["systems"]["baseline"]
    assert "coverage not scored" in text and "no pair" in text
    assert out["judge_calibration"]["calibrated"] is False

    priced = record("a", "baseline", 1, tokens_by_model={"gemini-3.1-flash-lite": [10, 5]})
    free = aggregate(topics, {("a", "baseline", 1): priced}, [], 1, None)
    assert (
        "list_price_equivalent_usd" not in free["systems"]["baseline"]
    )  # priced 0.0: not measured


def test_calibration_is_copied_from_judge_pairs(tmp_path):
    path = tmp_path / "jp.json"
    path.write_text(
        json.dumps({"calibrated": True, "agreement": {"value": 0.5, "n": 6}, "n_rated": 6}),
        encoding="utf-8",
    )
    out = aggregate([topic("a")], {}, [], 1, path)
    assert out["judge_calibration"]["agreement"]["n"] == 6
    assert "judge agreement with the owner's ratings" not in out["not_measured"]


def test_build_record_for_a_successful_trace():
    item = PlanItem(topic("a"), "multi", 1)
    rec = build_record(item, make_trace("multi", tokens=(200, 100)), "", None)
    assert rec.status == "success" and rec.tokens == 600 and rec.models == ["fake-model"]
    assert rec.tokens_by_model == {"fake-model": [400, 200]}
