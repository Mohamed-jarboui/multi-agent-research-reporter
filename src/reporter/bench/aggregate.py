"""Aggregate the run records into `results/benchmark.json`.

Every figure carries its sample size. A figure that cannot be computed is left out and named in
`not_measured`; nothing is estimated. Failed runs stay in every denominator they belong to.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from reporter.bench.records import JUDGEMENTS_DIR, RECORDS_DIR, SYSTEMS, RunRecord, load_records
from reporter.bench.scoring import Judgement
from reporter.bench.topics import TOPICS_PATH, Topic, TopicsError, load_topics
from reporter.pricing import PRICES_PER_MTOK

RESULTS_PATH = Path("results/benchmark.json")
JUDGE_PAIRS_PATH = Path("results/judge_pairs.json")
MIN_RUNS_FOR_P95 = 20


def ratio(numerator: float, denominator: float, **extra: Any) -> dict[str, Any]:
    """A rate with the counts it came from; None when the denominator is zero."""
    value = round(numerator / denominator, 4) if denominator else None
    return {"value": value, "numerator": numerator, "n": denominator, **extra}


def percentile(values: list[float], p: float) -> float | None:
    """Linear interpolation between the closest ranks (the usual 'inclusive' definition)."""
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    low, high = math.floor(pos), math.ceil(pos)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (pos - low), 3)


def latency_stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    return {
        "n": n,
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p95_meaningful": n >= MIN_RUNS_FOR_P95,
        "note": "" if n >= MIN_RUNS_FOR_P95 else f"p95 is not meaningful with n={n}",
    }


def list_price_usd(records: list[RunRecord]) -> float | None:
    """Cost at the per-token prices in pricing.py, only when every model used has a non-zero
    price; the free-tier prices are 0.0, so this is normally None (not measured)."""
    total = 0.0
    for record in records:
        for model, (tokens_in, tokens_out) in record.tokens_by_model.items():
            price = PRICES_PER_MTOK.get(model)
            if price is None or not any(price):
                return None
            total += (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
    return round(total, 6) if records else None


def system_stats(
    system: str, records: list[RunRecord], planned: int, topics: list[Topic]
) -> tuple[dict[str, Any], list[str]]:
    finished = [r for r in records if r.finished]
    ok = [r for r in finished if r.status == "success"]
    failed = [r for r in finished if r.status == "failed"]
    missing: list[str] = []
    attempted = len(finished)

    stats: dict[str, Any] = {
        "runs": {
            "planned": planned,
            "attempted": attempted,
            "succeeded": len(ok),
            "failed": len(failed),
            "not_run": planned - attempted,
        },
        "failure_rate": ratio(len(failed), attempted),
        "failure_reasons": dict(Counter(r.reason.split(":")[0] or "unknown" for r in failed)),
        "failure_limits": dict(Counter(r.limit or "none" for r in failed)),
    }

    scored = [r for r in ok if r.coverage is not None]
    covered = sum(r.coverage.covered for r in scored if r.coverage)
    total = sum(r.coverage.total for r in scored if r.coverage)
    per_topic = {}
    for topic in topics:
        rows = [r for r in scored if r.topic_id == topic.id and r.coverage]
        per_topic[topic.id] = ratio(
            sum(r.coverage.covered for r in rows if r.coverage),
            sum(r.coverage.total for r in rows if r.coverage),
            runs=len(rows),
        )
    stats["coverage"] = {
        "rate": ratio(covered, total, runs_scored=len(scored), runs_succeeded=len(ok)),
        "per_topic": per_topic,
    }
    if len(scored) < len(ok):
        missing.append(
            f"{system}: coverage not scored for {len(ok) - len(scored)} of {len(ok)} runs"
        )

    cited = [r.citations for r in ok if r.citations]
    stats["citations"] = {
        "valid_rate": ratio(
            sum(c.valid for c in cited), sum(c.cited for c in cited), runs=len(cited)
        ),
        "reports_with_no_problem": ratio(sum(c.clean for c in cited), len(cited)),
        "invalid_numbers": sum(len(c.invalid_citations) for c in cited),
        "unfetched_urls": sum(len(c.unfetched_urls) for c in cited),
        "uncited_sources": sum(len(c.uncited_sources) for c in cited),
    }

    all_tokens = sum(r.tokens for r in finished)
    all_cost = round(sum(r.cost_usd for r in finished), 6)
    per_ok = lambda x: round(x / len(ok), 6) if ok else None  # noqa: E731
    stats["tokens"] = {
        "total_attempted_runs": all_tokens,
        "n_runs": attempted,
        "per_successful_report": per_ok(all_tokens),
        "n_successful": len(ok),
        "runs_with_unreported_token_counts": sum(
            r.calls_without_token_counts > 0 for r in finished
        ),
    }
    stats["cost_usd"] = {
        "total_attempted_runs": all_cost,
        "n_runs": attempted,
        "per_successful_report": per_ok(all_cost),
        "n_successful": len(ok),
        "note": "recorded cost_usd; free-tier models are priced 0.0",
        "runs_with_unpriced_calls": sum(r.calls_without_price > 0 for r in finished),
    }
    equivalent = list_price_usd(finished)
    if equivalent is None:
        missing.append(f"{system}: list-price-equivalent cost (no non-zero prices in pricing.py)")
    else:
        stats["list_price_equivalent_usd"] = {
            "total_attempted_runs": equivalent,
            "per_successful_report": per_ok(equivalent),
            "n_runs": attempted,
        }
    stats["latency_s"] = {
        "successful_runs": latency_stats([r.latency_s for r in ok]),
        "all_attempted_runs": latency_stats([r.latency_s for r in finished]),
    }
    words = [r.words for r in ok]
    stats["words"] = {"median": percentile([float(w) for w in words], 50), "n": len(words)}
    return stats, missing


def judge_stats(
    judgements: list[Judgement], records: dict[tuple[str, str, int], RunRecord]
) -> dict[str, Any]:
    """Win rate over pairs where both systems succeeded; ties reported, not dropped."""
    valid = [
        j
        for j in judgements
        if all(
            (r := records.get((j.topic_id, s, j.run_index))) is not None and r.status == "success"
            for s in SYSTEMS
        )
    ]
    n = len(valid)
    wins = Counter(j.winner_system for j in valid)
    other = {"baseline": "multi", "multi": "baseline"}
    unequal = [
        j
        for j in valid
        if j.winner_system != "tie" and j.words[j.winner_system] != j.words[other[j.winner_system]]
    ]
    longer = sum(j.words[j.winner_system] > j.words[other[j.winner_system]] for j in unequal)
    decided = n - wins["tie"]
    return {
        "pairs": n,
        "multi_wins": wins["multi"],
        "baseline_wins": wins["baseline"],
        "ties": wins["tie"],
        "multi_win_rate_all_pairs": ratio(wins["multi"], n),
        "multi_win_rate_excluding_ties": ratio(wins["multi"], decided),
        "longer_report_won": ratio(longer, len(unequal)),
        "judge_models": sorted({j.judge_model for j in valid}),
        "pairs_judged_by_a_model_that_also_wrote_a_report": sum(
            bool(j.same_model_as_writer) for j in valid
        ),
        "swap_checked_pairs": sum(j.verdict.swap_checked for j in valid),
        "swap_inconsistent_pairs": sum(j.verdict.consistent is False for j in valid),
    }


def load_judgements(directory: Path = JUDGEMENTS_DIR) -> list[Judgement]:
    paths = sorted(directory.glob("*.json")) if directory.is_dir() else []
    return [Judgement.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


def aggregate(
    topics: list[Topic],
    records: dict[tuple[str, str, int], RunRecord],
    judgements: list[Judgement],
    runs_per_topic: int = 2,
    calibration_path: Path | None = JUDGE_PAIRS_PATH,
) -> dict[str, Any]:
    planned = len(topics) * runs_per_topic
    ids = {t.id for t in topics}
    out: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "design": {
            "topics": len(topics),
            "runs_per_topic": runs_per_topic,
            "planned_runs_per_system": planned,
        },
        "systems": {},
    }
    not_measured: list[str] = []
    for system in SYSTEMS:
        mine = [r for r in records.values() if r.system == system and r.topic_id in ids]
        out["systems"][system], missing = system_stats(system, mine, planned, topics)
        not_measured += missing
    pairs = [j for j in judgements if j.topic_id in ids]
    out["judge"] = judge_stats(pairs, records)
    if out["judge"]["pairs"] == 0:
        not_measured.append("pairwise judge: no pair where both systems succeeded was judged")
    calibration = None
    if calibration_path and calibration_path.is_file():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration and calibration.get("calibrated"):
        out["judge_calibration"] = {
            k: calibration.get(k) for k in ("calibrated", "agreement", "n_rated", "message")
        }
    else:
        out["judge_calibration"] = {
            "calibrated": False,
            "message": "the judge is not calibrated: no ratings by the owner yet",
        }
        not_measured.append("judge agreement with the owner's ratings")
    out["notes"] = [
        "Coverage, judge and citation scoring tokens are not included in system tokens or cost.",
        "Failed runs are included in attempted counts, total tokens, total cost and cost per "
        "successful report.",
    ]
    used = sum(1 for r in records.values() if r.gateway_used)
    if used:
        out["notes"].append(
            f"{used} run(s) sent calls to the third-party gateway provider (agentrouter); "
            "its answers are mixed into those runs' results."
        )
    out["not_measured"] = not_measured
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aggregate", description="Write results/benchmark.json.")
    p.add_argument("--topics-file", type=Path, default=TOPICS_PATH)
    p.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--judgements-dir", type=Path, default=JUDGEMENTS_DIR)
    p.add_argument("--runs-per-topic", type=int, default=2)
    p.add_argument("--out", type=Path, default=RESULTS_PATH)
    args = p.parse_args(argv)
    try:
        topics = load_topics(args.topics_file, expected=None)
    except (TopicsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = aggregate(
        topics,
        load_records(args.records_dir),
        load_judgements(args.judgements_dir),
        args.runs_per_topic,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for system, s in result["systems"].items():
        r = s["runs"]
        print(
            f"{system}: attempted {r['attempted']}, succeeded {r['succeeded']}, "
            f"failed {r['failed']}, not_run {r['not_run']}"
        )
    print(f"wrote {args.out}; not measured: {len(result['not_measured'])} item(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
