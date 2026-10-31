"""Blind report pairs for the owner (`make_pairs`) and the judge run on them (`judge_pairs`)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from reporter.agent.llm_call import LLM
from reporter.bench.aggregate import JUDGE_PAIRS_PATH, ratio
from reporter.bench.records import RECORDS_DIR, SYSTEMS, RunRecord, load_records
from reporter.bench.scoring import judge_ordered
from reporter.bench.topics import TOPICS_PATH, Topic, TopicsError, load_topics
from reporter.llm.base import ProviderError

PAIRS_MD = Path("evals/pairs_to_rate.md")
PAIRS_KEY = Path("evals/pairs_key.json")
HUMAN_RATINGS = Path("evals/pairwise_human.jsonl")
DEFAULT_PAIRS = 6


class PairSpec(BaseModel):
    pair: int
    topic_id: str
    run_index: int
    a_system: str
    b_system: str


def valid_pairs(records: dict[tuple[str, str, int], RunRecord]) -> list[tuple[str, int]]:
    """(topic id, run index) where both systems wrote a report, sorted."""
    found = {
        (t, k) for (t, s, k), r in records.items() if s == SYSTEMS[0] and r.status == "success"
    }
    return sorted(
        (t, k)
        for (t, k) in found
        if (r := records.get((t, SYSTEMS[1], k))) and r.status == "success"
    )


def choose_pairs(
    records: dict[tuple[str, str, int], RunRecord], n: int, seed: int
) -> list[PairSpec]:
    """Up to `n` pairs, distinct topics first (random, seeded), then second runs of those topics.
    The A/B order of each pair is seeded too."""
    rng = random.Random(seed)
    by_topic: dict[str, list[int]] = {}
    for topic_id, k in valid_pairs(records):
        by_topic.setdefault(topic_id, []).append(k)
    topic_ids = sorted(by_topic)
    rng.shuffle(topic_ids)
    chosen = [(t, rng.choice(by_topic[t])) for t in topic_ids]
    spare = [(t, k) for t in topic_ids for k in by_topic[t] if (t, k) not in chosen]
    rng.shuffle(spare)
    specs = []
    for number, (topic_id, k) in enumerate((chosen + spare)[:n], 1):
        baseline_first = random.Random(f"{seed}:pair:{number}").random() < 0.5
        a, b = SYSTEMS if baseline_first else SYSTEMS[::-1]
        specs.append(PairSpec(pair=number, topic_id=topic_id, run_index=k, a_system=a, b_system=b))
    return specs


INSTRUCTIONS = """# Report pairs to rate

For each pair below, read both reports on the same topic and decide which one you prefer: more
useful, accurate (check a few claims against their sources) and clear. Do not reward length for
its own sake. The order is random and the system names are hidden.

Write one line per pair in `evals/pairwise_human.jsonl`:

```
{{"pair": 1, "winner": "A", "why": "short reason"}}
```

`winner` is `"A"`, `"B"` or `"tie"`. Pairs available: {n}.{note}
"""


def render_markdown(
    specs: list[PairSpec], records: dict[tuple[str, str, int], RunRecord], topics: dict[str, Topic]
) -> str:
    wanted = DEFAULT_PAIRS
    note = (
        ""
        if len(specs) >= wanted
        else f" Only {len(specs)} valid pairs exist (target {wanted}): run more of the benchmark."
    )
    parts = [INSTRUCTIONS.format(n=len(specs), note=note)]
    for spec in specs:
        topic = topics[spec.topic_id].topic if spec.topic_id in topics else spec.topic_id
        parts.append(f"\n---\n\n# Pair {spec.pair}: {topic}\n")
        for label, system in (("A", spec.a_system), ("B", spec.b_system)):
            report = records[(spec.topic_id, system, spec.run_index)].report.strip()
            parts.append(f"\n## Report {label}\n\n{_demote(report)}\n")
    return "".join(parts)


def _demote(report: str) -> str:
    """Push the report's headings two levels down so they sit under 'Report A/B'."""
    lines = []
    for line in report.splitlines():
        lines.append("###" + line if line.startswith("#") else line)
    return "\n".join(lines)


def make_pairs(
    records: dict[tuple[str, str, int], RunRecord],
    topics: dict[str, Topic],
    seed: int,
    n: int = DEFAULT_PAIRS,
    md_path: Path = PAIRS_MD,
    key_path: Path = PAIRS_KEY,
) -> list[PairSpec]:
    specs = choose_pairs(records, n, seed)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(specs, records, topics), encoding="utf-8")
    key = {
        "seed": seed,
        "valid_pairs_available": len(valid_pairs(records)),
        "pairs": [s.model_dump() for s in specs],
    }
    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    return specs


def make_pairs_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="make_pairs", description="Write the blind pairs to rate.")
    p.add_argument("--topics-file", type=Path, default=TOPICS_PATH)
    p.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-n", type=int, default=DEFAULT_PAIRS)
    p.add_argument("--md", type=Path, default=PAIRS_MD)
    p.add_argument("--key", type=Path, default=PAIRS_KEY)
    args = p.parse_args(argv)
    try:
        topics = {t.id: t for t in load_topics(args.topics_file, expected=None)}
    except (TopicsError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    specs = make_pairs(load_records(args.records_dir), topics, args.seed, args.n, args.md, args.key)
    print(f"wrote {len(specs)} pairs to {args.md} (key: {args.key})")
    if len(specs) < args.n:
        print(f"only {len(specs)} valid pairs exist; wanted {args.n}", file=sys.stderr)
    return 0 if specs else 1


# ------------------------------------------------------------------------------- the judge


def load_human_ratings(path: Path) -> dict[int, str]:
    """pair number -> 'A' | 'B' | 'tie'. A bad line is an error, not skipped."""
    ratings: dict[int, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            winner = str(item["winner"]).strip()
            winner = "tie" if winner.lower() == "tie" else winner.upper()
            if winner not in ("A", "B", "tie"):
                raise ValueError(winner)
            ratings[int(item["pair"])] = winner
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{path} line {number}: not a valid rating ({exc})") from exc
    return ratings


def judge_pairs(
    llm: LLM,
    specs: list[PairSpec],
    records: dict[tuple[str, str, int], RunRecord],
    topics: dict[str, Topic],
    human: dict[int, str] | None,
    *,
    swap_check: bool = False,
) -> dict[str, Any]:
    """Judge each pair in the owner's A/B order and compare with the owner's ratings, if any."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        ra = records[(spec.topic_id, spec.a_system, spec.run_index)]
        rb = records[(spec.topic_id, spec.b_system, spec.run_index)]
        row: dict[str, Any] = {"pair": spec.pair, "topic_id": spec.topic_id}
        try:
            v = judge_ordered(
                llm, topics[spec.topic_id].topic, ra.report, rb.report, swap_check=swap_check
            )
        except Exception as exc:
            row["error"] = exc.short() if isinstance(exc, ProviderError) else type(exc).__name__
            rows.append(row)
            continue
        row |= {
            "judge_winner": v.winner,
            "judge_winner_system": {"A": spec.a_system, "B": spec.b_system, "tie": "tie"}[v.winner],
            "reason": v.reason,
            "judge_model": v.model,
            "same_model_as_writer": v.model in {*ra.models, *rb.models},
            "words": {"A": v.a_words, "B": v.b_words},
            "consistent_under_swap": v.consistent,
        }
        if human and spec.pair in human:
            row["human_winner"] = human[spec.pair]
            row["agree"] = human[spec.pair] == v.winner
        rows.append(row)
    rated = [r for r in rows if "agree" in r]
    out: dict[str, Any] = {
        "pairs": rows,
        "n_pairs": len(specs),
        "n_judged": sum("error" not in r for r in rows),
    }
    if rated:
        decisive = [r for r in rated if r["human_winner"] != "tie"]
        out |= {
            "calibrated": True,
            "n_rated": len(rated),
            "agreement": ratio(sum(r["agree"] for r in rated), len(rated)),
            "agreement_on_decisive_human_ratings": ratio(
                sum(r["agree"] for r in decisive), len(decisive)
            ),
            "message": f"judge agrees with the owner on {sum(r['agree'] for r in rated)} "
            f"of {len(rated)} rated pairs",
        }
    else:
        out |= {
            "calibrated": False,
            "n_rated": 0,
            "message": "the judge is not calibrated: no ratings by the owner "
            f"({HUMAN_RATINGS} is missing or has no usable line)",
        }
    return out


def judge_pairs_main(argv: list[str] | None = None) -> int:
    from reporter.bench.runner import build_scoring_llm
    from reporter.cli import load_env_file

    p = argparse.ArgumentParser(prog="judge_pairs", description="Judge the pairs the owner rated.")
    p.add_argument("--topics-file", type=Path, default=TOPICS_PATH)
    p.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    p.add_argument("--key", type=Path, default=PAIRS_KEY)
    p.add_argument("--human", type=Path, default=HUMAN_RATINGS)
    p.add_argument("--out", type=Path, default=JUDGE_PAIRS_PATH)
    p.add_argument("--judge-provider", choices=["auto", "gemini", "groq"], default="auto")
    p.add_argument("--swap-check", action="store_true")
    args = p.parse_args(argv)
    load_env_file()
    try:
        topics = {t.id: t for t in load_topics(args.topics_file, expected=None)}
        key = json.loads(args.key.read_text(encoding="utf-8"))
        specs = [PairSpec.model_validate(s) for s in key["pairs"]]
        human = load_human_ratings(args.human) if args.human.is_file() else None
        llm = build_scoring_llm(args.judge_provider)
    except (TopicsError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    result = judge_pairs(
        llm, specs, load_records(args.records_dir), topics, human, swap_check=args.swap_check
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(result["message"])
    return 0
