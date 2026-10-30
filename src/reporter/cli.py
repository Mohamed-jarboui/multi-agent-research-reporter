"""Command line: `reporter baseline "<topic>"`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from reporter.budget import Budget, MultiBudget


def load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables that are already set."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def build_parser() -> argparse.ArgumentParser:
    defaults = Budget()
    parser = argparse.ArgumentParser(prog="reporter", description="Research reporter.")
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline", help="single-agent research run")
    baseline.add_argument("topic", help="what to research")
    baseline.add_argument("--out", type=Path, help="also write the report to this file")
    baseline.add_argument("--runs-dir", type=Path, default=Path("runs"), help="trace folder")
    baseline.add_argument("--max-steps", type=int, default=defaults.max_steps)
    baseline.add_argument("--max-tokens", type=int, default=defaults.max_tokens)
    baseline.add_argument("--max-seconds", type=float, default=defaults.max_seconds)
    baseline.add_argument("--max-cost", type=float, default=defaults.max_cost_usd, help="USD")

    multi_defaults = MultiBudget()
    multi = commands.add_parser("multi", help="multi-agent research run (supervisor and workers)")
    multi.add_argument("topic", help="what to research")
    multi.add_argument("--out", type=Path, help="also write the report to this file")
    multi.add_argument("--runs-dir", type=Path, default=Path("runs"), help="trace folder")
    multi.add_argument("--max-steps", type=int, default=multi_defaults.run.max_steps)
    multi.add_argument("--max-tokens", type=int, default=multi_defaults.run.max_tokens)
    multi.add_argument("--max-seconds", type=float, default=multi_defaults.run.max_seconds)
    cost = multi_defaults.run.max_cost_usd
    multi.add_argument("--max-cost", type=float, default=cost, help="USD")
    multi.add_argument("--workers", type=int, default=multi_defaults.max_parallel_workers)
    multi.add_argument("--llm-slots", type=int, default=multi_defaults.max_parallel_llm_calls)
    multi.add_argument("--no-critic", action="store_true", help="skip the critic and the retries")
    multi.add_argument("--no-writer", action="store_true", help="print the research digest only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    # heavy imports after argument parsing so --help stays instant
    from reporter.agent import BaselineAgent, MultiAgent
    from reporter.llm import build_default_llm
    from reporter.tools import default_tools

    try:
        if args.command == "multi":
            budget = MultiBudget(
                run=Budget(
                    max_steps=args.max_steps,
                    max_tokens=args.max_tokens,
                    max_seconds=args.max_seconds,
                    max_cost_usd=args.max_cost,
                    stop_fraction=MultiBudget().run.stop_fraction,
                ),
                max_parallel_workers=args.workers,
                max_parallel_llm_calls=args.llm_slots,
            )
        else:
            budget = Budget(
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                max_seconds=args.max_seconds,
                max_cost_usd=args.max_cost,
            )
        llm = build_default_llm()
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.command == "multi":
        agent: Any = MultiAgent(
            llm,
            default_tools(),
            budget,
            runs_dir=args.runs_dir,
            critic=not args.no_critic,
            writer=not args.no_writer,
        )
    else:
        agent = BaselineAgent(llm, default_tools(), budget, runs_dir=args.runs_dir)
    try:
        result = agent.run(args.topic)
    except Exception as exc:
        print(f"error: run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"trace: {agent.last_trace_path}", file=sys.stderr)
        return 1

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result.report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print(result.report)
    totals = result.trace.totals
    extra = ""
    if args.command == "multi":
        total = len(result.merged.sections)
        extra = f" | sections {total - len(result.merged.incomplete)}/{total} complete"
    print(
        f"\nstopped: {result.trace.stop_reason}{extra} | steps {totals.react_steps} | "
        f"tokens {totals.total_tokens} | cost ${totals.cost_usd:.4f} | "
        f"{totals.latency_s:.1f}s | trace: {result.trace_path}",
        file=sys.stderr,
    )
    if args.command == "multi":
        print(multi_summary(result), file=sys.stderr)
    return 0


def multi_summary(result: Any) -> str:
    """One line: sections ok/weak/missing, retries, and the citation check of the final text."""
    merged, review, trace = result.merged, result.review, result.trace
    ok = weak = missing = 0
    for section in merged.sections:
        verdict = review.sections[section.brief.id].verdict if review else "unreviewed"
        if not section.findings:
            missing += 1
        elif verdict in ("weak", "fail"):
            weak += 1
        else:
            ok += 1
    kept = sum(r["kept"] == "retry" for r in trace.retries)
    checks = trace.checks
    problems = [
        f"{name} {checks[name]}"
        for name in ("invalid_citations", "uncited_sources", "unfetched_urls")
        if checks.get(name)
    ]
    verdict = "ok" if checks and not problems else ", ".join(problems) or "not checked"
    return (
        f"sections ok/weak/missing: {ok}/{weak}/{missing} | retries: {len(trace.retries)} "
        f"(kept {kept}) | critic: {trace.critic.get('status', 'off')} | "
        f"writer: {trace.writer.get('status', '?')} | citations: {verdict}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
