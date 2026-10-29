"""Command line: `reporter baseline "<topic>"`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from reporter.budget import Budget


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    # heavy imports after argument parsing so --help stays instant
    from reporter.agent import BaselineAgent
    from reporter.llm import build_default_llm
    from reporter.tools import default_tools

    try:
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
    print(
        f"\nstopped: {result.trace.stop_reason} | steps {totals.react_steps} | "
        f"tokens {totals.total_tokens} | cost ${totals.cost_usd:.4f} | "
        f"{totals.latency_s:.1f}s | trace: {result.trace_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
