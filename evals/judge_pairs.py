"""Thin entry point: `uv run python evals/judge_pairs.py [options]`
(same as `reporter bench judge-pairs`)."""

import sys

from reporter.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["judge-pairs", *sys.argv[1:]]))
