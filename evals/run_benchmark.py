"""Thin entry point: `uv run python evals/run_benchmark.py [options]`
(same as `reporter bench run`)."""

import sys

from reporter.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
