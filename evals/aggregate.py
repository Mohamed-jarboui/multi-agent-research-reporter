"""Thin entry point: `uv run python evals/aggregate.py [options]`
(same as `reporter bench aggregate`)."""

import sys

from reporter.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["aggregate", *sys.argv[1:]]))
