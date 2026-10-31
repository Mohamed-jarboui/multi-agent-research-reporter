"""Thin entry point: `uv run python evals/make_pairs.py [options]`
(same as `reporter bench make-pairs`)."""

import sys

from reporter.bench.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["make-pairs", *sys.argv[1:]]))
