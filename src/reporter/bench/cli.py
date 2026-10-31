"""`reporter bench <action> ...`: a thin dispatcher over the benchmark scripts."""

from __future__ import annotations

import sys

ACTIONS = ("run", "aggregate", "make-pairs", "judge-pairs")


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ACTIONS:
        print(f"usage: reporter bench {{{'|'.join(ACTIONS)}}} [options]", file=sys.stderr)
        return 2
    action, rest = argv[0], argv[1:]
    if action == "run":
        from reporter.bench.runner import main as run_main

        return run_main(rest)
    if action == "aggregate":
        from reporter.bench.aggregate import main as aggregate_main

        return aggregate_main(rest)
    if action == "make-pairs":
        from reporter.bench.pairs import make_pairs_main

        return make_pairs_main(rest)
    from reporter.bench.pairs import judge_pairs_main

    return judge_pairs_main(rest)
