"""Turn one run's OpenTelemetry spans into a single JSON file (default docs/traces/example.json).

    uv run python scripts/export_trace.py <run_id> --synthetic|--real
        [--runs-dir runs] [--out docs/traces/example.json]

The input is `<runs-dir>/<run_id>.spans.jsonl`, written by the service. The output is
`{label, synthetic, run_id, exported_at, spans}`. Say `--synthetic` for a run made with fake models
and tools: the label then states that its numbers are not measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reporter.service.telemetry import export_trace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--out", type=Path, default=Path("docs/traces/example.json"))
    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--synthetic", action="store_true", help="fake LLM and tools")
    kind.add_argument("--real", action="store_true", help="a real end-to-end run")
    parser.add_argument("--label", help="override the label")
    args = parser.parse_args(argv)
    try:
        doc = export_trace(
            args.runs_dir, args.run_id, args.out, synthetic=args.synthetic, label=args.label
        )
    except FileNotFoundError:
        print(f"error: no spans file for {args.run_id} in {args.runs_dir}")
        return 2
    print(f"wrote {args.out}: {doc['n_spans']} spans, synthetic={doc['synthetic']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
