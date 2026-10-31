# Multi-Agent Research Reporter

Give it a topic: a supervisor plans the report, workers research the sections in parallel, a critic checks sources and gaps, and a writer assembles a cited report. The headline result will be a measured comparison against a single agent doing the same job: quality, cost and latency on 12 topics.

**Status:** in progress. Build plan in [docs/plan.md](docs/plan.md); progress report in [docs/report.html](docs/report.html).

## Benchmark

Single agent versus multi-agent system on the 12 topics in `evals/topics.jsonl` (2 runs per topic and system). **Results: not measured yet.** No number is reported here until the benchmark has been run; the figures will come from `results/benchmark.json`, each with its sample size.

```bash
uv run reporter ping                                # is any provider's quota available?
uv run python evals/run_benchmark.py --dry-run      # print the plan, call nothing
uv run python evals/run_benchmark.py --max-runs 4   # run a few; finished runs are skipped next time
uv run python evals/aggregate.py                    # write results/benchmark.json
uv run python evals/make_pairs.py                   # 6 blind pairs for you: evals/pairs_to_rate.md
uv run python evals/judge_pairs.py                  # judge those pairs, compare with evals/pairwise_human.jsonl
```

The same commands exist as `uv run reporter bench run|aggregate|make-pairs|judge-pairs`. Options of `run`: `--topics`, `--systems`, `--runs-per-topic`, `--max-runs`, `--swap-check`, `--retry-failed`, `--no-score`.

- Resumable: one record per (topic, system, run) in `results/benchmark_runs/`, raw traces in `runs/benchmark/` (git-ignored). When every provider is out of its per-day quota the script stops with exit code 3 and records the remaining runs as `not_run`; run it again after the reset.
- A run without a written report is a failed run. It stays in the failure rate and in the cost per successful report.
- The manual workflow `.github/workflows/benchmark.yml` (Actions, "Run workflow") runs the same script with the repository secrets and commits the results to the `results` branch.
- Design: [docs/design/r3.md](docs/design/r3.md).
