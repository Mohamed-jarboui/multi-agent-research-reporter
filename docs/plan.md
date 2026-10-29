# Build plan: stages R0-R4

The main session runs these steps in order, each with the named subagent (see CLAUDE.md, "Stage runner").
Subagent models: `designer` = Opus 5.5 · high, `builder` = Sonnet 5.5 · medium, `evaluator` = Sonnet 5.5 · high, `reporter` = Sonnet 5.5 · medium.
A **PAUSE** step means: stop, tell the owner exactly what to do, and wait for them to say "done".

---

## R0 · Setup and single-agent baseline

**Goal:** a ReAct research agent that searches the web, reads pages and writes a cited report. This is the baseline the multi-agent version must beat.

**Step 1 · builder**
1. uv project, package `reporter` (src layout), pytest, ruff, Makefile (`install`, `lint`, `test`, `figures`), GitHub Actions CI (lint + test on push and PR).
2. Copy the provider layer from eu-ai-act-copilot (`src/copilot/llm` into `src/reporter/llm`, with a comment crediting the source repo). Look for it at `../eu-ai-act-copilot`; if missing, `gh repo clone Mohamed-jarboui/eu-ai-act-copilot` into a temp folder. Gemini + Groq, fallback, retries, usage records.
3. Tools: `web_search` (Tavily free tier, key `TAVILY_API_KEY`; fall back to DuckDuckGo), `fetch_page` (readable text, bounded length). Read-only, Pydantic inputs, informative errors.
4. Baseline single agent (ReAct with LangGraph): topic -> research -> cited Markdown report with a sources list. Budgets: max steps, tokens, time, cost. Stop condition: stop when every planned question has at least 2 sources or the budget is 80% used.
5. Every run saves a trace to `runs/<run_id>.json`: steps, tool calls, tokens, cost, latency. `runs/` is git-ignored.
6. CLI: `uv run reporter baseline "<topic>"`.
7. `scripts/make_figures.py` (matplotlib, `report` dependency group) and `docs/progress.json` with R0-R4 as `todo`.
8. `.env.example` with the three key names. Tests with a fake LLM and fake tools.

**Step 2 · evaluator** · Run the baseline on 2 test topics ("state of small language models for on-device use", "carbon capture: current costs and limits"). Save the reports to `examples/baseline/`, summary to `results/baseline_smoke.json`.

**Step 3 · reporter** · Create `docs/report.html` (first version) and the R0 section.

---

## R1 · Supervisor and parallel workers

**Goal:** a supervisor splits the topic into sections and dispatches workers in parallel, each with a self-contained brief and a structured output.

**Step 1 · designer** · Design R1: graph, state, worker brief format (goal, scope, what not to cover, expected output), worker output schema (findings, sources, confidence, open questions), failure handling, shared budget.

**Step 2 · builder**
1. Supervisor node: plans 3-6 sections from the topic, each with a self-contained brief.
2. Workers run in parallel (LangGraph `Send` fan-out), each a short ReAct loop with the R0 tools, returning the structured result.
3. Merge step: deduplicate sources across sections, keep per-section findings, collect open questions.
4. Failed worker: retry once, then mark the section incomplete.
5. Per-run budget shared across workers (cost ceiling), recorded in the trace with per-worker timing.
6. CLI: `uv run reporter multi "<topic>"`. Tests: plan parsing, parallel execution with fake workers, failure handling, budget.

**Step 3 · evaluator** · Run the multi-agent system on the same 2 topics; show from the trace that workers overlapped in time. Save `results/r1_smoke.json`.

**Step 4 · reporter** · R1 section with a worker timeline figure (Gantt-style, from the trace).

---

## R2 · Critic and writer

**Goal:** a critic that checks every claim has a source, finds gaps and contradictions and sends weak sections back once; a writer that produces the final cited report.

**Step 1 · builder**
1. Deterministic checks first: every citation number exists in the sources list, every cited URL was actually fetched in this run.
2. Critic node with a numbered rubric: every claim sourced, sources support the claim, no contradictions between sections, coverage of the plan. JSON output: per-section verdict, issues, evidence.
3. Weak sections go back to their worker with the issues, at most 1 retry; keep the best attempt.
4. Writer node: final Markdown report with numbered citations and a sources list.
5. Tests for the deterministic checks and the retry cap.

**Step 2 · evaluator** · Run on the 2 topics; record how many sections the critic sent back and whether the retry improved them (critic verdict before/after). Save one full report to `examples/multi/` and `results/r2_smoke.json`.

**Step 3 · reporter** · R2 section with the full graph diagram and a before/after critic figure.

---

## R3 · Benchmark: multi-agent vs single agent

**Goal:** on 12 topics, compare report quality, cost, latency and tokens between the single agent and the multi-agent system.

**Step 1 · builder**
1. Draft `evals/topics.jsonl`: 12 varied topics (tech, science, policy, business), each with 3-5 must-cover points.
2. `evals/run_benchmark.py`: both systems on all topics, 2 runs each, traces saved.
3. Scoring: coverage of must-cover points (LLM check per point, with the quote it matched), citation validity rate (deterministic checks), pairwise judge (random order, length-normalized prompt).
4. Cost, tokens, p50/p95 latency, cost per successful report.
5. A manual GitHub Actions workflow (`workflow_dispatch`) that runs the benchmark with repo secrets and commits results to a `results` branch.

**Step 2 · PAUSE** · The owner reviews and edits `evals/topics.jsonl`.

**Step 3 · evaluator** · Run the benchmark locally (or tell the owner to trigger the workflow if local runs fail). Then write `evals/pairs_to_rate.md`: 6 report pairs (A/B, order randomized, systems hidden).

**Step 4 · PAUSE** · The owner reads the 6 pairs and writes `evals/pairwise_human.jsonl` (`{"pair": 1, "winner": "A"|"B"|"tie", "why": "..."}`). If the owner skips this, the report must say the judge is not calibrated.

**Step 5 · evaluator** · Run the judge on all pairs, report agreement with the owner's 6 ratings, write `results/benchmark.json` and a short analysis: when multi-agent wins, when it is not worth the cost.

**Step 6 · reporter** · R3 section with charts: quality (coverage, citation validity, judge win rate), cost per report, latency p50/p95, tokens. Benchmark table in the README too.

---

## R4 · Deploy with jobs, streaming and traces

**Goal:** long runs handled properly: jobs with run IDs, streaming progress, resume from checkpoint, a live demo.

**Step 1 · designer** · Design the job model: run IDs, states, idempotency key, checkpoint storage, resume rules, streaming events.

**Step 2 · builder**
1. FastAPI: `POST /reports` starts a job and returns a run ID; `GET /reports/{id}` for status and result; `GET /reports/{id}/events` (SSE) for live progress (plan, worker done, critic verdict, written).
2. Checkpointed state (SQLite) so a crashed job resumes from the last finished node; idempotency key on job creation.
3. Dockerfile; CI/CD deploying a demo to a Hugging Face Docker Space (secret `HF_TOKEN`) with strict per-run budgets and a daily cap.
4. OpenTelemetry spans (Langfuse-compatible attributes: tokens, cost, model); export one full trace to `docs/traces/example.json`.
5. README: pitch, demo link, architecture, patterns used and why, benchmark results, cost per report, one real failure story from the PRs, how to run.

**Step 3 · PAUSE** · The owner adds `HF_TOKEN` to GitHub Secrets and creates the Space if the deploy step needs it.

**Step 4 · evaluator** · Test the deployed demo: one job end to end, one forced crash + resume. Save `results/r4_deploy.json`.

**Step 5 · reporter** · R4 section with the job state diagram and a span timeline figure from `docs/traces/example.json`. Final pass on the whole report.
