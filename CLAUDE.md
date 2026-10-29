# Multi-Agent Research Reporter

Give it a topic: a supervisor plans the report, workers research sections in parallel, a critic checks sources and gaps, and a writer assembles a cited report. The headline result is a measured comparison against a single agent doing the same job.

The stages and their prompts are in `docs/plan.md`. Progress is tracked in `docs/progress.json`.

## Stage runner (how every session works)

When the owner says "next stage" (or "run R2", etc.):

1. Read `docs/progress.json` and take the first stage whose status is not `merged` (or the stage named).
2. `git checkout main && git pull`, then create the branch `stage/rN-<short-name>`.
3. Run the stage's steps from `docs/plan.md` **in order**, each with the subagent the plan names (`designer`, `builder`, `evaluator`, `reporter`). Run each subagent in the **foreground** and wait for its result before starting the next. Give it the step's prompt word for word, plus the output of the previous step.
4. After the build steps: run `make lint` and `make test`. If they fail, send the failure back to `builder` (at most 2 attempts), then stop and report.
5. Always end the stage with the `reporter` subagent: it updates `docs/report.html` and `results/stage_rN.json`.
6. Commit (conventional style), push, open a PR into `main` with `gh pr create`. Set the stage to `pr_open` in `docs/progress.json`, with the PR URL.
7. **Stop.** Show the owner: the PR link, 5 bullets on what was built and how, the run cost if known, and 3 short questions that check their understanding. Wait. Do not start the next stage until the owner says it is merged; then set it to `merged`.

The owner chose: no `TODO(me)` parts in this project; build everything. One exception: in R3 the owner rates 6 report pairs to calibrate the judge (see the plan).

## Rules

### Commits and PRs
- No AI attribution anywhere: no co-author lines, no "generated with" text, no tool names, in commits or PR descriptions.
- Conventional commits (`feat:`, `fix:`, `chore:`, `test:`, `ci:`, `docs:`). Never commit directly to `main`.

### Numbers and claims
- Never invent metrics. Every number in docs, the README or the report must come from a script run, saved in `results/`, stated with its sample size. If a number is missing, write "not measured yet".

### Secrets
- Secrets only in `.env` (git-ignored) or GitHub Secrets: `GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`. Never read, print or commit `.env*` files; `.env.example` holds names only.

### Dependencies
- Keep dependencies light (the owner's internet is slow). Prefer the standard library; justify each new package in the PR description. `matplotlib` is allowed in a separate `report` dependency group.

### Budgets
- Every agent run has a max steps, token and time budget and a cost ceiling. Tests never call the network: use fake LLMs and fake tools.

## Commands

```bash
make install   # uv sync
make lint      # ruff check + ruff format --check
make test      # pytest
make figures   # regenerate docs/figs/*.svg from results/
```
