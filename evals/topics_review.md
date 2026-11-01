# Topic review (R3, step 2)

The plan has the owner review `evals/topics.jsonl`. On 2026-11-02 the owner delegated this review, so it was done in the main session rather than by the owner.

## What was checked

- 12 topics across 4 domains: tech 2, science 3, policy 3, business 4.
- Each topic has 3-5 must-cover points, phrased as aspects to cover, not as factual claims.
- The two smoke topics (`slm-on-device`, `carbon-capture`) are kept, so the benchmark can be read next to the R0-R2 smoke runs. `carbon-capture` failed often in the smoke runs (PDF-heavy and blocked sources); it stays as a deliberately hard case.
- Points must be specific enough that a report which ignores them would not count as covering them.

## Changes

| Topic | Before | After | Why |
|---|---|---|---|
| quantum-error-correction | Recent experimental milestones | At least one recent experimental milestone and the group that reported it | Too vague: almost any report would match it |
| creator-economy | Emerging tools and business models | Business models creators use beyond ads and sponsorships | Too vague, and it overlapped with the first point |

No topic was added or removed.

## Run plan on the free tier

The owner chose the free tier. Observed capacity is roughly one or two multi-agent runs per day (R2 fourth evaluation and R4 local test, n=4 runs), so the benchmark runs in small daily batches:

1. First pass: `--runs-per-topic 1` (24 runs: 12 topics x 2 systems), `--no-score`, a few runs per day after a `reporter ping`.
2. Second pass: the second run per topic, if time allows. If it does not, the report states n=1 per topic per system.
3. Scoring (coverage, citations, judge) once the runs exist, then the 6 pairs for the owner to rate.

The number of days this takes is not measured yet.
