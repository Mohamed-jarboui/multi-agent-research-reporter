# Understanding questions

One section per stage. Answer them when the project is finished.

## R0 · Setup and single-agent baseline

1. Why does the 80% budget rule leave room before the write step, and what happens if a model call fails during research?
2. The citation checker flagged sources 3 and 4 as invalid. What does it compare, and why might grouped citations like `[5, 6]` get past it?
3. Carbon capture ran out of time using only 14k of 150k tokens. Which budget was binding, and what change would have saved the run?

## R1 · Supervisor and parallel workers

1. The graph fans out with `Send` and caps parallelism at 3 workers, yet effective parallelism was only 2.50 and 2.84. What limits the speed-up, and which setting would you change first?
2. Why are global source numbers assigned by section order and not by the order workers finish, and what would break in R2's citation checks if they were not?
3. A section can be marked complete and still have stopped on `max_steps` or `worker_time`. What does "complete" actually mean here, and why is that not the same as a good section?

## R2 · Critic and writer

1. The first evaluation showed the retry never ran, although tokens and time were barely used. Which budget was binding, why did the 70% research cut-off make it worse, and how does the new retry gate differ?
2. The retry did run in the second evaluation (3 sections) yet improved none. Why can a retry fail to help when the section's problem is "no page fetched", and what would you change so a retry has a better chance?
3. The citation checker now parses `[5]`, `[5, 6]`, `[5][6]` and `[5-7]`, but only `[n]` appeared in real reports. Why are the unit tests still worth having, and what would a deterministic check not catch that the critic is supposed to?

## R3 · Benchmark: multi-agent vs single agent

1. Why does the benchmark keep failed runs in every denominator, and how could dropping them make one system look better than it is?
2. Coverage counts a must-cover point only if the quote the model returns is found in the report text. What failure does that check prevent, and what can it still miss?
3. The judge and the writer may be the same model, and the judge is only calibrated if you rate the 6 pairs. What biases does each of those introduce, and how does the benchmark report them?

## R4 · Deploy with jobs, streaming and traces

1. A job crashes in the middle of the parallel worker fan-out and is resumed. Which parts of the state are restored from SQLite, which are rebuilt fresh, and why are finished workers not recomputed?
2. What does the `Idempotency-Key` header guarantee for the same key with the same body, and for the same key with a different body? Why is the key stored only as a hash?
3. The free tier allows only a couple of runs per day. Which mechanisms in the service keep the public demo from burning through the quota, and what does a visitor see when the quota is gone?
