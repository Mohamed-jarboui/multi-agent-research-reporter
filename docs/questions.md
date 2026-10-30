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
