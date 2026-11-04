# Benchmark run records

One JSON record per (topic, system, run index), written by `reporter bench run`.

## Batch log

| Batch | Date (UTC) | Runs | Outcome |
|---|---|---|---|
| 2 | 2026-11-04 12:15 | slm-on-device baseline r1, slm-on-device multi r1, carbon-capture multi r1 | baseline success (24,194 tokens, 431.7 s, stopped on its time budget); both multi runs failed with no report, their trace shows Gemini's per-day request limit (5,715 and 13,699 tokens) |

Batch 2 started after Gemini's daily reset (ping: Gemini and Groq ok), so unlike batch 1 (see `results/benchmark_attempts/`) these failures happened inside the batch: the runs themselves used up the free daily requests. The records are kept as they are. Whether such quota failures count against a system is an open methodology question, recorded in the R3 section of the report.
