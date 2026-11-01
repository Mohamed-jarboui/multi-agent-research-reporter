# Benchmark attempts set aside

Runs here failed only because the free-tier quota was already used up, so they say nothing about either system. They are kept as a record and their slots are run again in a later batch (the runner treats a missing record as pending). Both systems' runs from the same batch are set aside together, so neither system gets an extra chance.

| Date (UTC) | Run | Outcome | Cause in the record |
|---|---|---|---|
| 2026-11-01 23:3x | slm-on-device baseline r1 | failed, 13,548 tokens, 71.8 s | Gemini per-day request limit; Groq per-minute token limit |
| 2026-11-01 23:3x | slm-on-device multi r1 | failed (no report), 18,873 tokens, 276.3 s | Gemini per-day request limit |

Earlier the same day the R4 local test (2 jobs) and several pings had used Gemini's daily requests. Batches now start after Gemini's daily reset.
