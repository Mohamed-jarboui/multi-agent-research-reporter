---
title: Multi-Agent Research Reporter
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Multi-Agent Research Reporter (demo)

Give a topic: a supervisor plans the report, workers research the sections in parallel, a critic
checks sources and gaps, and a writer assembles a cited Markdown report.

This Space runs the FastAPI service from the GitHub repository (the source of truth). The free
model quota allows only a few runs per day, so the demo has strict per-run budgets and a daily cap
(`GET /quota`).

- `POST /reports` starts a job and returns a run id; `GET /reports/{id}` shows its state;
  `GET /reports/{id}/events` streams progress (server-sent events).
- `GET /health` for liveness.

The three provider keys are Space secrets (`GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`).
