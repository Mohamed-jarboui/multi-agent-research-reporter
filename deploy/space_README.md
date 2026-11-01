---
title: Multi-Agent Research Reporter
emoji: 📝
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Multi-Agent Research Reporter (demo)

Give a topic: a supervisor plans the report, workers research the sections in parallel, a critic
checks sources and gaps, and a writer assembles a cited Markdown report.

This Space (Gradio SDK, free CPU basic hardware) runs the FastAPI service from the GitHub
repository (the source of truth): `app.py` starts it with uvicorn on port 7860. The free model
quota allows only a few runs per day, so the demo has strict per-run budgets and a daily cap
(`GET /quota`). There is no persistent storage: jobs are lost when the Space restarts.

- `POST /reports` starts a job and returns a run id; `GET /reports/{id}` shows its state;
  `GET /reports/{id}/events` streams progress (server-sent events).
- `GET /health` for liveness.

The three provider keys are Space secrets (`GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`).
