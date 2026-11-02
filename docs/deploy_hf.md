# Deploying the demo to a Hugging Face Space

> **Status: not used for the live demo.** ZeroGPU rejected the app at startup with "No @spaces.GPU function detected during startup". We do not add a dummy GPU function, and CPU basic needs a paid plan, so the live demo is on Render ([deploy_render.md](deploy_render.md)). The Gradio Space files (`deploy/space_app.py`, `deploy/space_README.md`) and the manual-only workflow `.github/workflows/deploy.yml` remain for anyone with a paid CPU basic plan; the rest of this page describes that path.

The workflow `.github/workflows/deploy.yml` pushes the service to an existing Hugging Face Gradio Space (by hand only: it has no push trigger any more), then waits for `GET /health`. It does not create the Space. Without `HF_TOKEN` it skips with a notice.

All of this is free: **ZeroGPU** hardware, no persistent storage. A free account can host Gradio Spaces on ZeroGPU only: CPU basic needs a paid plan, and Docker Spaces are paid too. ZeroGPU Spaces are exclusively compatible with the Gradio SDK, so `app.py` is a genuine Gradio app (`gr.Blocks`: topic box, Start button, live status and events, the report with its coverage summary and the quota left) and our FastAPI service runs in the same process on port 7860. `/reports`, `/reports/{id}`, `/reports/{id}/events`, `/reports/{id}/trace`, `/health` and `/quota` keep working and win over the UI; the Gradio page is at `/`. The UI calls the service in-process (an httpx client wired to the ASGI app, so the same validation, rate limits and daily cap apply) and reads progress events from the job store.

**No GPU code on purpose.** The app does not use a GPU, so it does not import `spaces` and has no `@spaces.GPU` function. We do not add a dummy GPU function to get past a platform check. If Hugging Face starts requiring one, the demo moves to another host instead.

Space settings come from `deploy/space_README.md`: `sdk: gradio`, `sdk_version: 6.31.0` (gradio is pre-installed at that version; it resolves with our locked fastapi 0.162.0 and starlette 1.10.0), `python_version: "3.12.12"` (ZeroGPU supports 3.12.12 and 3.10.13; our code needs 3.11 or newer). `requirements.txt` is generated from `uv.lock` without the web stack (fastapi, starlette, uvicorn, pydantic, httpx, huggingface-hub, pillow, numpy and a few more): gradio brings compatible versions, and pinning ours next to the pre-installed gradio could break it.

## 1. Create the Space (on huggingface.co)

1. Sign in, then open https://huggingface.co/new-space.
2. **Space name:** `research-reporter` (any name works; you will need it in step 4). The owner's Space is `Jarbouii/research-reporter`.
3. **SDK:** **Gradio**, template **Blank** (not Docker, which is paid).
4. **Hardware:** **ZeroGPU** (the free option on a free account; CPU basic needs a paid plan).
5. **Visibility:** **Public**. The workflow's health check calls the Space URL without a token, so a private Space makes the deploy job fail at the last step.
6. Click **Create Space**. Leave it empty: the workflow pushes the files.

## 2. Add the provider keys as Space secrets

In the Space: **Settings → Variables and secrets → New secret**, three times:

| Name | Value |
|---|---|
| `GEMINI_API_KEY` | your Gemini key |
| `GROQ_API_KEY` | your Groq key |
| `TAVILY_API_KEY` | your Tavily key |

Use **secrets**, not variables: secrets are hidden from visitors and from the Space's files.

Optional **variables** (the defaults are strict on purpose): `DAILY_MAX_JOBS`, `DAILY_MAX_TOKENS`.

Storage: free Spaces have no `/data` disk, so the job store is temporary and is wiped when the Space restarts. `GET /health` reports `"storage": "ephemeral"` in that case. That is expected.

## 3. Create a Hugging Face token

1. https://huggingface.co/settings/tokens → **Create new token**.
2. Prefer **Fine-grained**, with **write** access to this one Space only (`Jarbouii/research-reporter`). A classic **Write** token also works but can write to everything in your account.
3. Copy the token once. Do not paste it into chat or into any file in the repository.

## 4. Add the token and the Space id to GitHub

In the GitHub repository: **Settings → Secrets and variables → Actions**.

1. **Secrets** tab → **New repository secret**: name `HF_TOKEN`, value the token from step 3.
2. **Variables** tab → **New repository variable**: name `HF_SPACE_ID`, value `Jarbouii/research-reporter` (your Hugging Face username, not your GitHub one, if they differ).

## 5. Deploy

Either push anything to `main`, or open **Actions → Deploy demo → Run workflow** on `main`. The workflow pushes `app.py`, `requirements.txt` (generated from `uv.lock` with `uv export`), `src/` and `README.md`. Hugging Face installs the requirements with pip, which takes several minutes; the workflow waits up to 20 minutes for the health check.

## 6. Check it

- `https://jarbouii-research-reporter.hf.space/health` → `"status": "ok"` (dots and underscores in the id become dashes, all lower case).
- `https://jarbouii-research-reporter.hf.space/quota` → remaining jobs and tokens for today.

Then the R4 deploy test runs against the Space: one job end to end. The crash and resume test stays local, because a free Space has no persistent disk to resume from.

## Optional: run the benchmark on GitHub instead of a laptop

`.github/workflows/benchmark.yml` (manual) needs the same three provider keys as **GitHub** repository secrets (`GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`). It saves its results to a `results` branch and resumes from it.
