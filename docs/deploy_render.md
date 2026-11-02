# Deploying the demo on Render (free web service)

The live demo runs the repository's `Dockerfile` as a free **web service** on Render, described by `render.yaml` at the repository root. It serves our FastAPI API plus the static demo page at `/` (no Gradio, so it fits the 512 MB of RAM). Hugging Face's free tier could not host it (see [deploy_hf.md](deploy_hf.md)).

## What the free plan means

- 0.1 CPU and 512 MB RAM. Peak memory measured locally with fake providers: about 87 MB (see "Memory" in [design/r4.md](design/r4.md)); a real job was not measured.
- The service **spins down after 15 minutes without inbound traffic**. The next request waits about 1 minute (cold start). Open the URL once before a demo.
- **750 free instance hours per workspace per month** (one always-awake service uses about 720).
- The **filesystem is ephemeral**: jobs, traces and the daily-cap counters are lost on redeploy, restart and spin-down. `/health` reports `"storage": "ephemeral"`. The daily cap (`DAILY_MAX_JOBS`, `DAILY_MAX_TOKENS`) therefore resets with the instance; the provider quotas (Gemini, Groq, Tavily free tiers) stay the real ceiling.
- Render may **restart a free service at any time**. A running job is then lost (there is no disk to resume from), and an open SSE progress stream is cut. A client should reconnect to `GET /reports/{id}/events` with the `Last-Event-ID` header; if the job is gone (404), start a new one.

## Steps (once)

1. Sign up at https://render.com with your GitHub account.
2. Dashboard: **New > Blueprint**.
3. Connect the repository `multi-agent-research-reporter` (allow Render to access it).
4. Choose the branch **main** (the Blueprint file must be merged there first).
5. Review what Render lists: one web service `research-reporter`, plan **Free**, region Frankfurt. The region cannot be changed after creation.
6. Paste the three keys when prompted: `GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`. They are stored as secrets in Render, not in the repository.
7. Click **Deploy Blueprint**. The first build downloads the base image and installs the dependencies; wait until the service is **Live**. Render marks a deploy healthy when `GET /health` answers 2xx (a new deploy must pass within 15 minutes).
8. Open the URL Render shows for the service (of the form `https://research-reporter-XXXX.onrender.com`, assigned by Render) and check:
   - `<URL>/health` returns `{"status":"ok","storage":"ephemeral",...}`
   - `<URL>/quota` shows today's counters (2 jobs, 400 000 tokens)
   - `<URL>/` shows the demo page.
9. Put the URL in the README ("Demo" line).

After that, every commit on `main` whose GitHub checks pass is deployed automatically (`autoDeployTrigger: checksPass`).

## To verify in the Render dashboard

- The `envVars` form in `render.yaml` (`- key: NAME` with `sync: false`) is the common Blueprint syntax but was not copied verbatim from the docs: confirm that Render prompts for the three keys at step 6.
- The service's **Auto-Deploy** setting reads "After CI checks pass". If Render cannot see the checks, set `autoDeployTrigger: commit` in `render.yaml`.
- The instance type is Free and no disk is attached.

## Changing the caps

Edit `DAILY_MAX_JOBS` / `DAILY_MAX_TOKENS` in `render.yaml` (or in the service's Environment tab). `REPORTER_TEST_HOOKS` must stay unset.

## Run the same image locally

```bash
docker build -t research-reporter .
docker run --rm -e PORT=10000 -p 10000:10000 -e GEMINI_API_KEY -e GROQ_API_KEY -e TAVILY_API_KEY research-reporter
```

Without `PORT` the container listens on 7860.
