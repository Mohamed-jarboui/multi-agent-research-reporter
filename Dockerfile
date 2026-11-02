# Image for the research reporter service: the FastAPI API plus the static demo page at `/`.
# The public demo runs this image on Render's free web service tier (render.yaml,
# docs/deploy_render.md). Render sets PORT (default 10000); locally the port is 7860.
# No gradio in here on purpose: a free Render instance has 512 MB of RAM.
# Storage is ephemeral by default (/health says so). To keep jobs across restarts locally, mount a
# volume and set REPORTER_DATA_DIR, e.g. `-v reporter-data:/data -e REPORTER_DATA_DIR=/data`
# (the volume must be writable by uid 1000).
FROM python:3.12-slim

# uv from its official image; no dev dependencies in the final environment
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app \
    && chown user:user /app
USER user
ENV HOME=/home/user \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first (cached while pyproject.toml and uv.lock do not change)
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --locked --no-dev

EXPOSE 7860
# one worker on purpose: the queue, the rate limits and the quota latch live in this process.
# Shell form with exec so uvicorn is PID 1 and gets the stop signal; binds 0.0.0.0 on $PORT.
CMD exec .venv/bin/uvicorn reporter.service.app:app --host 0.0.0.0 --port "${PORT:-7860}" --workers 1

# the image has no curl: use Python (Render ignores this and uses healthCheckPath)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD .venv/bin/python -c "import os, urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '7860'), timeout=4)"
