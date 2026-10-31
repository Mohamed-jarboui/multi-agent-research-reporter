# Demo image for the research reporter service (Hugging Face Docker Space: port 7860).
FROM python:3.12-slim

# uv from its official image; no dev dependencies in the final environment
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

RUN useradd --create-home --uid 1000 user \
    && mkdir -p /app /data \
    && chown user:user /app /data
USER user
ENV HOME=/home/user \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    REPORTER_DATA_DIR=/data
WORKDIR /app

# dependencies first (cached while pyproject.toml and uv.lock do not change)
COPY --chown=user pyproject.toml uv.lock README.md ./
COPY --chown=user src ./src
RUN uv sync --locked --no-dev

EXPOSE 7860
# one worker on purpose: the queue, the rate limits and the quota latch live in this process
CMD [".venv/bin/uvicorn", "reporter.service.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

# the image has no curl: use Python
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD [".venv/bin/python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:7860/health', timeout=4)"]
