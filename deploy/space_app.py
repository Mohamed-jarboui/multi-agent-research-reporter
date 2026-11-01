"""Entry point of the Hugging Face Gradio Space (pushed to the Space as `app.py`).

A Gradio Space runs `python app.py` and serves whatever listens on port 7860. We do not use
Gradio itself: this starts our FastAPI service with uvicorn. The Space has a bundled `src/`
folder next to this file, so the package is imported from there (no install step).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the Space layout is app.py + src/; in the repository this file is deploy/space_app.py
for candidate in (HERE / "src", HERE.parent / "src"):
    if (candidate / "reporter").is_dir():
        sys.path.insert(0, str(candidate))
        break

import uvicorn  # noqa: E402

from reporter.service.app import app_from_env  # noqa: E402

if __name__ == "__main__":
    # one worker on purpose: the queue, the rate limits and the quota latch live in this process
    uvicorn.run(app_from_env(), host="0.0.0.0", port=7860, workers=1)
