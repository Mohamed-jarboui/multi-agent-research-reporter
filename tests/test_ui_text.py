"""The plain-text pieces of the Gradio demo page."""

from reporter.service.ui_text import event_line, quota_line, report_markdown, status_line

QUOTA = {
    "jobs": {"left": 1, "limit": 2},
    "queue": {"waiting": 0},
    "resets_at": "2026-11-03T00:00:00Z",
    "latch_until": None,
}
RESULT = {
    "report_markdown": "# Batteries\n\nText [1].",
    "coverage": {"sections_ok": 2, "sections_weak": 1, "sections_missing": 0},
}


def test_quota_line_mentions_latch() -> None:
    assert "1 of 2 runs left" in quota_line(QUOTA)
    assert "used up until" in quota_line({**QUOTA, "latch_until": "2026-11-02T20:00:00Z"})


def test_status_line_shows_progress_and_reason() -> None:
    running = {"state": "running", "progress": {"sections_done": 1, "sections_total": 3}}
    assert status_line(running) == "running (1/3 sections done)"
    assert status_line({"state": "degraded", "message": "No report."}) == "degraded: No report."


def test_event_line_is_type_and_json() -> None:
    assert event_line({"type": "plan", "data": {"n": 2}}) == 'plan {"n": 2}'


def test_report_markdown_has_coverage_and_quota() -> None:
    text = report_markdown({"state": "succeeded", "result": RESULT}, QUOTA)
    assert "# Batteries" in text
    assert "2 sections ok, 1 weak, 0 missing" in text
    assert "1 of 2 runs left" in text


def test_degraded_run_leads_with_the_reason() -> None:
    job = {"state": "degraded", "message": "The run budget ran out.", "result": RESULT}
    assert report_markdown(job, None).startswith("> **Degraded run:** The run budget ran out.")
    assert report_markdown({"state": "failed"}, None) == ""
