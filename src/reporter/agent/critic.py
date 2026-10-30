"""The critic: one compact LLM call that grades sections against the rubric, plus the retry rules.

The deterministic checks (`checks.py`) run first and need no model. The critic never raises: when
the model fails, `run_critic` returns a failure reason and the caller keeps the checks alone.
"""

from __future__ import annotations

import re
from typing import Any

from reporter.agent import prompts
from reporter.agent.llm_call import generate_charged
from reporter.agent.parsing import extract_json
from reporter.agent.schemas import (
    VERDICT_RANK,
    Contradiction,
    Issue,
    MergedResearch,
    Review,
    SectionOut,
    SectionReview,
    WorkerResult,
)
from reporter.agent.worker import WorkerContext
from reporter.budget import BudgetExhausted
from reporter.llm.base import ProviderError
from reporter.trace import StepRecord

CRITIC_MAX_TOKENS = 2_500
EXCERPT_CHARS = 900
RETRY_TEMPERATURE = 0.7
MAX_RETRY_SECTIONS = 3
MIN_RETRY_SECONDS = 30.0  # research time that must be left before the stop fraction
RETRYABLE = ("fail", "weak")


def excerpt(text: str, claims: list[str], size: int = EXCERPT_CHARS) -> str:
    """The `size`-char window of `text` that shares the most words with the claims."""
    text = " ".join(text.split())
    if len(text) <= size:
        return text
    words = {w for c in claims for w in re.findall(r"[a-z0-9]{4,}", c.lower())}
    step = max(size // 3, 1)
    best_start, best_score = 0, -1
    for start in range(0, len(text) - size + step, step):
        window = text[start : start + size].lower()
        score = sum(1 for w in words if w in window)
        if score > best_score:
            best_start, best_score = start, score
    return text[best_start : best_start + size]


def render_critic_sections(merged: MergedResearch, ids: list[str]) -> str:
    """Brief, findings and source excerpts of the given sections, in a compact text."""
    by_id = {s.id: s for s in merged.sources}
    blocks: list[str] = []
    for section in merged.sections:
        brief = section.brief
        if brief.id not in ids:
            continue
        lines = [
            f"## {brief.id}: {brief.title}",
            f"Goal: {brief.goal}",
            "Scope: " + "; ".join(brief.scope),
            "Findings:",
        ]
        claims_by_source: dict[int, list[str]] = {}
        for n, finding in enumerate(section.findings, 1):
            cites = "".join(f"[{i}]" for i in finding.source_ids)
            lines.append(f"F{n} {cites}: {finding.claim}")
            for i in finding.source_ids:
                claims_by_source.setdefault(i, []).append(finding.claim)
        lines.append("Sources:")
        for i, claims in sorted(claims_by_source.items()):
            source = by_id.get(i)
            page = merged.pages.get(source.url) if source else None
            text = excerpt(page.text, claims) if page else "(page text not available)"
            lines.append(f"[{i}] {source.title if source else '?'}: {text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def parse_review(
    text: str, ids: list[str]
) -> tuple[dict[str, SectionReview], list[Contradiction], list[str]]:
    """Validate a critic reply. A section the reply left out is `unreviewed`. Raises ValueError."""
    data = extract_json(text)
    raw = data.get("sections")
    if not isinstance(raw, list):
        raise ValueError("'sections' is missing or not a list")
    reviews: dict[str, SectionReview] = {}
    for item in raw:
        if not isinstance(item, dict) or str(item.get("id", "")) not in ids:
            continue
        sid = str(item["id"])
        verdict = str(item.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "weak", "fail"):
            continue
        issues = [i for i in map(_issue, _as_list(item.get("issues"))) if i is not None]
        if verdict != "pass" and not issues:
            issues = [Issue(issue="the critic found this section weak without giving detail")]
        reviews[sid] = SectionReview(section_id=sid, verdict=verdict, issues=issues)  # type: ignore[arg-type]
    if not reviews:
        raise ValueError("no section of the reply has a valid id and verdict")
    for sid in ids:
        reviews.setdefault(sid, SectionReview(section_id=sid, verdict="unreviewed"))
    contradictions: list[Contradiction] = []
    for item in _as_list(data.get("contradictions")):
        if not isinstance(item, dict) or not str(item.get("issue", "")).strip():
            continue
        named = [str(s) for s in _as_list(item.get("sections")) if str(s) in ids]
        if named:
            contradictions.append(
                Contradiction(
                    section_ids=named,
                    issue=str(item["issue"]).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                )
            )
    gaps = [str(g).strip() for g in _as_list(data.get("coverage_gaps")) if str(g).strip()]
    return reviews, contradictions, gaps


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _issue(item: Any) -> Issue | None:
    if not isinstance(item, dict) or not str(item.get("issue", "")).strip():
        return None
    rubric = item.get("rubric", 0)
    return Issue(
        rubric=rubric if isinstance(rubric, int) and 0 <= rubric <= 4 else 0,
        issue=str(item["issue"]).strip(),
        evidence=str(item.get("evidence", "")).strip(),
    )


def run_critic(
    ctx: WorkerContext, merged: MergedResearch, ids: list[str]
) -> tuple[Review | None, str]:
    """Grade the sections `ids` with one model call (one more try if the reply is unusable).

    Returns (review, "") or (None, reason). The review holds only the graded sections.
    """
    if not ids:
        return Review(critic_status="ok"), ""
    prompt = prompts.CRITIC_PROMPT.format(
        topic=ctx.topic, sections=render_critic_sections(merged, ids)
    )
    reason = ""
    for attempt in range(2):
        step = StepRecord(index=0, kind="critic")
        try:
            gen = generate_charged(
                ctx.llm,
                step,
                prompt,
                prompts.CRITIC_SYSTEM,
                CRITIC_MAX_TOKENS,
                ctx.meter,
                ctx.gate,
                clock=ctx.clock,
                temperature=0.0 if attempt == 0 else RETRY_TEMPERATURE,
                enforce_limit=True,
                max_wait_s=max(ctx.budget.run.max_seconds - ctx.meter.elapsed_s, 0.0),
                on_wait=lambda w: ctx.trace.add_rate_limit_wait("critic", ctx.meter.elapsed_s, w),
                cost_fn=ctx.cost_fn,
            )
        except (ProviderError, BudgetExhausted) as exc:
            return None, f"model call failed: {exc}"
        finally:
            ctx.trace.add_step(step)
        try:
            reviews, contradictions, gaps = parse_review(gen.text, ids)
        except ValueError as exc:
            reason = f"unusable reply ({str(exc).splitlines()[0][:100]})"
            step.note = f"{step.note} {reason}".strip()
            prompt += prompts.CRITIC_RETRY_HINT.format(reason=reason)
            continue
        review = Review(
            sections=reviews, contradictions=contradictions, coverage_gaps=gaps, critic_status="ok"
        )
        return review, ""
    return None, reason


# ------------------------------------------------------------------- retry rules


def rank_verdict(verdict: str) -> int:
    return VERDICT_RANK.get(verdict, VERDICT_RANK["unreviewed"])


def attempt_rank(verdict: str, result: WorkerResult | SectionOut) -> tuple[int, int, int]:
    """'Best attempt' order: critic verdict, then fetched pages, then number of findings."""
    pages = result.n_pages if isinstance(result, SectionOut) else len(result.pages)
    return rank_verdict(verdict), pages, len(result.findings)


def retry_candidates(
    review: Review, retried: set[str], limit: int = MAX_RETRY_SECTIONS
) -> list[str]:
    """Sections to send back: weak or failed, not yet retried, worst first, at most `limit`."""
    weak = [
        r
        for r in review.sections.values()
        if r.verdict in RETRYABLE and r.section_id not in retried
    ]
    weak.sort(key=lambda r: (rank_verdict(r.verdict), r.section_id))
    return [r.section_id for r in weak[:limit]]


def feedback_text(review: Review, section_id: str, urls: list[str]) -> str:
    """The note a retried worker gets: the issues found, the contradictions, the pages read."""
    lines = [
        f"- {i.issue}" + (f" (evidence: {i.evidence})" if i.evidence else "")
        for i in review.sections[section_id].issues
    ]
    lines += [
        f"- contradiction with another section: {c.issue}"
        for c in review.contradictions
        if section_id in c.section_ids
    ]
    return prompts.RETRY_FEEDBACK.format(issues="\n".join(lines), urls=", ".join(urls) or "none")
