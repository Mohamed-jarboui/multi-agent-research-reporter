"""The writer: turns the checked findings into a cited Markdown report.

The model writes the body only. Code builds the Sources list from the global sources list and the
"Coverage notes" that name missing and weak sections, so neither depends on the model.
"""

from __future__ import annotations

import re

from reporter.agent import prompts
from reporter.agent.checks import clean_text, section_reason
from reporter.agent.llm_call import generate_charged
from reporter.agent.parsing import citation_numbers, strip_sources_section
from reporter.agent.schemas import Finding, MergedResearch, Review, SectionOut
from reporter.agent.worker import WorkerContext
from reporter.budget import BudgetExhausted
from reporter.llm.base import ProviderError
from reporter.trace import StepRecord

WRITER_MAX_TOKENS = 3_500
MIN_BODY_CHARS = 40


_FINDING_REF = re.compile(r"(?<![A-Za-z0-9])F([0-9]+)(?![A-Za-z0-9])")


MIN_EVIDENCE_CHARS = 15  # a quote of this length (besides F<n>) counts as evidence


def flagged_findings(section: SectionOut, review: Review) -> dict[int, bool]:
    """0-based index -> True when the critic quoted evidence, for findings it marked as unsourced
    or unsupported (rubric 1 or 2, naming the finding as F<n>). Decided in code."""
    verdict = review.sections.get(section.brief.id)
    flagged: dict[int, bool] = {}
    for issue in verdict.issues if verdict else []:
        if issue.origin == "critic" and issue.rubric in (1, 2):
            quoted = (
                len(_FINDING_REF.sub("", issue.evidence).strip(" .,;:()'\"")) >= MIN_EVIDENCE_CHARS
            )
            for ref in _FINDING_REF.findall(f"{issue.evidence} {issue.issue}"):
                if 1 <= int(ref) <= len(section.findings):
                    index = int(ref) - 1
                    flagged[index] = flagged.get(index, False) or quoted
    return flagged


def unverified_findings(section: SectionOut, review: Review) -> list[int]:
    return sorted(flagged_findings(section, review))


def unverified_lines(merged: MergedResearch, review: Review) -> list[str]:
    """Claims flagged without quoted evidence: kept, but labelled unverified in the report."""
    lines: list[str] = []
    for section in merged.sections:
        for i, quoted in sorted(flagged_findings(section, review).items()):
            if not quoted:
                finding = section.findings[i]
                cites = "".join(f"[{n}]" for n in finding.source_ids)
                lines.append(f"{finding.claim} {cites}")
    return lines


def verified_findings(section: SectionOut, review: Review) -> list[Finding]:
    """The findings the writer may state as facts."""
    out = set(unverified_findings(section, review))
    return [f for i, f in enumerate(section.findings) if i not in out]


def usable_sections(merged: MergedResearch, review: Review | None = None) -> list[SectionOut]:
    """Sections with at least one finding the writer may use."""
    if review is None:
        return [s for s in merged.sections if s.findings]
    return [s for s in merged.sections if verified_findings(s, review)]


def gap_lines(merged: MergedResearch, review: Review, *, for_prompt: bool = False) -> list[str]:
    """Plain lines about missing, weak and unverified material. No raw error text."""
    lines: list[str] = []
    for section in merged.sections:
        title = section.brief.title
        verdict = review.sections.get(section.brief.id)
        if not section.findings:
            reason = section_reason(section)
            lines.append(f'Missing: "{title}" has no usable findings ({reason}).')
            continue
        if verdict is not None and verdict.verdict in ("weak", "fail"):
            first = next((i for i in verdict.issues if i.origin == "critic"), None)
            detail = clean_text(first.issue) if first else "no detail given"
            lines.append(f'Weak: "{title}" was judged {verdict.verdict} by the critic ({detail}).')
        n_issues = sum(i.origin == "critic" for i in verdict.issues) if verdict else 0
        flagged = flagged_findings(section, review)
        marked = [i for i, quoted in sorted(flagged.items()) if quoted]
        if n_issues:
            lines.append(
                f'Critic issues: "{title}" has {n_issues} issue(s); {len(marked)} claim(s) '
                f"omitted (evidence quoted), {len(flagged) - len(marked)} listed as unverified."
            )
        for i in [] if for_prompt else marked:
            lines.append(
                f"Unverified, not used in the text: {clean_text(section.findings[i].claim)}"
            )
    lines += [f"Gap: {clean_text(gap)}" for gap in review.coverage_gaps]
    lines += [f"Contradiction: {clean_text(c.issue)}" for c in review.contradictions]
    if review.critic_status != "ok":
        lines.append("The critic did not review the claims; they were not independently checked.")
    return lines


def coverage_notes(merged: MergedResearch, review: Review) -> str:
    lines = gap_lines(merged, review)
    return "## Coverage notes\n\n" + "\n".join(f"- {line}" for line in lines) if lines else ""


def sources_list(merged: MergedResearch) -> str:
    return "\n".join(f"{s.id}. [{s.title}]({s.url})" for s in merged.sources)


def render_writer_prompt(topic: str, today: str, merged: MergedResearch, review: Review) -> str:
    blocks: list[str] = []
    for section in usable_sections(merged, review):
        lines = [f"## {section.brief.title}"]
        lines += [
            f"- {f.claim} " + "".join(f"[{n}]" for n in f.source_ids)
            for f in verified_findings(section, review)
        ]
        questions = [q for sid, q in merged.open_questions if sid == section.brief.id]
        if questions:
            lines.append("Open questions: " + "; ".join(questions))
        blocks.append("\n".join(lines))
    gaps = gap_lines(merged, review, for_prompt=True)
    return prompts.WRITER_PROMPT.format(
        topic=topic,
        today=today,
        findings="\n\n".join(blocks),
        gaps="\n".join(f"- {g}" for g in gaps) or "- none",
        sources="; ".join(f"[{s.id}] {s.title}" for s in merged.sources),
    )


def run_writer(
    ctx: WorkerContext, merged: MergedResearch, review: Review, today: str
) -> tuple[str, str]:
    """One model call for the report body. Returns (body, "") or ("", reason). Never raises."""
    step = StepRecord(index=0, kind="write")
    try:
        gen = generate_charged(
            ctx.llm,
            step,
            render_writer_prompt(ctx.topic, today, merged, review),
            prompts.WRITER_SYSTEM,
            WRITER_MAX_TOKENS,
            ctx.meter,
            ctx.gate,
            clock=ctx.clock,
            enforce_limit=True,
            max_wait_s=max(ctx.budget.run.max_seconds - ctx.meter.elapsed_s, 0.0),
            on_wait=lambda w: ctx.trace.add_rate_limit_wait("writer", ctx.meter.elapsed_s, w),
            cost_fn=ctx.cost_fn,
        )
    except (ProviderError, BudgetExhausted) as exc:
        ctx.trace.notes.append(f"writer_error: {exc}")  # raw detail stays in the trace
        return "", "model call failed"
    finally:
        ctx.trace.add_step(step)
    body = strip_sources_section(gen.text)
    if len(body) < MIN_BODY_CHARS:
        return "", "the writer reply was empty or too short"
    if not body.startswith("#"):
        body = f"# {ctx.topic}\n\n{body}"
    return body, ""


# ------------------------------------------------------------------ writer output rules

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_STRAY_TOKEN = re.compile(r"\s*\[[A-Za-z_][A-Za-z_ ]{0,40}\](?!\()")  # [Critic], [budget_steps]
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_ASSURANCE = re.compile(
    r"\b(?:all|every)\b.{0,40}\b(?:statements?|claims?|facts?|sentences?|information)\b"
    r".{0,60}\b(?:sourced|cited|supported|backed|verified)\b",
    re.IGNORECASE,
)


def _title_key(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(w for w in words if w != "and")


def sanitize_body(body: str, merged: MergedResearch, review: Review) -> tuple[str, dict[str, int]]:
    """Enforce the writer output rules in code. Returns (text, counts of what was removed).

    - bracketed non-numeric tokens such as [Critic] are removed;
    - blanket assurance lines ("all statements are sourced") are removed;
    - a section with no usable findings is dropped (it lives in the coverage notes only);
    - an uncited bullet under a section heading is dropped.
    """
    counts = {"stray_tokens": 0, "assurance_lines": 0, "dropped_sections": 0, "uncited_bullets": 0}
    empty = {_title_key(s.brief.title) for s in merged.sections} - {
        _title_key(s.brief.title) for s in usable_sections(merged, review)
    }
    # block = [heading line or None, level, kept body lines, dropped something]
    blocks: list[list] = [[None, 0, [], False]]
    skip_level = 0  # inside a dropped section: skip lines until a heading of this level or higher
    skip_continuation = False
    for raw in body.splitlines():
        line, n = _STRAY_TOKEN.subn("", raw)
        counts["stray_tokens"] += n
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skip_level and level > skip_level:
                continue
            skip_level = 0
            skip_continuation = False
            if level >= 2 and _title_key(heading.group(2)) in empty:
                skip_level = level
                counts["dropped_sections"] += 1
                continue
            blocks.append([line, level, [], False])
            continue
        assurance = _ASSURANCE.search(line) is not None
        if skip_level:
            counts["assurance_lines"] += assurance
            continue
        block = blocks[-1]
        if assurance:
            counts["assurance_lines"] += 1
            block[3] = True
            while block[2] and (not block[2][-1].strip() or _RULE.match(block[2][-1])):
                block[2].pop()
            continue
        if block[0] is not None and block[1] >= 2:
            if _BULLET.match(line) and not citation_numbers(line):
                counts["uncited_bullets"] += 1
                block[3] = True
                skip_continuation = True
                continue
            if skip_continuation and line.startswith((" ", "\t")) and not _BULLET.match(line):
                continue
        skip_continuation = False
        block[2].append(line)
    out: list[str] = []
    for head, _level, lines, dropped in blocks:
        if head is not None and dropped and not any(ln.strip() for ln in lines):
            continue  # the section only had removed lines
        out += ([head] if head is not None else []) + lines
    return "\n".join(out).strip() + "\n", counts


def claim_counts(merged: MergedResearch, review: Review) -> tuple[int, int, int]:
    """(claims cited, claims left out or unverified, sections missing or weak), from the data."""
    cited = left_out = 0
    for section in merged.sections:
        cited += len(verified_findings(section, review))
        left_out += len(flagged_findings(section, review))
    problems = 0
    for section in merged.sections:
        verdict = review.sections.get(section.brief.id)
        if not section.findings or (verdict is not None and verdict.verdict in ("weak", "fail")):
            problems += 1
    return cited, left_out, problems


def check_line(merged: MergedResearch, review: Review) -> str:
    cited, left_out, problems = claim_counts(merged, review)
    return (
        f"{cited} claims cited, {left_out} claims left out or marked unverified, "
        f"{problems} sections missing or weak."
    )


def assemble_report(body: str, merged: MergedResearch, review: Review) -> str:
    """Body, then the code-built coverage notes, the claim count line and the Sources list."""
    parts = [body.rstrip()]
    if unverified := unverified_lines(merged, review):
        parts.append(
            "## Unverified claims\n\nThe critic could not confirm these against the source "
            "excerpt it saw; treat them as unverified.\n\n"
            + "\n".join(f"- (unverified) {line}" for line in unverified)
        )
    if notes := coverage_notes(merged, review):
        parts.append(notes)
    parts.append(f"**Claim check:** {check_line(merged, review)}")
    parts.append("## Sources\n\n" + sources_list(merged))
    return "\n\n".join(parts) + "\n"


def no_report_text(topic: str, merged: MergedResearch, review: Review) -> str:
    """The honest output when no section has a usable finding."""
    lines = [
        f"# {topic}",
        "",
        "No report could be written: no section has a finding backed by a fetched page.",
    ]
    if notes := coverage_notes(merged, review):
        lines += ["", notes]
    return "\n".join(lines) + "\n"
