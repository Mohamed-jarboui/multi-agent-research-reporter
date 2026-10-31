"""Scoring: citations (no LLM), coverage (one LLM call per report, quotes verified in code) and the
pairwise judge (random order, identities hidden, length-normalised prompt)."""

from __future__ import annotations

import random
import re
from typing import Any

from pydantic import BaseModel, Field

from reporter.agent.checks import check_citations, fetched_urls, normalize_url
from reporter.agent.llm_call import LLM
from reporter.agent.parsing import extract_json, strip_sources_section
from reporter.bench.records import CitationScore, CoverageScore
from reporter.bench.topics import Topic
from reporter.trace import Trace

MIN_QUOTE_CHARS = 20
COVERAGE_REPORT_CHARS = 16_000  # keeps the call small for an 8,000 tokens-per-minute limit
JUDGE_MAX_WORDS = 1_400  # each report is cut to this many words before judging
SCORING_MAX_TOKENS = 1_500
RETRY_TEMPERATURE = 0.7


def word_count(text: str) -> int:
    return len(text.split())


# ------------------------------------------------------------------------------- citations


def score_citations(report: str, trace: Trace) -> CitationScore:
    """Deterministic: the report body against the run's sources list and fetched URLs.

    A citation is valid when its number is in the sources list and that source's URL was fetched
    in this run. `clean` means no invalid number, no uncited source and no unfetched URL.
    """
    fetched = {normalize_url(u) for u in fetched_urls(trace)}
    check = check_citations(strip_sources_section(report), trace.sources, fetched)
    urls = {s.id: normalize_url(s.url) for s in trace.sources}
    valid = [n for n in check.cited if n in urls and urls[n] in fetched]
    return CitationScore(
        cited=len(check.cited),
        valid=len(valid),
        invalid_citations=check.invalid_citations,
        uncited_sources=check.uncited_sources,
        unfetched_urls=check.unfetched_urls,
        clean=check.ok,
    )


# -------------------------------------------------------------------------------- LLM json


class LLMJsonError(Exception):
    """The scoring model produced no usable JSON object after the retry."""


def ask_json(llm: LLM, prompt: str, system: str) -> tuple[dict[str, Any], str]:
    """One JSON answer and the model that gave it. A bad or empty reply is retried once, warmer."""
    last = "no reply"
    for attempt in range(2):
        gen = llm.generate(
            prompt,
            system=system,
            tier="large",
            temperature=0.0 if attempt == 0 else RETRY_TEMPERATURE,
            max_output_tokens=SCORING_MAX_TOKENS,
        )
        try:
            return extract_json(gen.text), gen.model
        except ValueError as exc:
            last = str(exc)
    raise LLMJsonError(last)


# ------------------------------------------------------------------------------- coverage

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
_DASHES = re.compile(r"[‐-―]")
_MARKUP = re.compile(r"[*_`#>]")


def normalize_text(text: str) -> str:
    """For quote matching: lowercase, plain quotes and dashes, no markdown marks, one space."""
    text = _DASHES.sub("-", text.translate(_QUOTES))
    return " ".join(_MARKUP.sub("", text).lower().split())


def quote_in_text(quote: str, text: str) -> bool:
    """True when `quote` (at least 20 characters) occurs in `text` after normalising both."""
    needle = normalize_text(quote)
    return len(needle) >= MIN_QUOTE_CHARS and needle in normalize_text(text)


COVERAGE_SYSTEM = "You check whether a report covers given points. Answer with JSON only."
COVERAGE_PROMPT = """Report topic: {topic}

For each numbered point, say whether the report below covers it. A point is covered only if the
report says something specific about it. If covered, copy ONE sentence or phrase from the report,
word for word (at least 20 characters), that shows it.

Points:
{points}

Reply with JSON only: {{"points": [{{"n": 1, "matched": true, "quote": "exact words"}}]}}
Use "matched": false and "quote": "" for a point that is not covered.

Report:
{report}
"""


def score_coverage(llm: LLM, topic: Topic, report: str) -> CoverageScore:
    """One call for all points. A point counts as covered only if the quote it returned occurs in
    the report text, so a model that claims a match without proof scores 0 for that point."""
    body = strip_sources_section(report)[:COVERAGE_REPORT_CHARS]
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(topic.must_cover, 1))
    data, model = ask_json(
        llm,
        COVERAGE_PROMPT.format(topic=topic.topic, points=numbered, report=body),
        COVERAGE_SYSTEM,
    )
    answers: dict[int, dict[str, Any]] = {}
    for item in data.get("points") or []:
        if isinstance(item, dict) and isinstance(item.get("n"), int):
            answers.setdefault(item["n"], item)
    points: list[dict[str, Any]] = []
    for i, point in enumerate(topic.must_cover, 1):
        item = answers.get(i, {})
        claimed = item.get("matched") is True
        quote = str(item.get("quote") or "")
        points.append(
            {
                "point": point,
                "claimed": claimed,
                "covered": claimed and quote_in_text(quote, body),
                "quote": quote if claimed else "",
            }
        )
    return CoverageScore(
        model=model,
        points=points,
        covered=sum(p["covered"] for p in points),
        total=len(points),
    )


# --------------------------------------------------------------------------------- judge


class Verdict(BaseModel):
    """The judge's answer for reports in a fixed A/B order."""

    winner: str  # "A", "B" or "tie"
    reason: str = ""
    model: str = ""
    a_words: int = 0
    b_words: int = 0
    truncated: bool = False
    swap_checked: bool = False
    consistent: bool | None = None  # only set when the swap check ran


class Judgement(BaseModel):
    """A verdict mapped back to systems, with what is needed to audit it."""

    topic_id: str
    run_index: int
    seed: int
    a_system: str
    b_system: str
    winner_system: str  # a system name or "tie"
    verdict: Verdict
    words: dict[str, int] = Field(default_factory=dict)  # system -> words
    judge_model: str = ""
    same_model_as_writer: bool | None = None  # self-preference risk


JUDGE_SYSTEM = "You are an impartial judge of research reports. Answer with JSON only."
JUDGE_PROMPT = """Topic: {topic}

Compare two research reports on this topic. Judge accuracy of the claims against the cited
sources shown by [n] markers, coverage of the important aspects, clarity and usefulness to a
reader. IGNORE LENGTH: a longer report is not better, and a shorter one is not better either.
Do not reward padding, repetition or formatting. If the quality is about equal, answer "tie".

=== Report A ===
{a}

=== Report B ===
{b}

Reply with JSON only: {{"winner": "A" or "B" or "tie", "reason": "one or two sentences"}}
"""


def _cut(report: str) -> tuple[str, bool]:
    words = strip_sources_section(report).split()
    cut = len(words) > JUDGE_MAX_WORDS
    return " ".join(words[:JUDGE_MAX_WORDS]) + (" [...]" if cut else ""), cut


def _ask_winner(llm: LLM, topic: str, a: str, b: str) -> tuple[str, str, str]:
    data, model = ask_json(llm, JUDGE_PROMPT.format(topic=topic, a=a, b=b), JUDGE_SYSTEM)
    winner = str(data.get("winner", "")).strip()
    if winner.lower() == "tie":
        winner = "tie"
    elif winner.upper() in ("A", "B"):
        winner = winner.upper()
    else:
        raise LLMJsonError(f"unusable winner: {winner!r}")
    return winner, str(data.get("reason", ""))[:500], model


def judge_ordered(
    llm: LLM, topic: str, report_a: str, report_b: str, *, swap_check: bool = False
) -> Verdict:
    """Judge two reports in the given order. With `swap_check`, ask again with A and B swapped;
    an answer that does not follow the swap is recorded as inconsistent and becomes a tie."""
    a, cut_a = _cut(report_a)
    b, cut_b = _cut(report_b)
    winner, reason, model = _ask_winner(llm, topic, a, b)
    verdict = Verdict(
        winner=winner,
        reason=reason,
        model=model,
        a_words=word_count(strip_sources_section(report_a)),
        b_words=word_count(strip_sources_section(report_b)),
        truncated=cut_a or cut_b,
    )
    if swap_check:
        swapped, _, _ = _ask_winner(llm, topic, b, a)
        flipped = {"A": "B", "B": "A", "tie": "tie"}[swapped]
        verdict.swap_checked = True
        verdict.consistent = flipped == winner
        if not verdict.consistent:
            verdict.winner = "tie"
    return verdict


def baseline_is_a(seed: int, topic_id: str, run_index: int) -> bool:
    """True when the baseline report is shown as A. The same inputs always give the same order."""
    return random.Random(f"{seed}:{topic_id}:{run_index}").random() < 0.5


def judge_run_pair(
    llm: LLM,
    topic: Topic,
    run_index: int,
    reports: dict[str, str],
    models_used: set[str],
    *,
    seed: int,
    swap_check: bool = False,
) -> Judgement:
    """Baseline versus multi for one topic and run index, in a seeded random order."""
    a_system, b_system = (
        ("baseline", "multi") if baseline_is_a(seed, topic.id, run_index) else ("multi", "baseline")
    )
    verdict = judge_ordered(
        llm, topic.topic, reports[a_system], reports[b_system], swap_check=swap_check
    )
    winner = {"A": a_system, "B": b_system, "tie": "tie"}[verdict.winner]
    return Judgement(
        topic_id=topic.id,
        run_index=run_index,
        seed=seed,
        a_system=a_system,
        b_system=b_system,
        winner_system=winner,
        verdict=verdict,
        words={a_system: verdict.a_words, b_system: verdict.b_words},
        judge_model=verdict.model,
        same_model_as_writer=verdict.model in models_used,
    )
