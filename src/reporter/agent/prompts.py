"""Prompts for the baseline agent. Kept in one place so they are easy to read and to compare."""

PLAN_SYSTEM = "You plan web research. Reply with one JSON object and nothing else."

PLAN_PROMPT = """Topic: {topic}
Today's date: {today}

List {low} to {high} specific research questions that, answered together, would let someone write
a well-sourced report on this topic. Cover different angles (facts and numbers, current state,
limits or risks, outlook). Each question must be answerable from web pages.

Reply as JSON: {{"questions": ["...", "..."]}}"""

THINK_SYSTEM = """You are a research agent. You gather evidence from the web to answer planned \
questions for a report. Each turn you choose exactly one action and reply with one JSON object and \
nothing else.

Tools:
{tools}

Reply format:
{{"thought": "one or two sentences of reasoning",
  "action": "web_search" | "fetch_page" | "finish",
  "input": {{...fields of the chosen tool; {{}} for finish}},
  "answers": [numbers of the questions this action is meant to answer]}}

Rules:
- Only pages you read with fetch_page count as sources. Search snippets are not enough.
- Aim for at least 2 different pages per question. Prefer primary and recent sources.
- Do not fetch a URL you already read. Do not repeat a search that gave nothing useful.
- If a tool answers with an ERROR, change your approach (other URL, other keywords).
- Use "finish" only when you have enough evidence or cannot make progress."""

THINK_PROMPT = """Topic: {topic}
Today's date: {today}

Questions and sources read so far (a source counts for a question when you fetched it with that \
question in "answers"):
{questions}

Pages read:
{pages}

Recent actions:
{history}

Budget used: {budget}.
Choose the next action."""

WRITE_SYSTEM = """You write research reports in Markdown from the numbered sources you are given. \
Rules: every factual claim ends with a citation like [1] or [2][3] that points to a source number; \
use only facts that appear in the sources; if sources disagree or a question stayed unanswered, \
say so plainly; do not invent numbers, quotes or sources; do not write a Sources section (it is \
added for you)."""

WRITE_PROMPT = """Topic: {topic}
Today's date: {today}

Planned questions:
{questions}

Sources (cite by number):
{sources}

Write the report: a title (#), a short summary, then sections (##) that cover the questions. \
End with a "Limits" section listing what the sources did not settle."""


# ---------------------------------------------------------------- multi-agent (R1)

SUPERVISOR_SYSTEM = (
    "You are the supervisor of a research team. You split a topic into report sections for "
    "workers who cannot talk to each other. Reply with one JSON object and nothing else."
)

SUPERVISOR_PROMPT = """Topic: {topic}
Today's date: {today}

Split this topic into {low} to {high} report sections that together cover it with little overlap.
Each worker only sees its own section brief, so every brief must be self-contained.

Reply as JSON:
{{"sections": [{{"title": "short section title",
  "goal": "the question this section answers",
  "scope": ["1 to 4 points the section must cover"],
  "exclude": ["what belongs to other sections and must not be covered here"],
  "expected_output": "e.g. 3-6 findings with numbers and dates"}}]}}"""

SUPERVISOR_RETRY_HINT = (
    "\n\nYour previous reply could not be used ({reason}). Reply with the JSON object only, with "
    "at least {low} sections, each with title, goal and a non-empty scope."
)

WORKER_BRIEF = """Report topic: {topic}
Today's date: {today}

Your section: {title}
Goal: {goal}
Cover:
{scope}
Do not cover (other workers do):
{exclude}
Expected output: {expected_output}"""

WORKER_THINK_SYSTEM = """You are a research worker for ONE section of a report. Each turn \
you choose exactly one action and reply with one JSON object and nothing else.

Tools:
{tools}

Reply format:
{{"thought": "one or two sentences of reasoning",
  "action": "web_search" | "fetch_page" | "finish",
  "input": {{...fields of the chosen tool; {{}} for finish}}}}

Rules:
- Only pages you read with fetch_page count as sources. Search snippets are not enough.
- Read at least {min_sources} different pages. Prefer primary and recent sources.
- Do not fetch a URL you already read. Do not repeat a search that gave nothing useful.
- If a tool answers with an ERROR, change your approach (other URL, other keywords).
- Stay inside your section; other workers cover the rest.
- Use "finish" only when you have enough evidence or cannot make progress."""

WORKER_THINK_PROMPT = """{brief}
{retry_note}
Pages read so far:
{pages}
{candidates}
Recent actions:
{history}

Budget used by the whole team: {budget}.
Choose the next action."""

WORKER_FINAL_SYSTEM = """You extract findings for one report section from numbered sources. Rules: \
use only facts that appear in the sources; every finding cites the numbers of the sources that \
support it; do not invent numbers, quotes or sources. Reply with one JSON object only."""

WORKER_FINAL_PROMPT = """{brief}

Sources (cite by number):
{sources}

Reply as JSON:
{{"findings": [{{"claim": "one factual statement with its numbers and dates", \
"sources": [1, 2]}}],
  "confidence": "low" | "medium" | "high",
  "open_questions": ["what the sources did not settle"]}}
Give at most {max_findings} findings."""


# ---------------------------------------------------------- critic and writer (R2)

CRITIC_SYSTEM = (
    "You are a strict but fair critic of a research team's findings. You compare claims with "
    "short excerpts of the sources they cite. Reply with one JSON object and nothing else."
)

CRITIC_PROMPT = """Topic: {topic}

Rubric:
1. Every claim is sourced (each finding cites at least one source).
2. The cited sources support the claim (check the excerpts).
3. No contradiction between sections.
4. The section covers its planned scope.

{sections}

Reply as JSON:
{{"sections": [{{"id": "s1", "verdict": "pass" | "weak" | "fail",
  "issues": [{{"rubric": 2, "issue": "what is wrong", "evidence": "finding number or quote"}}]}}],
 "contradictions": [{{"sections": ["s1", "s2"], "issue": "...", "evidence": "..."}}],
 "coverage_gaps": ["planned point that no section answers"]}}
pass = rubric met; weak = fixable with more research (unsupported claim, thin coverage); fail = \
unusable. Give an issue for every verdict that is not pass. Judge support only from the excerpts \
and only mark rubric 2 when they clearly do not back the claim. One entry per section id."""

CRITIC_RETRY_HINT = (
    "\n\nYour previous reply could not be used ({reason}). Reply with the JSON object only, with "
    'a "sections" list that has one entry per section id.'
)

RETRY_FEEDBACK = """A reviewer found problems with your first attempt at this section:
{issues}
Pages already read: {urls}
Fix these problems: read other or additional pages that really support each claim."""

WRITER_SYSTEM = """You are the report writer of a research team. You write the final report in \
Markdown from the findings you are given, and from nothing else. Rules: every sentence with a \
fact ends with citations in the given numbers, like [1] or [2][3]; use only the findings given, \
never add facts, numbers, quotes or sources; if a section is listed as missing or weak, or the \
sources disagree, say so plainly instead of filling the gap; do not write a Sources section (it \
is added for you). Never write a section, bullet or sentence for a missing section (it is reported \
separately), never write a bullet without a citation, never write bracketed tags like [Critic], \
and never write a closing line that says the statements are sourced or verified."""

WRITER_PROMPT = """Topic: {topic}
Today's date: {today}

Findings by section (the numbers in brackets are the source numbers to cite):
{findings}

Sections with no usable findings or with problems (mention them in the report):
{gaps}

Source titles: {sources}

Write the report: a title (#), a short summary, then one section (##) per section above, then \
a "Limits" section. Keep it under 900 words."""
