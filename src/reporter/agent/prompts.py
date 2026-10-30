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
