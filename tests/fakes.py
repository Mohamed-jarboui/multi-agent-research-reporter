"""Fakes shared by the tests: a scripted LLM and a scripted tool. No network anywhere."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from typing import Any, ClassVar

from pydantic import BaseModel

from reporter.llm.base import Generation, ProviderError
from reporter.tools import Page, ToolError, ToolOutput


class FakeLLM:
    """Plays back replies in order. A str becomes a Generation; an exception is raised."""

    def __init__(self, replies: list[str | Exception], tokens: tuple[int, int] = (100, 50)) -> None:
        self.replies = list(replies)
        self.tokens = tokens
        self.prompts: list[str] = []
        self.temperatures: list[float] = []

    def generate(self, prompt: str, **kwargs: Any) -> Generation:
        self.prompts.append(prompt)
        self.temperatures.append(kwargs.get("temperature", 0.0))
        if not self.replies:
            raise ProviderError("FakeLLM script exhausted")
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return Generation(
            text=item,
            model="fake-model",
            input_tokens=self.tokens[0],
            output_tokens=self.tokens[1],
        )


def plan(*questions: str) -> str:
    return json.dumps({"questions": list(questions)})


def action(name: str, answers: list[int] | None = None, **tool_input: Any) -> str:
    return json.dumps(
        {"thought": f"do {name}", "action": name, "input": tool_input, "answers": answers or []}
    )


class FakeInput(BaseModel):
    query: str = ""
    url: str = ""


class FakeTool:
    """A read-only tool. `pages` maps a url to the Page it returns; unknown urls fail."""

    description: ClassVar[str] = "fake"
    input_model: ClassVar[type[BaseModel]] = FakeInput

    def __init__(self, name: str, pages: dict[str, Page] | None = None) -> None:
        self.name = name
        self.pages = pages or {}
        self.calls: list[FakeInput] = []

    def run(self, args: FakeInput) -> ToolOutput:
        self.calls.append(args)
        if self.name == "web_search":
            return ToolOutput(observation="1. Result\n   https://a.example/1\n   snippet")
        page = self.pages.get(args.url)
        if page is None:
            raise ToolError(f"{args.url} answered HTTP 404. Pick another source.")
        return ToolOutput(observation=f"Page: {page.title}\n{page.text}", pages=[page])


def page(n: int) -> Page:
    return Page(
        url=f"https://site{n}.example/p", title=f"Site {n}", text=f"Facts from site {n}. " * 5
    )


# ------------------------------------------------------------- multi-agent fakes


class FakeClock:
    """A clock that only moves when told to; its `sleep` advances it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += seconds


def sections_json(n: int = 3) -> str:
    return json.dumps(
        {
            "sections": [
                {
                    "title": f"Part {k}",
                    "goal": f"Goal {k}",
                    "scope": [f"point {k}"],
                    "exclude": ["the other parts"],
                    "expected_output": "3 findings",
                }
                for k in range(1, n + 1)
            ]
        }
    )


def team_page(k: int, j: int) -> Page:
    return Page(
        url=f"https://s{k}-{j}.example/p", title=f"S{k}-{j}", text=f"Facts of {k}-{j}. " * 20
    )


def team_pages(n_sections: int = 6) -> dict[str, Page]:
    return {team_page(k, j).url: team_page(k, j) for k in range(1, n_sections + 1) for j in (1, 2)}


class TeamLLM:
    """Thread-safe fake for the multi-agent graph. It answers by kind of prompt (plan, think,
    final) and by section ("Part k"): fetch two pages, finish, then write two findings.

    `on_call(kind, title, nth)` may return a str or an Exception to replace the default reply.
    """

    def __init__(self, n_sections: int = 3, on_call=None, tokens: tuple[int, int] = (100, 50)):
        self.n_sections = n_sections
        self.on_call = on_call
        self.tokens = tokens
        self.lock = threading.Lock()
        self.counts: Counter[tuple[str, str]] = Counter()
        self.systems: list[str] = []

    def generate(self, prompt: str, **kwargs: Any) -> Generation:
        system = kwargs.get("system") or ""
        kind = (
            "plan"
            if "supervisor" in system
            else "final"
            if "extract findings" in system
            else "think"
        )
        match = re.search(r"Your section: (.*)", prompt)
        title = match.group(1).strip() if match else ""
        with self.lock:
            self.counts[(kind, title)] += 1
            nth = self.counts[(kind, title)]
        reply = self.on_call(kind, title, nth) if self.on_call else None
        if reply is None:
            reply = self._default(kind, title, prompt)
        if isinstance(reply, Exception):
            raise reply
        return Generation(reply, "fake-model", self.tokens[0], self.tokens[1])

    def _default(self, kind: str, title: str, prompt: str) -> str:
        if kind == "plan":
            return sections_json(self.n_sections)
        k = title.split()[-1]
        if kind == "final":
            return json.dumps(
                {
                    "findings": [
                        {"claim": f"{title} fact one", "sources": [1]},
                        {"claim": f"{title} fact two", "sources": [1, 2, 9]},
                    ],
                    "confidence": "high",
                    "open_questions": [f"Open about {title}"],
                }
            )
        read = prompt.count(f"(https://s{k}-")
        if read >= 2:
            return action("finish")
        return action("fetch_page", url=team_page(int(k), read + 1).url)


def team_tools(fetch: FakeTool | None = None) -> dict[str, FakeTool]:
    return {
        "web_search": FakeTool("web_search"),
        "fetch_page": fetch or FakeTool("fetch_page", team_pages()),
    }
