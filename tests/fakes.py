"""Fakes shared by the tests: a scripted LLM and a scripted tool. No network anywhere."""

from __future__ import annotations

import json
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
