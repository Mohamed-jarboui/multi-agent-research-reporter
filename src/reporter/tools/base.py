"""Tool interface: a name, a description for the model, a Pydantic input model and run()."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """A tool failed in a way the agent can act on. The message is shown to the model, so it says
    what went wrong and what to try instead."""


class Page(BaseModel):
    """A page that was fetched and reduced to readable text."""

    url: str
    title: str
    text: str
    truncated: bool = False


class ToolOutput(BaseModel):
    """What a tool returns: text for the model, plus the pages it actually read (if any)."""

    observation: str
    pages: list[Page] = []
    error: bool = False  # True when `observation` is an error message for the model


class Tool(Protocol):
    """Read-only tool. `input_model` validates what the model sends before run() is called."""

    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[type[BaseModel]]

    def run(self, args: Any) -> ToolOutput: ...


def describe(tool: Tool) -> str:
    """One prompt line per tool: name, purpose and its input fields."""
    fields = ", ".join(
        f"{name} ({field.description or 'value'})"
        for name, field in tool.input_model.model_fields.items()
    )
    return f"- {tool.name}: {tool.description} Input fields: {fields}."


def call_tool(tool: Tool, raw_input: Any) -> ToolOutput:
    """Validate `raw_input`, run the tool and turn every failure into an informative observation.

    Never raises: the agent must always get text back so it can adapt.
    """
    try:
        args = tool.input_model.model_validate(raw_input)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}" for e in exc.errors()
        )
        return _error(f"ERROR invalid input for {tool.name}: {problems}")
    try:
        return tool.run(args)
    except ToolError as exc:
        return _error(f"ERROR {tool.name} failed: {exc}")
    except Exception as exc:  # a bug or an unexpected network failure must not kill the run
        return _error(f"ERROR {tool.name} crashed: {type(exc).__name__}: {exc}")


def _error(message: str) -> ToolOutput:
    return ToolOutput(observation=message, error=True)
