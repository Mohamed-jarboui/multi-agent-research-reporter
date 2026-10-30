"""Read-only research tools."""

from reporter.tools.base import Page, Tool, ToolError, ToolOutput, call_tool, describe
from reporter.tools.fetch import FetchPage
from reporter.tools.search import WebSearch


def default_tools() -> dict[str, Tool]:
    """The two tools the agents use, keyed by name."""
    tools: list[Tool] = [WebSearch(), FetchPage()]
    return {t.name: t for t in tools}


__all__ = [
    "FetchPage",
    "Page",
    "Tool",
    "ToolError",
    "ToolOutput",
    "WebSearch",
    "call_tool",
    "default_tools",
    "describe",
]
