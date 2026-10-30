"""Research agents."""

from reporter.agent.baseline import BaselineAgent, RunResult
from reporter.agent.multi import MultiAgent, MultiRunResult

__all__ = ["BaselineAgent", "MultiAgent", "MultiRunResult", "RunResult"]
