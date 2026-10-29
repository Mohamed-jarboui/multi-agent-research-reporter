"""Run budgets: a hard ceiling on steps, tokens, wall time and cost, tracked by one meter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Budget:
    """Ceilings for one run. The agents stop researching at `stop_fraction` of any of them, which
    leaves the rest for writing the report."""

    max_steps: int = 15  # ReAct iterations (one model decision plus its tool call)
    max_tokens: int = 150_000  # input + output, as reported by the providers
    max_seconds: float = 300.0
    max_cost_usd: float = 0.50
    stop_fraction: float = 0.8

    def __post_init__(self) -> None:
        if min(self.max_steps, self.max_tokens, self.max_seconds, self.max_cost_usd) <= 0:
            raise ValueError("every budget limit must be positive")
        if not 0 < self.stop_fraction <= 1:
            raise ValueError("stop_fraction must be in (0, 1]")


class BudgetMeter:
    """Counts what a run has used. Tokens a provider did not report count as zero (never guessed);
    the trace records how many calls were unreported."""

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self.steps = 0
        self.tokens = 0
        self.cost_usd = 0.0

    def add_step(self) -> None:
        self.steps += 1

    def add_call(self, tokens: int, cost_usd: float) -> None:
        self.tokens += tokens
        self.cost_usd += cost_usd

    @property
    def elapsed_s(self) -> float:
        return self._clock() - self._start

    def fractions(self) -> dict[str, float]:
        """Share of each limit used so far (1.0 = limit reached)."""
        b = self.budget
        return {
            "steps": self.steps / b.max_steps,
            "tokens": self.tokens / b.max_tokens,
            "seconds": self.elapsed_s / b.max_seconds,
            "cost": self.cost_usd / b.max_cost_usd,
        }

    def stop_reason(self) -> str | None:
        """Name the first limit that is at or past `stop_fraction`, or None while there is room."""
        for name, fraction in self.fractions().items():
            if fraction >= self.budget.stop_fraction:
                return f"budget_{name}"
        return None
