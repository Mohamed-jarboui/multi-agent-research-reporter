"""Run budgets: a hard ceiling on steps, tokens, wall time and cost, tracked by one meter."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


class BudgetExhausted(Exception):
    """A hard limit (100% of steps, tokens, time or cost) was reached; no more model calls."""


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


@dataclass(frozen=True)
class MultiBudget:
    """Budget of a multi-agent run: one shared run budget plus limits per worker (section).

    These are starting defaults, not measured values.
    """

    run: Budget = field(
        default_factory=lambda: Budget(
            max_steps=60,  # research gets 70% (42): 5 sections x 6 think steps = 30, room to spare
            max_tokens=300_000,
            max_seconds=480.0,
            max_cost_usd=0.50,
            stop_fraction=0.7,  # research stops here; the last 30% is for critic and writer
        )
    )
    worker_max_steps: int = 6
    worker_max_tokens: int = 50_000
    worker_max_seconds: float = 150.0  # per section, both attempts
    tool_max_seconds: float = 20.0
    max_pages_per_worker: int = 4
    min_sources: int = 2
    max_parallel_workers: int = 3
    max_parallel_llm_calls: int = 2

    def __post_init__(self) -> None:
        positive = (
            self.worker_max_steps,
            self.worker_max_tokens,
            self.worker_max_seconds,
            self.tool_max_seconds,
            self.max_pages_per_worker,
            self.min_sources,
            self.max_parallel_workers,
            self.max_parallel_llm_calls,
        )
        if min(positive) <= 0:
            raise ValueError("every multi-agent budget limit must be positive")


class BudgetMeter:
    """Counts what a run has used. Tokens a provider did not report count as zero (never guessed);
    the trace records how many calls were unreported. Safe to share between threads."""

    def __init__(self, budget: Budget, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self._lock = threading.Lock()
        self.steps = 0
        self.tokens = 0
        self.cost_usd = 0.0

    def add_step(self) -> None:
        with self._lock:
            self.steps += 1

    def add_call(self, tokens: int, cost_usd: float) -> None:
        with self._lock:
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

    def stop_reason(self, fraction: float | None = None) -> str | None:
        """Name the first limit at or past `fraction` (default: the budget's `stop_fraction`),
        or None while there is room."""
        limit = self.budget.stop_fraction if fraction is None else fraction
        for name, used in self.fractions().items():
            if used >= limit:
                return f"budget_{name}"
        return None

    def hard_limit_reached(self) -> str | None:
        """Name the first limit that is at 100% (no model call may start), or None."""
        for name, fraction in self.fractions().items():
            if fraction >= 1.0:
                return f"budget_{name}"
        return None
