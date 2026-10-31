"""One charged model call, shared by the baseline and the multi-agent system.

`generate_charged` calls the model, records tokens, cost and latency on the trace step and charges
the run budget. `LLMGate` limits how many model calls run at once and makes all workers pause
together after a rate-limit error.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from reporter import otel
from reporter.budget import BudgetExhausted, BudgetMeter
from reporter.llm.base import Generation, ProviderError, RateLimitError, Tier
from reporter.pricing import call_cost_usd
from reporter.trace import StepRecord

MAX_EMPTY_REPLY_ATTEMPTS = 2
RETRY_TEMPERATURE = 0.7  # a retry at temperature 0 would repeat the same empty reply
COOLDOWN_BASE_S = 10.0
COOLDOWN_MAX_S = 30.0


class LLM(Protocol):
    """What the agents need from a model: FallbackLLM satisfies it, and so do test fakes."""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        tier: Tier = "large",
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
    ) -> Generation: ...


def llm_summary(llm: object) -> dict[str, Any]:
    """The router's per-provider summary for `trace.llm` ({} for a model without one)."""
    summary = getattr(llm, "summary", None)
    return summary() if callable(summary) else {}


def is_rate_limit(error: BaseException) -> bool:
    """A rate limit, directly or hidden as the cause of AllProvidersFailedError."""
    return isinstance(error, RateLimitError) or isinstance(error.__cause__, RateLimitError)


class LLMGate:
    """Caps concurrent model calls and shares one cooldown after a rate limit.

    A call that fails with a rate limit starts a cooldown (10 s, then 20 s, capped at 30 s); every
    caller waits for it before its next call, and the failed call is retried once. Tools never go
    through the gate, so one worker can fetch a page while another waits for the model.
    """

    def __init__(
        self,
        max_parallel: int = 2,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._slots = threading.Semaphore(max_parallel)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._cooldown_until = 0.0
        self._trips = 0

    def generate(
        self,
        llm: LLM,
        prompt: str,
        *,
        system: str | None,
        temperature: float,
        max_output_tokens: int | None,
        max_wait_s: float | None = None,
        on_wait: Callable[[float], None] | None = None,
        before_call: Callable[[], None] | None = None,
    ) -> Generation:
        """llm.generate under a slot. `max_wait_s` caps one cooldown wait; `on_wait` hears of it.

        `before_call` runs once the slot is held, right before the model is called (it may raise
        to cancel the call), so a check made there sees every call that is already in flight.
        """
        for attempt in range(2):
            self._wait_cooldown(max_wait_s, on_wait)
            try:
                with self._slots:
                    if before_call is not None:
                        before_call()
                    gen = llm.generate(
                        prompt,
                        system=system,
                        tier="large",
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    )
            except ProviderError as exc:
                if not is_rate_limit(exc) or attempt == 1:
                    raise
                self._trip()
                continue
            with self._lock:
                self._trips = 0
            return gen
        raise AssertionError("unreachable")  # pragma: no cover

    def _trip(self) -> None:
        with self._lock:
            self._trips += 1
            wait = min(COOLDOWN_BASE_S * self._trips, COOLDOWN_MAX_S)
            self._cooldown_until = max(self._cooldown_until, self._clock() + wait)

    def _wait_cooldown(self, max_wait_s: float | None, on_wait: Callable[[float], None] | None):
        with self._lock:
            remaining = self._cooldown_until - self._clock()
        if remaining <= 0:
            return
        if max_wait_s is not None:
            remaining = min(remaining, max(max_wait_s, 0.0))
        self._sleep(remaining)
        if on_wait is not None:
            on_wait(remaining)


def generate_charged(
    llm: LLM,
    step: StepRecord,
    prompt: str,
    system: str,
    max_tokens: int,
    meter: BudgetMeter,
    gate: LLMGate | None = None,
    *,
    clock: Callable[[], float] = time.monotonic,
    temperature: float = 0.0,
    enforce_limit: bool = False,
    max_wait_s: float | None = None,
    on_wait: Callable[[float], None] | None = None,
    cost_fn: Callable[[str, int | None, int | None], float | None] = call_cost_usd,
) -> Generation:
    """Call the model and charge the step and the budget for it.

    An empty reply is retried once at a higher temperature (gpt-oss on Groq often returns
    only reasoning at temperature 0, measured on 2026-10-29); both calls are charged
    to the same step, so the trace shows what the step really cost. With `enforce_limit`, no call
    starts once a hard budget limit is reached (BudgetExhausted).
    """

    def check_budget() -> None:
        if enforce_limit and (reason := meter.hard_limit_reached()):
            raise BudgetExhausted(reason)

    for attempt in range(MAX_EMPTY_REPLY_ATTEMPTS):
        check_budget()
        temp = temperature if attempt == 0 else max(temperature, RETRY_TEMPERATURE)
        started = clock()
        with otel.span(
            "llm.generate", kind="generation", attributes={"gen_ai.operation.name": "chat"}
        ) as sp:
            try:
                if gate is None:
                    gen = llm.generate(
                        prompt,
                        system=system,
                        tier="large",
                        temperature=temp,
                        max_output_tokens=max_tokens,
                    )
                else:
                    gen = gate.generate(
                        llm,
                        prompt,
                        system=system,
                        temperature=temp,
                        max_output_tokens=max_tokens,
                        max_wait_s=max_wait_s,
                        on_wait=on_wait,
                        before_call=check_budget,
                    )
            except ProviderError as exc:
                otel.set_llm_failure(sp, exc)
                raise
            finally:
                step.llm_latency_s = round(step.llm_latency_s + clock() - started, 3)
            cost = cost_fn(gen.model, gen.input_tokens, gen.output_tokens)
            otel.set_generation(
                sp,
                provider=gen.provider,
                model=gen.model,
                input_tokens=gen.input_tokens,
                output_tokens=gen.output_tokens,
                cost_usd=cost,
            )
        step.model = gen.model
        step.input_tokens = _add(step.input_tokens, gen.input_tokens, first=attempt == 0)
        step.output_tokens = _add(step.output_tokens, gen.output_tokens, first=attempt == 0)
        step.cost_usd = _add(step.cost_usd, cost, first=attempt == 0)
        meter.add_call((gen.input_tokens or 0) + (gen.output_tokens or 0), cost or 0.0)
        if gen.text.strip():
            break
        step.note = "model returned an empty reply" + (" twice" if attempt else "; retried")
    return gen


def _add(total: float | None, new: float | None, *, first: bool) -> float | None:
    """Sum a per-call figure into a step total; None (not reported) stays None unless known."""
    if first:
        return new
    if total is None or new is None:
        return None
    return total + new
