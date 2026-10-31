"""OpenTelemetry through the API only: spans are no-ops until a provider is set.

The CLI and the benchmark never set one, so they pay nothing. The service sets one with
`set_provider` (see `reporter.service.telemetry`). Attribute names follow the OTel GenAI
conventions plus the `langfuse.*` keys Langfuse reads. No prompt, completion or tool input is
ever recorded, and an exception's message is not recorded either (only its class name).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

SESSION_ID = "langfuse.session.id"
OBSERVATION_TYPE = "langfuse.observation.type"
TRACE_NAME = "langfuse.trace.name"

_provider: Any = None


def set_provider(provider: Any) -> None:
    """Use this TracerProvider (None goes back to the no-op one). Not the OTel global, so tests
    and the service can each have their own."""
    global _provider
    _provider = provider


def get_tracer() -> Tracer:
    if _provider is not None:
        return _provider.get_tracer("reporter")
    return trace.get_tracer("reporter")


def job_context(span: Span) -> otel_context.Context:
    """A context whose current span is `span`, to pass explicitly as a parent."""
    return trace.set_span_in_context(span)


def current_span() -> Span:
    return trace.get_current_span()


@contextmanager
def span(
    name: str,
    *,
    kind: str = "span",
    parent: otel_context.Context | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """A span that is current inside the block. `parent=None` uses the current context.

    On an exception the span gets status ERROR and `error.type` (the class name only), and
    the exception goes on.
    """
    attrs = {OBSERVATION_TYPE: kind, **(attributes or {})}
    with get_tracer().start_as_current_span(
        name,
        context=parent,
        attributes=attrs,
        record_exception=False,
        set_status_on_exception=False,
    ) as sp:
        try:
            yield sp
        except BaseException as exc:
            sp.set_attribute("error.type", type(exc).__name__)
            sp.set_status(Status(StatusCode.ERROR))
            raise


def set_generation(
    sp: Span,
    *,
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    cost_usd: float | None,
) -> None:
    """The gen_ai.* attributes of a finished call. Unknown figures are left out, never 0."""
    if provider:
        sp.set_attribute("gen_ai.system", provider)
    if model:
        sp.set_attribute("gen_ai.request.model", model)
        sp.set_attribute("gen_ai.response.model", model)
    if input_tokens is not None:
        sp.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    if output_tokens is not None:
        sp.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    if cost_usd is not None:
        sp.set_attribute("gen_ai.usage.cost", cost_usd)


def set_llm_failure(sp: Span, error: BaseException) -> None:
    """Limit type and retry delay of a rate-limit error (fixed enum values, never raw text)."""
    if not getattr(error, "limit", None) and error.__cause__ is not None:
        error = error.__cause__  # AllProvidersFailedError hides the 429 as its cause
    limit = getattr(error, "limit", None)
    if limit:
        sp.set_attribute("reporter.llm.limit_type", str(limit))
    retry_after = getattr(error, "retry_after_s", None)
    if retry_after is not None:
        sp.set_attribute("reporter.llm.retry_after_s", float(retry_after))
