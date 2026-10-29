# Copied from github.com/Mohamed-jarboui/eu-ai-act-copilot (tests/); package renamed.
"""FallbackLLM.generate_stream with fake providers."""

import pytest

from reporter.llm import FallbackLLM, RateLimitError, ServerError
from reporter.llm.base import Generation, StreamEvent


class FakeStreamProvider:
    """Streams its pieces; `fail_before` raises on the first pull, `fail_after` after n pieces."""

    def __init__(self, name, pieces=(), *, fail_before=None, fail_after=None):
        self.name, self.pieces = name, list(pieces)
        self.fail_before, self.fail_after, self.calls = fail_before, fail_after, 0

    def generate_stream(self, prompt, **kwargs):
        self.calls += 1
        if self.fail_before:
            raise self.fail_before
        for i, piece in enumerate(self.pieces):
            if self.fail_after is not None and i == self.fail_after:
                raise ServerError("connection dropped", provider=self.name)
            yield StreamEvent(text=piece)
        yield StreamEvent(final=Generation("".join(self.pieces), f"{self.name}-m", 11, 7))


def make_llm(*providers):
    ticks = iter(range(0, 1000))
    return FallbackLLM(
        list(providers), max_attempts=2, sleep=lambda _s: None, clock=lambda: float(next(ticks))
    )


def run(llm):
    return list(llm.generate_stream("hi"))


def test_streams_deltas_then_a_final_event():
    events = run(make_llm(FakeStreamProvider("gemini", ["Hel", "lo"])))
    assert [e.text for e in events[:-1]] == ["Hel", "lo"]
    assert events[-1].final.text == "Hello"


def test_usage_is_recorded_with_the_final_totals():
    llm = make_llm(FakeStreamProvider("gemini", ["a", "b"]))
    run(llm)
    (record,) = llm.usage
    assert (record.provider, record.model) == ("gemini", "gemini-m")
    assert (record.input_tokens, record.output_tokens, record.error) == (11, 7, None)
    assert record.latency_s > 0


def test_falls_back_when_the_first_provider_fails_before_streaming():
    primary = FakeStreamProvider("gemini", fail_before=RateLimitError("429"))
    backup = FakeStreamProvider("groq", ["from ", "groq"])
    llm = make_llm(primary, backup)
    events = run(llm)

    assert "".join(e.text for e in events) == "from groq"
    assert primary.calls == 2 and backup.calls == 1  # retried once, then fell back
    assert [r.provider for r in llm.usage] == ["gemini", "gemini", "groq"]


def test_an_error_after_text_started_is_raised_not_retried():
    primary = FakeStreamProvider("gemini", ["part one ", "part two"], fail_after=1)
    backup = FakeStreamProvider("groq", ["never used"])
    llm = make_llm(primary, backup)

    seen = []
    with pytest.raises(ServerError):
        for event in llm.generate_stream("hi"):
            seen.append(event.text)
    assert seen == ["part one "] and backup.calls == 0  # switching mid-answer would contradict it


def test_a_provider_without_streaming_is_skipped():
    class NoStream:
        name = "plain"

    backup = FakeStreamProvider("groq", ["ok"])
    assert "".join(e.text for e in run(make_llm(NoStream(), backup))) == "ok"
