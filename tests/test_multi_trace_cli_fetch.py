"""Trace thread safety, the `multi` command and the fetch fixes."""

import threading
from pathlib import Path

import httpx
import pytest

from fakes import FakeClock, TeamLLM, team_tools
from reporter import cli
from reporter.tools import FetchPage, ToolError
from reporter.trace import StepRecord, Trace


def test_trace_gives_unique_indexes_to_concurrent_steps():
    trace = Trace(run_id="r", system="multi", topic="T", started_at="now")

    def add():
        for _ in range(200):
            trace.add_step(StepRecord(index=0, kind="think", worker="s1"))

    threads = [threading.Thread(target=add) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(s.index for s in trace.steps) == list(range(1, 1601))


def test_multi_flags_are_parsed():
    args = cli.build_parser().parse_args(
        ["multi", "Topic", "--workers", "2", "--llm-slots", "1", "--max-cost", "0.1"]
    )
    assert (args.command, args.workers, args.llm_slots, args.max_cost) == ("multi", 2, 1, 0.1)


def test_multi_command_prints_the_digest_and_saves_the_trace(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("reporter.llm.build_default_llm", lambda: TeamLLM(3))
    monkeypatch.setattr("reporter.tools.default_tools", team_tools)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["multi", "Topic", "--out", "out/report.md"])

    captured = capsys.readouterr()
    assert code == 0
    assert "## Part 1" in captured.out and "## Sources" in captured.out
    assert "sections 3/3 complete" in captured.err
    assert Path("out/report.md").read_text(encoding="utf-8").startswith("# Topic")
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1


# ------------------------------------------------------------------------ fetch


class CountingStream(httpx.SyncByteStream):
    def __init__(self, chunks, on_chunk=None):
        self.chunks = chunks
        self.read = 0
        self.on_chunk = on_chunk

    def __iter__(self):
        for chunk in self.chunks:
            self.read += 1
            if self.on_chunk:
                self.on_chunk()
            yield chunk


def fetcher(stream, content_type, **kw) -> FetchPage:
    def handler(request):
        return httpx.Response(200, headers={"content-type": content_type}, stream=stream)

    client = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)
    return FetchPage(client, **kw)


def test_fetch_refuses_a_pdf_from_the_headers_without_reading_the_body():
    stream = CountingStream([b"%PDF" * 1000] * 50)
    tool = fetcher(stream, "application/pdf")
    with pytest.raises(ToolError, match="not text or HTML"):
        tool.run(tool.input_model(url="https://x.example/a.pdf"))
    assert stream.read == 0


def test_fetch_stops_a_slow_download_at_the_deadline():
    clock = FakeClock()
    stream = CountingStream([b"readable words " * 40] * 20, on_chunk=lambda: clock.sleep(5))
    tool = fetcher(stream, "text/plain", max_seconds=12, clock=clock)

    out = tool.run(tool.input_model(url="https://x.example/slow"))

    assert stream.read == 3  # 5 s, 10 s, 15 s: the third chunk passes the 12 s deadline
    assert out.pages and "readable words" in out.pages[0].text
