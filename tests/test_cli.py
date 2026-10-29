import os

import pytest

from fakes import FakeLLM, FakeTool, action, page, plan
from reporter import cli


def test_load_env_file_sets_missing_variables_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nFOO_KEY="abc"\nBAR_KEY=keep-me\n\nnot a pair\n', encoding="utf-8")
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.setenv("BAR_KEY", "already-set")
    cli.load_env_file(env)
    assert os.environ["FOO_KEY"] == "abc"
    assert os.environ["BAR_KEY"] == "already-set"
    monkeypatch.delenv("FOO_KEY")


def test_load_env_file_ignores_missing_file(tmp_path):
    cli.load_env_file(tmp_path / "absent.env")


def test_parser_requires_a_topic():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["baseline"])


def test_baseline_command_prints_report_and_saves_trace(tmp_path, monkeypatch, capsys):
    llm = FakeLLM(
        [
            plan("Q?"),
            action("fetch_page", [1], url=page(1).url),
            action("finish"),
            action("finish"),
            action("finish"),
            "# Topic\n\nA fact [1].",
        ]
    )
    tools = {"fetch_page": FakeTool("fetch_page", {page(1).url: page(1)})}
    monkeypatch.setattr("reporter.llm.build_default_llm", lambda: llm)
    monkeypatch.setattr("reporter.tools.default_tools", lambda: tools)
    monkeypatch.chdir(tmp_path)

    code = cli.main(["baseline", "Topic", "--out", "out/report.md", "--max-steps", "6"])

    captured = capsys.readouterr()
    assert code == 0
    assert "A fact [1]" in captured.out and "## Sources" in captured.out
    assert (tmp_path / "out" / "report.md").read_text(encoding="utf-8").startswith("# Topic")
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
    assert "stopped: model_finished" in captured.err


def test_missing_api_keys_give_a_clear_error(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(os.path.dirname(__file__))  # no .env here
    assert cli.main(["baseline", "Topic"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err
