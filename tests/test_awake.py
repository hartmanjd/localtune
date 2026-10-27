# Tests for keeping the computer awake during benchmarks. Off Windows nothing is changed; the
# terminal command (cli.py) holds it for the benchmark commands only, until they end.

import sys
from contextlib import contextmanager

from typer.testing import CliRunner

from localtune import cli
from localtune.awake import keep_awake

runner = CliRunner()


def test_does_nothing_outside_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with keep_awake():
        pass


def held_awake(monkeypatch) -> list[str]:
    events = []

    @contextmanager
    def fake_keep_awake():
        events.append("held")
        try:
            yield
        finally:
            events.append("released")

    monkeypatch.setattr(cli, "keep_awake", fake_keep_awake)
    return events


def test_benchmark_commands_keep_the_computer_awake_until_they_end(monkeypatch):
    events = held_awake(monkeypatch)
    monkeypatch.setattr(cli, "find_tool", lambda name: None)
    runner.invoke(cli.app, ["sweep", "qwen3:14b"])
    assert events == ["held", "released"]


def test_other_commands_leave_sleep_alone(monkeypatch):
    events = held_awake(monkeypatch)
    runner.invoke(cli.app, ["results"])
    assert events == []
