# Tests for the logo and the loading animations. They check that every animation draws a few
# small lines with the step's label beside them, that they stay out of the way when nobody is
# watching a terminal or they are switched off, and that they play on a terminal.

import io
import re

import pytest
from rich.console import Console

from localtune import animations
from localtune.animations import LOADERS, LOGO, Loader, enabled, logo_scene, play_logo, working


def terminal() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=100)


def shown(console: Console) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", console.file.getvalue())


def test_the_logo_spells_localtune_in_four_even_rows_and_assembles_itself():
    assert len(LOGO) == 4 and len({len(row) for row in LOGO}) == 1
    frames = logo_scene(seed=1)
    assert 30 <= len(frames) <= 50
    assert frames[0].plain.count(" ") > frames[-1].plain.count(" ")
    assert frames[-1].plain.splitlines() == LOGO


@pytest.mark.parametrize("scene", sorted(LOADERS))
@pytest.mark.parametrize("progress", [None, 0.0, 0.5, 1.0])
def test_every_loader_is_small_and_shows_its_label(scene, progress):
    loader = Loader(scene, "Run 3 of 12", progress)
    for tenth in range(30):
        lines = loader.frame(tenth / 10).plain.splitlines()
        assert len(lines) <= 4
        assert all(len(line) <= 60 for line in lines)
        assert sum("Run 3 of 12" in line for line in lines) == 1


def test_the_bar_fills_as_the_runs_finish():
    assert "[" not in Loader("run", "", None).frame(0).plain
    assert Loader("run", "", 0.5).frame(0).plain.count("=") == 5


def test_animations_only_play_on_a_terminal_and_can_be_switched_off(monkeypatch):
    assert not enabled(Console(file=io.StringIO()))
    assert enabled(terminal())
    monkeypatch.setenv(animations.OFF_SWITCH, "1")
    assert not enabled(terminal())


def test_the_logo_plays_through(monkeypatch):
    monkeypatch.delenv(animations.OFF_SWITCH, raising=False)
    monkeypatch.setattr(animations.time, "sleep", lambda seconds: None)
    console = terminal()
    play_logo(console)
    assert LOGO[3] in shown(console)


def test_a_loader_runs_while_the_work_does(monkeypatch):
    monkeypatch.delenv(animations.OFF_SWITCH, raising=False)
    console = terminal()
    with working(console, "sweep", "Run 1 of 2", 0.5):
        pass
    with working(Console(file=io.StringIO()), "sweep", "Run 1 of 2"):
        pass
