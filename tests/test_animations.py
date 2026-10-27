# Tests for the opening animations. They check that every scene has frames to show, that the
# scenes stay out of the way when nobody is watching a terminal or they are switched off, and
# that a scene plays through on a terminal (without waiting between frames).

import io

import pytest
from rich.console import Console

from localtune import animations
from localtune.animations import LOGO, SCENES, enabled, play


def terminal() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=100)


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_every_scene_has_frames_that_fit_a_terminal(scene):
    frames = SCENES[scene]()
    assert len(frames) >= 10
    assert all(len(line) <= 80 for frame in frames for line in frame.plain.splitlines())


def test_the_logo_spells_localtune_in_five_rows():
    assert len(LOGO) == 5 and len({len(row) for row in LOGO}) == 1


def test_scenes_only_play_on_a_terminal_and_can_be_switched_off(monkeypatch):
    assert not enabled(Console(file=io.StringIO()))
    assert enabled(terminal())
    monkeypatch.setenv(animations.OFF_SWITCH, "1")
    assert not enabled(terminal())


def test_a_scene_plays_through(monkeypatch):
    monkeypatch.delenv(animations.OFF_SWITCH, raising=False)
    monkeypatch.setattr(animations.time, "sleep", lambda seconds: None)
    console = terminal()
    play(console, "logo")
    assert "█" in console.file.getvalue()
