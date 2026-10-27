"""The short, colorful scenes localtune plays when a command starts."""

# Just for fun, drawn in classic ASCII art (slashes, pipes and underscores) in color, a few lines
# tall. "localtune" on its own opens with the logo, colors running across it. While a command
# works, a small loading animation runs beside what it's doing: a broom sweeping the dust for
# "sweep", a knob turning for "tune", a revving speedometer for "run" and a heartbeat for
# "doctor", with a bar filling up when the number of runs left is known. They take a sliver of
# CPU and nothing from the GPU (and every run saves how busy the CPU was, background.py). They
# are skipped when the output isn't a terminal a person is watching (a file, a test, another
# program), or when LOCALTUNE_NO_ANIMATION is set; then a plain spinner shows instead. cli.py
# decides which animation each step shows.

import math
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from rich.console import Console
from rich.live import Live
from rich.text import Text

FPS = 20
OFF_SWITCH = "LOCALTUNE_NO_ANIMATION"


def enabled(console: Console) -> bool:
    return console.is_terminal and not os.environ.get(OFF_SWITCH) and not console.no_color


def play_logo(console: Console) -> None:
    if not enabled(console):
        return
    frames = logo_scene()
    with Live(frames[0], console=console, refresh_per_second=FPS) as live:
        for frame in frames[1:]:
            time.sleep(1 / FPS)
            live.update(frame)


# The logo: "localtune" in small ASCII lettering, four rows high. Each frame colors the letters
# along a rainbow that shifts a little further each time, so the colors seem to run across it.

LOGO = [
    r" _              _ _                 ",
    r"| |___  __ __ _| | |_ _  _ _ _  ___ ",
    r"| / _ \/ _/ _` | |  _| || | ' \/ -_)",
    r"|_\___/\__\__,_|_|\__|\_,_|_||_\___|",
]
WIDTH = len(LOGO[0])
DUST = ".,'`"


def rainbow(position: float) -> str:
    """A color around the color wheel, from 0 to 1, as a hex code."""
    red, green, blue = (
        int(127 + 127 * math.sin(2 * math.pi * (position + shift))) for shift in (0, 1 / 3, 2 / 3)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def logo_frame(shift: float, upto: int = WIDTH, dust: bool = False) -> list[Text]:
    rows = []
    for number, line in enumerate(LOGO):
        row = Text()
        for column, char in enumerate(line):
            if column < upto:
                row.append(char, style=f"bold {rainbow(column / WIDTH + shift)}")
            elif dust and char != " ":
                row.append(DUST[(column * 7 + number * 3) % len(DUST)], style="grey50")
            else:
                row.append(" ")
        rows.append(row)
    return rows


def stack(rows: list[Text]) -> Text:
    return Text("\n").join(rows)


def logo_scene() -> list[Text]:
    return [stack(logo_frame(step / 20)) for step in range(20)]


# The loading animations. Each one draws a frame for how long the step has been running and,
# when it's known, how far along the whole command is (0 to 1), as a few lines with the step's
# label beside them. working() keeps one on screen while the step runs and clears it after.

TRACK = 24


def bar(progress: float | None, width: int = TRACK) -> Text:
    if progress is None:
        return Text()
    filled = round(width * min(max(progress, 0.0), 1.0))
    row = Text("[")
    for column in range(width):
        if column < filled:
            row.append("=", style=f"bold {rainbow(column / width / 2)}")
        else:
            row.append("-", style="grey50")
    return row.append("] ")


# Sweep: a broom sways at the edge of the swept floor, which grows as the runs finish. Behind it
# the floor is clean, ahead of it dusty.

BRISTLES = "/|||\\"


def sweep_loader(seconds: float, progress: float | None) -> list[Text]:
    room = TRACK - len(BRISTLES)
    sway = math.sin(seconds * 5)
    if progress is None:
        at = round(room / 2 + room / 2 * sway)
    else:
        at = min(room, max(0, round(room * progress + 2 * sway)))
    pushing_right = math.cos(seconds * 5) > 0
    handle = Text(" " * (at + 2)).append("\\" if pushing_right else "/", style="bold #d7af5f")
    floor = Text()
    for column in range(at):
        floor.append("_", style=rainbow(column / TRACK / 2))
    floor.append(BRISTLES, style="bold #af875f")
    for column in range(at + len(BRISTLES), TRACK):
        floor.append(DUST[(column * 7) % len(DUST)], style="grey50")
    return [handle, floor.append("  ")]


# Tune: a knob turning back and forth, beside a bar of the runs done.

KNOB_TURNS = "-\\|/|\\"


def tune_loader(seconds: float, progress: float | None) -> list[Text]:
    pointer = KNOB_TURNS[int(seconds * 6) % len(KNOB_TURNS)]
    row = Text("(", style="#5fafff").append(pointer, style="bold #ffd75f").append(") ", "#5fafff")
    return [row + bar(progress)]


# Run: a speedometer needle revving near the top, with speed lines streaming past.

NEEDLE_REVS = "|/|/-/"
STREAKS = "  -=>  -=>  "


def run_loader(seconds: float, progress: float | None) -> list[Text]:
    needle = NEEDLE_REVS[int(seconds * 8) % len(NEEDLE_REVS)]
    shift = int(seconds * 12) % len(STREAKS)
    streaks = (STREAKS[-shift:] + STREAKS[:-shift]) if shift else STREAKS
    row = Text("(", style="#ff8700").append(needle, style="bold #ffffff").append(")", "#ff8700")
    row.append(streaks + " ", style=rainbow(0.6))
    return [row + bar(progress)]


# Doctor: a heartbeat line scrolling past.

BEAT = [
    r"     /\     ",
    r"____/  \  __",
    r"        \/  ",
]


def doctor_loader(seconds: float, progress: float | None) -> list[Text]:
    shift = int(seconds * 16) % len(BEAT[0])
    rows = []
    for line in BEAT:
        looped = (line * 3)[shift : shift + len(line) * 2]
        rows.append(Text(looped, style="bold #ff5f87").append("  "))
    return rows


LOADERS: dict[str, Callable[[float, float | None], list[Text]]] = {
    "sweep": sweep_loader,
    "tune": tune_loader,
    "run": run_loader,
    "doctor": doctor_loader,
}


class Loader:
    def __init__(self, scene: str, label: str, progress: float | None) -> None:
        self.draw, self.label, self.progress = LOADERS[scene], label, progress
        self.started = time.monotonic()

    def frame(self, seconds: float) -> Text:
        rows = self.draw(seconds, self.progress)
        rows[-1 if len(rows) < 3 else 1] = rows[-1 if len(rows) < 3 else 1] + Text(self.label)
        return Text("  ") + Text("\n  ").join(rows)

    def __rich__(self) -> Text:
        return self.frame(time.monotonic() - self.started)


@contextmanager
def working(
    console: Console, scene: str, label: str, progress: float | None = None
) -> Iterator[None]:
    if not enabled(console):
        with console.status(label):
            yield
        return
    with Live(
        Loader(scene, label, progress), console=console, refresh_per_second=FPS, transient=True
    ):
        yield
