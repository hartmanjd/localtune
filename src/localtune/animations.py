"""The colorful logo and the loading animations localtune shows in the terminal."""

# Just for fun, drawn in classic ASCII art (slashes, pipes and underscores) in color, a few lines
# tall. "localtune" on its own opens with the logo assembling itself in teal and purple. While
# a command works, a small loading animation runs beside what it's doing: a broom sweeping the
# dust for "sweep", a switchboard of sliders and dials for "tune", a little runner for "run" and
# a heartbeat for "doctor", with a bar filling up when the number of runs left is known. They
# take a sliver of CPU and nothing from the GPU (and every run saves how busy the CPU was,
# background.py). They are skipped when the output isn't a terminal a person is watching (a
# file, a test, another program), or when LOCALTUNE_NO_ANIMATION is set; then a plain spinner
# shows instead. cli.py decides which animation each step shows.

import math
import os
import random
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


# The colors, a retro "vaporwave" look: teal blending through blue into deep purple.

COLORS = ["#5fffd7", "#00d7d7", "#00afd7", "#5f87ff", "#875fff", "#8700d7", "#5f00af"]
TEAL, PURPLE, DEEP = COLORS[0], COLORS[4], COLORS[-1]


def color(position: float) -> str:
    """A color along the teal-to-purple blend and back, wrapping around past 1."""
    wave = (math.sin(2 * math.pi * position) + 1) / 2
    return COLORS[min(len(COLORS) - 1, int(wave * len(COLORS)))]


def stack(rows: list[Text]) -> Text:
    return Text("\n").join(rows)


# The logo: "LocalTune" in slanted ASCII lettering, five rows high, about four seconds long. Every
# piece starts as a random dim stroke and snaps into place at its own moment, roughly left to
# right; then a light shimmer sweeps across the finished word and back, splits in the middle
# into two that reach both ends together, and it ends with a small glint at the tip of the "e".

LOGO = [
    r"    __                     ________               ",
    r"   / /   ____  _________ _/ /_  __/_  ______  ___ ",
    "  / /   / __ \\/ ___/ __ `/ / / / / / / / __ \\/ _ \\",
    r" / /___/ /_/ / /__/ /_/ / / / / / /_/ / / / /  __/",
    r"/_____/\____/\___/\__,_/_/ /_/  \__,_/_/ /_/\___/ ",
]
WIDTH = len(LOGO[0])
STROKES = "/\\|_-.'`"
BUILD_FRAMES = 32
SHIMMER_FRAMES = 12
SHINE = 0.3
SHINE_COLOR = "#ffffff"
SHINE_DARK = "#005f5f"
# Each shimmer as (when it starts, after the build; where it starts and ends, from 0 at the left
# edge to 1 at the right, a little past each so it slides fully off): across and back, and
# halfway back it splits, one half running on to the left, the other back to the right, so they
# reach both ends together.
EDGE = SHINE / 2
SHIMMERS = [
    (0, -EDGE, 1 + EDGE),
    (SHIMMER_FRAMES, 1 + EDGE, -EDGE),
    (SHIMMER_FRAMES * 3 // 2, 0.5, 1 + EDGE),
]
SHIMMER_END = SHIMMER_FRAMES * 2 - 1
# The glint, one drawing per moment, as (row, column, character) just off the tip of the "e":
# a point of light that flares into a small cross and fades.
TIP = (1, WIDTH)
GLINT = [
    [(0, 0, "*")],
    [(0, 0, "*"), (-1, 0, "|"), (1, 0, "|"), (0, -1, "-"), (0, 1, "-")],
    [(0, 0, "+")],
    [(0, 0, ".")],
]
GLINT_FRAMES = 3
HOLD_FRAMES = 4


def logo_shade(column: int) -> str:
    """The logo's own color: teal at the left, blending into deep purple at the right."""
    return COLORS[column * len(COLORS) // WIDTH]


def shine_shade(column: int) -> str:
    """The shimmer's color over a column: dark teal over the light teal half and white over the
    deep purple half, so it stands out all the way across."""
    return SHINE_DARK if column * 2 < WIDTH else SHINE_COLOR


def shine(frame: int, column: int) -> bool:
    """Whether a shimmer is passing over this column of the logo at this frame."""
    into = frame - BUILD_FRAMES
    for start, begin, end in SHIMMERS:
        frames = round(abs(end - begin) / (1 + SHINE) * SHIMMER_FRAMES)
        if 0 <= into - start < frames:
            middle = begin + (end - begin) * (into - start + 1) / frames
            if abs(column / WIDTH - middle) < SHINE / 2:
                return True
    return False


def glint(frame: int) -> dict[tuple[int, int], str]:
    into = frame - BUILD_FRAMES - SHIMMER_END
    if not 0 <= into < len(GLINT) * GLINT_FRAMES:
        return {}
    row, column = TIP
    return {(row + up, column + side): char for up, side, char in GLINT[into // GLINT_FRAMES]}


def logo_frame(frame: int, lands: dict[tuple[int, int], int], rng: random.Random) -> Text:
    light = glint(frame)
    rows = []
    for number, line in enumerate(LOGO):
        row = Text()
        for column, char in enumerate(line + "  "):
            land = lands.get((number, column))
            if (number, column) in light:
                spark = light[(number, column)]
                row.append(spark, style="bold #ffffff" if spark in "*+" else f"bold {TEAL}")
            elif land is None or frame < land - 6:
                row.append(" ")
            elif frame < land:
                row.append(rng.choice(STROKES), style=DEEP)
            else:
                shade = shine_shade(column) if shine(frame, column) else logo_shade(column)
                row.append(char, style=f"bold {shade}")
        rows.append(row)
    return stack(rows)


def logo_scene(seed: int | None = None) -> list[Text]:
    rng = random.Random(seed)
    lands = {
        (number, column): round(column / WIDTH * (BUILD_FRAMES - 8)) + rng.randint(2, 8)
        for number, line in enumerate(LOGO)
        for column, char in enumerate(line)
        if char != " "
    }
    frames = BUILD_FRAMES + SHIMMER_END + len(GLINT) * GLINT_FRAMES + HOLD_FRAMES
    return [logo_frame(frame, lands, rng) for frame in range(frames)]


# The loading animations. Each one draws a frame for how long the step has been running and,
# when it's known, how far along the whole command is (0 to 1), as a few short lines with the
# step's label beside the bottom one. working() keeps one on screen while the step runs and
# clears it after.

TRACK = 14
DUST = ".,'`"


def bar(progress: float | None, width: int = 10) -> Text:
    if progress is None:
        return Text()
    filled = round(width * min(max(progress, 0.0), 1.0))
    row = Text("[")
    for column in range(width):
        if column < filled:
            row.append("=", style=f"bold {COLORS[column * len(COLORS) // width]}")
        else:
            row.append("-", style="grey50")
    return row.append("] ")


# Sweep: a broom sways at the edge of the swept floor, which grows as the runs finish. Behind it
# the floor is clean, ahead of it dusty. The handle leans back from the way it's pushing.

BRISTLES = "/|||\\"
HANDLE = 3


def sweep_loader(seconds: float, progress: float | None) -> list[Text]:
    room = TRACK - len(BRISTLES)
    sway = math.sin(seconds * 5)
    if progress is None:
        at = round(room / 2 + room / 2 * sway)
    else:
        at = min(room, max(0, round(room * progress + 2 * sway)))
    pushing_right = math.cos(seconds * 5) > 0
    rows = []
    for height in range(HANDLE):
        lean = HANDLE - 1 - height
        column = at + 2 - lean if pushing_right else at + 2 + lean
        stick = "\\" if pushing_right else "/"
        rows.append(Text(" " * column).append(stick, style=f"bold {PURPLE}"))
    floor = Text()
    for column in range(at):
        floor.append("_", style=COLORS[column * len(COLORS) // TRACK])
    floor.append(BRISTLES, style=f"bold {TEAL}")
    for column in range(at + len(BRISTLES), TRACK):
        floor.append(DUST[(column * 7) % len(DUST)], style="grey50")
    return rows + [floor.append("  ")]


# Tune: a little switchboard, sliders gliding up and down between dials that turn, each at its
# own pace, beside a bar of the runs done.

SLIDER_ROWS = 3
DIAL_TURNS = "-\\|/"
BOARD = "sdsds"


def tune_loader(seconds: float, progress: float | None) -> list[Text]:
    rows = [Text("| ", style=DEEP) for _ in range(SLIDER_ROWS)]
    for number, part in enumerate(BOARD):
        pace = 2 + number * 0.7
        if part == "s":
            level = round((math.sin(seconds * pace + number) + 1) / 2 * (SLIDER_ROWS - 1))
            for height, row in enumerate(rows):
                knob = height == SLIDER_ROWS - 1 - level
                row.append("=" if knob else "|", style=f"bold {TEAL}" if knob else "grey50")
        else:
            pointer = DIAL_TURNS[int(seconds * pace * 2) % len(DIAL_TURNS)]
            for height, row in enumerate(rows):
                if height == 1:
                    row.append("(", style=PURPLE).append(pointer, "bold #ffffff")
                    row.append(")", style=PURPLE)
                else:
                    row.append(" o " if height == 0 else "   ", style=COLORS[2])
        for row in rows:
            row.append(" ")
    for row in rows:
        row.append("|", style=DEEP)
    rows[-1].append("  ").append(bar(progress))
    return rows


# Run: a little runner dashing back and forth, legs and arms swinging.

RUNNER = [
    [" o ", "/|\\", "/ >"],
    [" o ", "<|-", " |\\"],
]
MIRROR = str.maketrans("/\\<>", "\\/><")


def run_loader(seconds: float, progress: float | None) -> list[Text]:
    room = TRACK - 3
    steps = int(seconds * 10)
    lap = steps % (room * 2)
    going_right = lap < room
    at = lap if going_right else room * 2 - lap
    pose = RUNNER[steps % 2]
    if not going_right:
        pose = [part[::-1].translate(MIRROR) for part in pose]
    rows = [
        Text(" " * at).append(part, style=f"bold {TEAL if height == 0 else PURPLE}")
        for height, part in enumerate(pose)
    ]
    rows[-1].append(" " * (room - at) + "  ").append(bar(progress))
    return rows


# Doctor: a heartbeat line scrolling past, its teal and purple drifting along it.

BEAT = [
    r"     /\     ",
    r"____/  \  __",
    r"        \/  ",
]


def doctor_loader(seconds: float, progress: float | None) -> list[Text]:
    shift = int(seconds * 16) % len(BEAT[0])
    rows = []
    for line in BEAT:
        looped = (line * 3)[shift : shift + TRACK + 4]
        row = Text()
        for column, char in enumerate(looped):
            row.append(char, style=f"bold {color(column / TRACK / 2 + seconds / 2)}")
        rows.append(row.append("  "))
    return rows


LOADERS: dict[str, Callable[[float, float | None], list[Text]]] = {
    "sweep": sweep_loader,
    "tune": tune_loader,
    "run": run_loader,
    "doctor": doctor_loader,
}
# Which line the label goes on: the bottom one, except the heartbeat's middle (its baseline).
LABEL_ROW = {"doctor": 1}


class Loader:
    def __init__(self, scene: str, label: str, progress: float | None) -> None:
        self.scene, self.label, self.progress = scene, label, progress
        self.started = time.monotonic()

    def frame(self, seconds: float) -> Text:
        rows = LOADERS[self.scene](seconds, self.progress)
        at = LABEL_ROW.get(self.scene, len(rows) - 1)
        rows[at] = rows[at] + Text(self.label)
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
