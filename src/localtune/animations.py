"""The short, colorful scenes localtune plays when a command starts."""

# Just for fun: each command opens with its own scene of about a second, drawn in classic ASCII
# art (slashes, pipes and underscores) in color, a few lines tall, before any work starts.
# "localtune" on its own shows the logo with colors running across it, "sweep" a broom sweeping
# the logo clean, "tune" a knob turning until it clicks into place, "doctor" a heartbeat that
# ends with an OK and "run" a speedometer needle swinging up. The scenes never play while
# anything is measured, and they are skipped when the output isn't a terminal a person is
# watching (a file, a test, another program), or when LOCALTUNE_NO_ANIMATION is set, so they
# never get in the way. cli.py decides which scene each command plays.

import math
import os
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

FPS = 20
OFF_SWITCH = "LOCALTUNE_NO_ANIMATION"


def enabled(console: Console) -> bool:
    return console.is_terminal and not os.environ.get(OFF_SWITCH) and not console.no_color


def play(console: Console, scene: str) -> None:
    if not enabled(console):
        return
    frames = SCENES[scene]()
    keep = scene == "logo"
    with Live(frames[0], console=console, refresh_per_second=FPS, transient=not keep) as live:
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


# Sweep: a broom moves left to right over the dust, and the logo appears clean behind it.

BROOM = ["  |  ", "  |  ", " /|\\ ", "/|||\\"]


def sweep_scene() -> list[Text]:
    frames = []
    for upto in range(0, WIDTH + 1, 2):
        rows = []
        for full, part in zip(logo_frame(upto / WIDTH / 2, upto, dust=True), BROOM, strict=True):
            row = full[:upto]
            row.append(part, style="bold #af875f" if "|||" in part else "bold #d7af5f")
            rows.append(row + full[upto + len(part) :])
        frames.append(stack(rows))
    frames.append(stack(logo_frame(0.5)))
    return frames


# Tune: a knob turns from low to high, past its sweet spot and back, and clicks into place.

KNOB = [
    r"  .-'''-.  ",
    r" /       \ ",
    r"|    o    |",
    r" '-.___.-' ",
]
# Where the knob's pointer is drawn for each turn, from all the way left to all the way right.
POINTERS = [
    [(2, 3, "-"), (2, 4, "-")],
    [(1, 3, "\\")],
    [(1, 5, "|")],
    [(1, 7, "/")],
    [(2, 6, "-"), (2, 7, "-")],
]


def knob_frame(turn: int, clicked: bool) -> Text:
    grid = [list(line) for line in KNOB]
    for row, column, char in POINTERS[turn]:
        grid[row][column] = char
    pointer = {(row, column) for row, column, _ in POINTERS[turn]}
    rows = []
    for number, line in enumerate(grid):
        row = Text("  ")
        for column, char in enumerate(line):
            if (number, column) in pointer:
                row.append(char, style="bold #ffd75f" if clicked else "bold #ffffff")
            else:
                row.append(char, style=rainbow(0.55 + column / len(line) / 3))
        if number == 1:
            row.append("   tuning...", style="bold")
        if number == 2 and clicked:
            row.append("   *click*  tuned", style="bold #87d787")
        rows.append(row)
    return stack(rows)


def tune_scene() -> list[Text]:
    path = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 3, 3, 2, 2, 3, 3]
    return [knob_frame(turn, clicked=False) for turn in path] + [knob_frame(3, clicked=True)] * 5


# Doctor: a heartbeat line runs across the screen and ends with an OK.

BEAT = [
    r"     /\     ",
    r"____/  \  __",
    r"        \/  ",
]


def doctor_scene() -> list[Text]:
    lines = [part * 3 for part in BEAT]
    frames = []
    for length in range(2, len(lines[0]) + 1, 2):
        rows = []
        for line in lines:
            row = Text("  ")
            row.append(line[:length], style="bold #ff5f87")
            rows.append(row)
        frames.append(stack(rows))
    rows = []
    for number, line in enumerate(lines):
        row = Text("  ")
        row.append(line, style="bold #ff5f87")
        if number == 1:
            row.append("  [ OK ]", style="bold #87d787")
            row.append(" checking your computer", style="bold")
        rows.append(row)
    return frames + [stack(rows)] * 5


# Run: a speedometer needle swings from zero up to full speed while the reading climbs.

GAUGE = [
    r"    _.-----._    ",
    r"  .'         '.  ",
    r" /      o      \ ",
    r" 0           max ",
]
# Where the needle is drawn as it swings, from zero on the left to full speed on the right.
NEEDLES = [
    [(2, 4, "-"), (2, 5, "-"), (2, 6, "-")],
    [(1, 5, "\\")],
    [(1, 8, "|")],
    [(1, 11, "/")],
    [(2, 10, "-"), (2, 11, "-"), (2, 12, "-")],
]
TOP_SPEED = 120


def gauge_frame(swing: int, reading: int) -> Text:
    grid = [list(line) for line in GAUGE]
    for row, column, char in NEEDLES[swing]:
        grid[row][column] = char
    needle = {(row, column) for row, column, _ in NEEDLES[swing]}
    rows = []
    for number, line in enumerate(grid):
        row = Text("  ")
        for column, char in enumerate(line):
            if (number, column) in needle:
                row.append(char, style="bold #ffffff")
            elif char == "o":
                row.append(char, style="bold #ff8700")
            else:
                row.append(char, style=rainbow(0.66 - column / len(line) / 3))
        if number == 2:
            row.append(f"  {reading:>3} tok/s", style="bold")
        rows.append(row)
    return stack(rows)


def run_scene() -> list[Text]:
    steps = 16
    frames = [
        gauge_frame(step * len(NEEDLES) // steps, step * TOP_SPEED // (steps - 1))
        for step in range(steps)
    ]
    return frames + [gauge_frame(len(NEEDLES) - 1, TOP_SPEED)] * 4


SCENES = {
    "logo": logo_scene,
    "sweep": sweep_scene,
    "tune": tune_scene,
    "doctor": doctor_scene,
    "run": run_scene,
}
