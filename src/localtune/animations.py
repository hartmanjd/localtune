"""The short, colorful scenes localtune plays when a command starts."""

# Just for fun: each command opens with its own scene of about a second, drawn with letters in
# the terminal, before any work starts. "localtune" on its own shows the logo with colors
# running across it, "sweep" a broom sweeping the logo into view, "tune" a knob turning until
# it clicks into place, "doctor" a heartbeat that ends in a check mark and "run" a speedometer
# needle swinging up. The scenes never play while anything is measured, and they are skipped
# when the output isn't a terminal a person is watching (a file, a test, another program), or
# when LOCALTUNE_NO_ANIMATION is set, so they never get in the way. cli.py decides which scene
# each command plays.

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


# The logo: "localtune" in block letters, five rows high. Each frame colors the letters along a
# rainbow that shifts a little further each time, so the colors seem to run across the word.

LETTERS = {
    "l": ["█   ", "█   ", "█   ", "█   ", "████"],
    "o": [" ██ ", "█  █", "█  █", "█  █", " ██ "],
    "c": [" ███", "█   ", "█   ", "█   ", " ███"],
    "a": [" ██ ", "█  █", "████", "█  █", "█  █"],
    "t": ["███", " █ ", " █ ", " █ ", " █ "],
    "u": ["█  █", "█  █", "█  █", "█  █", " ██ "],
    "n": ["█  █", "██ █", "█ ██", "█  █", "█  █"],
    "e": ["████", "█   ", "███ ", "█   ", "████"],
}
LOGO = [" ".join(LETTERS[letter][row] for letter in "localtune") for row in range(5)]
WIDTH = len(LOGO[0])


def rainbow(position: float) -> str:
    """A color around the color wheel, from 0 to 1, as a hex code."""
    red, green, blue = (
        int(127 + 127 * math.sin(2 * math.pi * (position + shift))) for shift in (0, 1 / 3, 2 / 3)
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def logo_frame(shift: float, upto: int = WIDTH, dust: bool = False) -> list[Text]:
    rows = []
    for line in LOGO:
        row = Text()
        for column, char in enumerate(line):
            if column < upto:
                row.append(char, style=rainbow(column / WIDTH + shift) if char != " " else "")
            else:
                row.append("·" if dust and char != " " else " ", style="grey50")
        rows.append(row)
    return rows


def stack(rows: list[Text]) -> Text:
    return Text("\n").join(rows)


def logo_scene() -> list[Text]:
    return [stack(logo_frame(step / 20)) for step in range(20)]


# Sweep: a broom moves left to right over dusty dots, and the logo appears behind it.

BROOM = ["  ║  ", "  ║  ", "  ║  ", " ╱║╲ ", "╱╱║╲╲"]


def sweep_scene() -> list[Text]:
    frames = []
    for upto in range(0, WIDTH + 1, 3):
        rows = []
        for full, handle in zip(logo_frame(upto / WIDTH / 2, upto, dust=True), BROOM, strict=True):
            row = full[:upto]
            row.append(handle, style="bold #c8963e")
            rows.append(row + full[upto + len(handle) :])
        frames.append(stack(rows))
    frames.append(stack(logo_frame(0.5)))
    return frames


# Tune: a knob on a scale swings past its sweet spot, back, and settles there with a click.

SCALE = 31


def knob_frame(position: int, clicked: bool) -> Text:
    row = Text("  ")
    for spot in range(SCALE):
        if spot == position:
            row.append("◉", style="bold #ffd75f" if clicked else "bold #5fafff")
        else:
            row.append("─", style=rainbow(spot / SCALE / 2 + 0.5))
    if clicked:
        row.append("  click! tuned", style="bold #87d787")
    return Text("\n").join([Text("  tuning…", style="bold"), row])


def tune_scene() -> list[Text]:
    sweet = SCALE * 2 // 3
    path = [0, 4, 8, 13, 18, 23, 26, 27, 25, 22, 19, 18, 19, 20, sweet, sweet]
    frames = [knob_frame(min(spot, SCALE - 1), clicked=False) for spot in path]
    return frames + [knob_frame(sweet, clicked=True)] * 4


# Doctor: a heartbeat line runs across the screen and ends in a check mark.

BEAT = "────╮╭──╯╰╮╭─"


def doctor_scene() -> list[Text]:
    line = BEAT * 3
    frames = []
    for length in range(2, len(line) + 1, 2):
        row = Text("  ")
        row.append(line[:length], style="bold #ff5f87")
        frames.append(row)
    done = Text("  ")
    done.append(line, style="bold #ff5f87")
    done.append("  ✓ checking your computer", style="bold #87d787")
    return frames + [done] * 4


# Run: a speedometer needle swings from zero up to full speed.

RADIUS = 6


def gauge_frame(angle: float) -> Text:
    """A half-circle dial, with the needle at angle (pi is zero speed, 0 is full speed)."""
    width, height = RADIUS * 4 + 1, RADIUS + 1
    grid = [[(" ", "")] * width for _ in range(height)]
    for step in range(41):
        a = math.pi * step / 40
        x, y = round(RADIUS * 2 + RADIUS * 2 * math.cos(a)), round(RADIUS - RADIUS * math.sin(a))
        grid[y][x] = ("·", rainbow(0.66 - step / 40 / 3))
    for distance in range(1, RADIUS):
        x = round(RADIUS * 2 + distance * 2 * math.cos(angle))
        y = round(RADIUS - distance * math.sin(angle))
        grid[y][x] = ("●" if distance == RADIUS - 1 else "•", "bold #ffffff")
    grid[RADIUS][RADIUS * 2] = ("◆", "bold #ff8700")
    rows = []
    for line in grid:
        row = Text("  ")
        for char, style in line:
            row.append(char, style=style)
        rows.append(row)
    return stack(rows)


def run_scene() -> list[Text]:
    angles = [math.pi * (1 - (step / 16) ** 0.6) for step in range(17)]
    return [gauge_frame(angle) for angle in angles] + [gauge_frame(0.0)] * 3


SCENES = {
    "logo": logo_scene,
    "sweep": sweep_scene,
    "tune": tune_scene,
    "doctor": doctor_scene,
    "run": run_scene,
}
