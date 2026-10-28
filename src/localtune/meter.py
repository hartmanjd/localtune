"""A live gauge of the GPU's memory, shown before a benchmark when other programs are using it."""

# Programs a person forgets are running (a browser playing a 4K video, a game in the background,
# Ollama keeping a model loaded) take GPU memory and time from the model being measured, and the
# results come out slower than the computer can really go. Before a benchmark starts, cli.py
# reads what else is going on (background.py); when the GPU is already in use, this gauge shows
# how full its memory is, updated every second, along with the programs NVIDIA says are using
# it. Closing a program makes the bar drop right away, and the benchmark starts as soon as the
# GPU is free, or straight away when Enter is pressed. On Windows, NVIDIA often can't name the
# programs drawing on the screen (browsers, games), so the list can be short or empty while the
# bar still shows the total.

import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path, PureWindowsPath

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from localtune.background import OTHER_VRAM_GB
from localtune.gpu_monitor import BUSY_UTILIZATION_PCT, GpuSample

BAR_WIDTH = 30


def is_busy(sample: GpuSample) -> bool:
    return sample.vram_used_gb >= OTHER_VRAM_GB or sample.utilization_pct >= BUSY_UTILIZATION_PCT


# Drawing the gauge: a bar as wide as the GPU's memory, filled as far as it is in use. It is
# green once there is room again, yellow while other programs hold some of it and red when they
# hold most of it.


def gauge(sample: GpuSample, total_gb: float) -> Text:
    share = min(1.0, sample.vram_used_gb / total_gb) if total_gb else 0.0
    filled = round(share * BAR_WIDTH)
    color = "green" if not is_busy(sample) else "red" if share >= 0.75 else "yellow"
    text = Text("  GPU memory  ")
    text.append("█" * filled, style=color)
    text.append("░" * (BAR_WIDTH - filled), style="dim")
    text.append(f"  {sample.vram_used_gb:.1f} of {total_gb:.0f} GB in use")
    if sample.utilization_pct >= BUSY_UTILIZATION_PCT:
        text.append(f", GPU {sample.utilization_pct:.0f}% busy")
    return text


def screen(sample: GpuSample, total_gb: float, programs: list[str]) -> Group:
    lines = [gauge(sample, total_gb), Text("")]
    if programs:
        lines.append(Text("  Using the GPU:"))
        lines += [Text(f"    • {program}") for program in programs]
        lines.append(Text(""))
    lines += [
        Text("  For the best results, please close any programs you don't need.", style="dim"),
        Text(
            f"  The benchmark will start once GPU usage falls below {BUSY_UTILIZATION_PCT}%\n"
            f"  and other programs are using less than {OTHER_VRAM_GB:.0f} GB of its memory.",
            style="dim",
        ),
        Text(""),
        Text("  Press Enter to start now.", style="bold"),
    ]
    return Group(*lines)


# The programs using the GPU, as nvidia-smi lists them: each one's file name and how much memory
# it holds, biggest first. A program running several times is listed once, with its memory added
# up. Windows often doesn't tell NVIDIA how much memory a game or browser holds; those come last
# and say so, rather than looking like they hold none.


def parse_programs(text: str) -> list[str]:
    memory: dict[str, float | None] = {}
    for line in text.strip().splitlines():
        path, _, used = line.rpartition(",")
        name = PureWindowsPath(path.strip()).name if path.strip() else ""
        if not name:
            continue
        used_gb = int(used.strip()) / 1024 if used.strip().isdigit() else None
        known = [gb for gb in (memory.get(name), used_gb) if gb is not None]
        memory[name] = sum(known) if known else None
    biggest_first = sorted(memory.items(), key=lambda item: (item[1] is None, -(item[1] or 0)))
    return [
        f"{name}  {gb:.1f} GB" if gb is not None else f"{name}  (memory not reported)"
        for name, gb in biggest_first
    ]


def gpu_programs(nvidia_smi: Path, gpu_index: int = 0) -> list[str]:
    try:
        result = subprocess.run(
            [
                str(nvidia_smi),
                f"--id={gpu_index}",
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_programs(result.stdout)


# Whether Enter was pressed, without waiting for it: Windows asks the console for a waiting key;
# elsewhere a typed line shows up as input ready to read.


def enter_pressed() -> bool:
    if sys.platform == "win32":
        import msvcrt

        while msvcrt.kbhit():
            if msvcrt.getwch() in "\r\n":
                return True
        return False
    import select

    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False


# Showing the gauge until the GPU is free ("free") or Enter is pressed ("enter"). A fresh
# reading is taken every second; between readings it only checks for Enter, so pressing it feels
# immediate.


def wait_for_room(
    read: Callable[[], GpuSample | None],
    total_gb: float,
    console: Console,
    programs: list[str],
    pressed: Callable[[], bool] = enter_pressed,
    tick_s: float = 0.25,
    read_every_s: float = 1.0,
) -> str:
    sample = read()
    if sample is None:
        return "enter"
    last_read = time.monotonic()
    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while True:
            if not is_busy(sample):
                return "free"
            if pressed():
                return "enter"
            live.update(screen(sample, total_gb, programs))
            time.sleep(tick_s)
            if time.monotonic() - last_read >= read_every_s:
                sample, last_read = read() or sample, time.monotonic()
