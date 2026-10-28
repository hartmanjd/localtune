"""A live gauge of the GPU's memory, shown before a benchmark when other programs are using it."""

# Programs a person forgets are running (a browser playing a 4K video, a game in the background,
# Ollama keeping a model loaded) take GPU memory and time from the model being measured, and the
# results come out slower than the computer can really go. Before a benchmark starts, cli.py
# reads what else is going on (background.py); when the GPU is already in use, this gauge shows
# how full its memory is, updated every second, along with the programs NVIDIA says are using
# it. Closing a program makes the bar drop right away, and the benchmark starts as soon as the
# GPU is free, when Enter is pressed, or after a minute, so a sweep left running overnight still
# carries on. On Windows, NVIDIA often can't name the programs drawing on the screen (browsers,
# games), so the list can be short or empty while the bar still shows the total.

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
WAIT_S = 60


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


def screen(sample: GpuSample, total_gb: float, programs: list[str], seconds_left: float) -> Group:
    lines = [gauge(sample, total_gb)]
    if programs:
        lines.append(Text(f"  Using the GPU: {', '.join(programs)}"))
    lines.append(
        Text(
            "  Close the programs you don't need and watch the bar drop. The benchmark starts "
            f"when the GPU is free, when you press Enter, or in {seconds_left:.0f} s.",
            style="dim",
        )
    )
    return Group(*lines)


# The programs using the GPU, as nvidia-smi lists them: each one's file name, and how much
# memory it holds when NVIDIA knows.


def parse_programs(text: str) -> list[str]:
    programs = []
    for line in text.strip().splitlines():
        path, _, memory = line.rpartition(",")
        name = PureWindowsPath(path.strip()).name if path.strip() else ""
        if not name:
            continue
        memory = memory.strip()
        programs.append(f"{name} ({int(memory) / 1024:.1f} GB)" if memory.isdigit() else name)
    return programs


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


# Showing the gauge until the GPU is free ("free"), Enter is pressed ("enter") or the wait is
# over ("timeout"). A fresh reading is taken every second; between readings it only checks for
# Enter, so pressing it feels immediate.


def wait_for_room(
    read: Callable[[], GpuSample | None],
    total_gb: float,
    console: Console,
    programs: list[str],
    wait_s: float = WAIT_S,
    pressed: Callable[[], bool] = enter_pressed,
    tick_s: float = 0.25,
    read_every_s: float = 1.0,
) -> str:
    sample = read()
    if sample is None:
        return "timeout"
    started = last_read = time.monotonic()
    with Live(console=console, refresh_per_second=4, transient=True) as live:
        while True:
            if not is_busy(sample):
                return "free"
            if pressed():
                return "enter"
            left = wait_s - (time.monotonic() - started)
            if left <= 0:
                return "timeout"
            live.update(screen(sample, total_gb, programs, left))
            time.sleep(tick_s)
            if time.monotonic() - last_read >= read_every_s:
                sample, last_read = read() or sample, time.monotonic()
