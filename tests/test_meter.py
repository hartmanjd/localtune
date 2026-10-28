# Tests for the live gauge of the GPU's memory shown before a benchmark.

import io

from rich.console import Console

from localtune.gpu_monitor import GpuSample
from localtune.meter import BAR_WIDTH, gauge, is_busy, parse_programs, screen, wait_for_room


def reading(vram_gb, utilization=5.0):
    return GpuSample(
        time_s=0.0,
        power_w=30.0,
        vram_used_gb=vram_gb,
        utilization_pct=utilization,
        temperature_c=40.0,
    )


def terminal():
    return Console(file=io.StringIO(), force_terminal=True, width=120)


def test_the_bar_fills_as_far_as_the_memory_is_in_use():
    text = gauge(reading(12.0), 16.0).plain
    assert "█" * round(0.75 * BAR_WIDTH) + "░" in text
    assert "12.0 of 16 GB in use" in text
    assert "busy" not in text
    assert "GPU 85% busy" in gauge(reading(1.0, utilization=85.0), 16.0).plain


def test_the_hint_says_when_the_benchmark_starts():
    console = Console(file=io.StringIO(), width=300)
    console.print(screen(reading(9.0), 16.0, [], 47))
    text = console.file.getvalue()
    assert "once GPU usage falls below 20% and other programs are using less than 2 GB" in text
    assert "on its own in 47 s." in text


def test_the_gpu_counts_as_busy_from_2_gb_or_a_busy_gpu():
    assert not is_busy(reading(0.8))
    assert is_busy(reading(2.5))
    assert is_busy(reading(0.8, utilization=60.0))


def test_programs_are_named_by_their_file_with_memory_when_known():
    text = "C:\\Program Files\\Ollama\\ollama.exe, 9830\n/usr/bin/python3, [N/A]\n"
    assert parse_programs(text) == ["ollama.exe (9.6 GB)", "python3"]
    assert parse_programs("") == []


def test_the_benchmark_starts_once_programs_free_the_memory():
    readings = iter([reading(9.0), reading(6.0), reading(0.9)])
    outcome = wait_for_room(
        lambda: next(readings),
        16.0,
        terminal(),
        ["ollama.exe"],
        pressed=lambda: False,
        tick_s=0,
        read_every_s=0,
    )
    assert outcome == "free"


def test_enter_starts_the_benchmark_anyway():
    outcome = wait_for_room(
        lambda: reading(9.0), 16.0, terminal(), [], pressed=lambda: True, tick_s=0
    )
    assert outcome == "enter"


def test_the_benchmark_starts_by_itself_after_the_wait():
    outcome = wait_for_room(
        lambda: reading(9.0), 16.0, terminal(), [], wait_s=0, pressed=lambda: False, tick_s=0
    )
    assert outcome == "timeout"
