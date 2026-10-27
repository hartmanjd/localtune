# Tests for the reading of what else the computer is doing before a run. The CPU and RAM
# numbers come from text copied from a real Linux machine, so they run anywhere; Windows reads
# the same numbers through its own calls, which only work on Windows.

import sys

import pytest

from localtune import background
from localtune.background import Background, busy_reasons, cpu_load_pct

STAT_BEFORE = "cpu  1000 0 500 8000 500 0 0 0 0 0\ncpu0 500 0 250 4000 250 0 0 0 0 0\n"
STAT_AFTER = "cpu  1300 0 600 8500 600 0 0 0 0 0\ncpu0 650 0 300 4250 300 0 0 0 0 0\n"

MEMINFO = """MemTotal:       33554432 kB
MemFree:         1000000 kB
MemAvailable:   25165824 kB
"""


def test_cpu_load_is_the_busy_share_between_two_readings():
    before = background.parse_cpu_times(STAT_BEFORE)
    after = background.parse_cpu_times(STAT_AFTER)
    # 400 busy out of 1,000 ticks, with waiting for the disk counted as idle.
    assert cpu_load_pct(before, after) == 40.0
    assert cpu_load_pct(None, after) is None
    assert cpu_load_pct(after, after) is None


def test_ram_in_use_is_total_minus_available():
    assert background.parse_ram_used_gb(MEMINFO) == 8.0
    assert background.parse_ram_used_gb("MemTotal: 1 kB\n") is None


def test_reading_without_a_gpu_still_reads_the_cpu_and_ram(monkeypatch):
    monkeypatch.setattr(background, "find_nvidia_smi", lambda: None)
    seen = background.read_background(seconds=0.01)
    assert seen.gpu_utilization_pct is None and seen.gpu_power_w is None
    if sys.platform == "linux":
        assert seen.cpu_load_pct is not None and seen.ram_used_gb is not None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_windows_reports_cpu_times_and_ram():
    busy, total = background.windows_cpu_times()
    assert 0 <= busy <= total
    assert background.ram_used_gb() > 0


def test_a_busy_computer_is_noticed():
    assert busy_reasons(Background(gpu_utilization_pct=5, gpu_vram_used_gb=1.0)) == []
    assert busy_reasons(Background(gpu_utilization_pct=45, gpu_vram_used_gb=9.3)) == [
        "the GPU is already 45% busy"
    ]
    assert busy_reasons(Background(gpu_utilization_pct=0, gpu_vram_used_gb=9.3)) == [
        "9.3 GB of the GPU's memory is already in use"
    ]
    assert busy_reasons(Background(cpu_load_pct=60)) == ["the CPU is already 60% busy"]
    assert busy_reasons(None) == []
