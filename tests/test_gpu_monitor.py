# Tests for the GPU monitor. Like the hardware tests, they use readings typed in by hand instead
# of a real GPU, so they can check the maths (averages, peaks, energy) with numbers whose right
# answer is known, and check that the background watcher starts, collects and stops cleanly.

import itertools
import time

from localtune import gpu_monitor
from localtune.gpu_monitor import GpuMonitor, GpuSample


def sample(time_s, power_w, vram_gb=2.0, utilization=50.0, temperature=60.0):
    return GpuSample(
        time_s=time_s,
        power_w=power_w,
        vram_used_gb=vram_gb,
        utilization_pct=utilization,
        temperature_c=temperature,
    )


def test_parse_sample():
    s = gpu_monitor.parse_sample("285.32, 9216, 98, 71\n", time_s=1.0)
    assert s.power_w == 285.32
    assert s.vram_used_gb == 9.0
    assert s.utilization_pct == 98
    assert s.temperature_c == 71


def test_parse_sample_without_power():
    s = gpu_monitor.parse_sample("[N/A], 1024, 0, 40", time_s=1.0)
    assert s.power_w is None
    assert s.vram_used_gb == 1.0


def test_parse_sample_with_unreadable_output():
    assert gpu_monitor.parse_sample("", time_s=1.0) is None


def test_summarize_energy_and_averages():
    # Two 3-second gaps: averaging 200 W (600 J), then 250 W (750 J). 1350 J = 0.375 Wh.
    samples = [sample(0, 200), sample(3, 200), sample(6, 300, vram_gb=10.0, temperature=75)]
    summary = gpu_monitor.summarize(samples, baseline=sample(-1, 30, vram_gb=1.5))
    assert summary.duration_s == 6
    assert summary.energy_wh == 0.375
    assert summary.peak_power_w == 300
    assert round(summary.avg_power_w, 1) == 233.3
    assert summary.added_vram_gb == 8.5
    assert summary.peak_temperature_c == 75


def test_busy_power_leaves_out_quiet_readings():
    samples = [sample(0, 60, utilization=2), sample(1, 280, utilization=97), sample(2, 300)]
    assert gpu_monitor.summarize(samples).busy_avg_power_w == 290


def test_summarize_without_power_leaves_energy_empty():
    summary = gpu_monitor.summarize([sample(0, None), sample(1, None)])
    assert summary.energy_wh is None
    assert summary.avg_power_w is None


def test_summarize_with_no_samples():
    assert gpu_monitor.summarize([]) is None


def test_monitor_collects_in_the_background():
    clock = itertools.count()
    monitor = GpuMonitor(interval_s=0.01, reader=lambda: sample(next(clock), 250))
    with monitor:
        deadline = time.monotonic() + 5
        while len(monitor.samples) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert monitor.baseline.time_s == 0
    assert monitor.summary().sample_count >= 3


def test_monitor_without_a_gpu():
    with GpuMonitor(interval_s=0.01, reader=lambda: None) as monitor:
        pass
    assert monitor.summary() is None


def test_other_use_averages_a_few_quick_readings():
    readings = iter([sample(0, None, 3.0, 10), sample(1, None, 2.5, 30), None])
    seen = gpu_monitor.other_use(lambda: next(readings), readings=3, interval_s=0)
    assert seen.vram_used_gb == 2.5
    assert seen.utilization_pct == 20


def test_a_busy_or_full_gpu_is_noticed():
    assert gpu_monitor.busy_reason(sample(0, None, 1.0, 5)) is None
    assert gpu_monitor.busy_reason(sample(0, None, 1.0, 45)) == "the GPU is already 45% busy"
    full = gpu_monitor.busy_reason(sample(0, None, 9.3, 0))
    assert full == "9.3 GB of the GPU's memory is already in use"
    assert gpu_monitor.busy_reason(None) is None
