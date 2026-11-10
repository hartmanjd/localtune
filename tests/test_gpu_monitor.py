# Tests for the GPU monitor. Like the hardware tests, they use readings typed in by hand instead
# of a real GPU, so they can check the maths (averages, peaks, energy) with numbers whose right
# answer is known, and check that the background watcher starts, collects and stops cleanly.

import itertools
import time

from localtune import gpu_monitor
from localtune.gpu_monitor import GpuMonitor, GpuSample


def sample(time_s, power_w, vram_gb=2.0, utilization=50.0, temperature=60.0, clock=None):
    return GpuSample(
        time_s=time_s,
        power_w=power_w,
        vram_used_gb=vram_gb,
        utilization_pct=utilization,
        temperature_c=temperature,
        sm_clock_mhz=clock,
    )


def test_parse_sample():
    s = gpu_monitor.parse_sample("285.32, 9216, 98, 71, 2610\n", time_s=1.0)
    assert s.power_w == 285.32
    assert s.vram_used_gb == 9.0
    assert s.utilization_pct == 98
    assert s.temperature_c == 71
    assert s.sm_clock_mhz == 2610


def test_parse_sample_without_power():
    s = gpu_monitor.parse_sample("[N/A], 1024, 0, 40, [N/A]", time_s=1.0)
    assert s.power_w is None
    assert s.sm_clock_mhz is None
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


def test_busy_clock_leaves_out_quiet_readings():
    samples = [
        sample(0, 60, utilization=2, clock=210),
        sample(1, 280, utilization=97, clock=2600),
        sample(2, 300, clock=2400),
    ]
    assert gpu_monitor.summarize(samples).busy_avg_sm_clock_mhz == 2500


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
    assert monitor.baseline[0].time_s == 0
    assert monitor.summary().sample_count >= 3


def test_monitor_without_a_gpu():
    with GpuMonitor(interval_s=0.01, reader=lambda: None) as monitor:
        pass
    assert monitor.summary() is None


def test_other_use_takes_a_few_quick_readings():
    readings = iter([sample(0, 20.0, 3.0, 10), sample(1, 30.0, 2.5, 30), None])
    seen = gpu_monitor.other_use(lambda: next(readings), readings=3, interval_s=0)
    assert seen.vram_used_gb == 2.5
    assert seen.utilization_pct == 10
    assert seen.power_w == 25


def test_the_end_of_the_last_benchmark_doesnt_count_as_busy():
    # Right after a run, only the first reading still shows it; the GPU is idle after that.
    readings = iter(
        [sample(0, 300.0, 0.0, 100), sample(1, 270.0, 0.0, 0), sample(2, 250.0, 0.0, 0)]
    )
    seen = gpu_monitor.other_use(lambda: next(readings), readings=3, interval_s=0)
    assert seen.utilization_pct == 0


# A computer with two GPUs. Readings are typed in the way nvidia-smi answers for every card at
# once: one line per card, starting with its number.


def test_every_card_is_read_from_one_answer():
    cards = gpu_monitor.parse_cards("0, 120.5, 8192, 45, 60, 1300\n1, 110, 7168, 40, 55, 1300\n", 1)
    assert sorted(cards) == [0, 1]
    assert cards[1].vram_used_gb == 7.0


def two_cards(second_card_gb):
    clock = itertools.count()

    def read():
        t = next(clock)
        loaded = t > 0  # the first reading is taken before the model loads
        return {
            0: sample(t, 100.0, vram_gb=8.0 if loaded else 0.5, temperature=60),
            1: sample(t, 80.0, vram_gb=second_card_gb if loaded else 0.5, temperature=70),
        }

    return read


def watch(read):
    monitor = GpuMonitor(interval_s=0.01, cards_reader=read)
    with monitor:
        deadline = time.monotonic() + 5
        while len(monitor.readings) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    return monitor.summary()


def test_a_model_spread_over_two_cards_adds_them_up():
    summary = watch(two_cards(second_card_gb=7.5))
    assert summary.cards == [0, 1]
    assert summary.card_peak_vram_gb == [8.0, 7.5]
    assert summary.peak_vram_gb == 15.5
    assert summary.baseline_vram_gb == 1.0
    assert summary.avg_power_w == 180
    assert summary.peak_temperature_c == 70


def test_a_card_the_model_didnt_touch_is_left_out():
    summary = watch(two_cards(second_card_gb=0.5))
    assert summary.cards == [0]
    assert summary.peak_vram_gb == 8.0
    assert summary.avg_power_w == 100
