"""Watch the GPU's power, memory, load and temperature while something runs."""

# While a model runs, a helper in the background keeps asking nvidia-smi for fresh readings.
# When the run ends, the readings are boiled down into one summary: average and peak power,
# the energy used, peak memory and temperature.
#
# Used like this:
#     with GpuMonitor() as monitor:
#         run_the_model()
#     summary = monitor.summary()

import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from localtune.hardware import find_nvidia_smi

# The shape of the data: one reading at a moment in time, and the summary of a whole run.
# The summary is what gets saved with benchmark results later.


class GpuSample(BaseModel):
    time_s: float
    power_w: float | None
    vram_used_gb: float
    utilization_pct: float
    temperature_c: float
    sm_clock_mhz: float | None = None


class GpuSummary(BaseModel):
    duration_s: float
    sample_count: int
    avg_power_w: float | None
    peak_power_w: float | None
    energy_wh: float | None
    busy_avg_power_w: float | None = None
    baseline_vram_gb: float
    peak_vram_gb: float
    added_vram_gb: float
    avg_utilization_pct: float
    peak_temperature_c: float
    busy_avg_sm_clock_mhz: float | None = None


# Taking one reading. nvidia-smi is asked for five numbers about one GPU. Some GPUs don't
# report power or clock speed, which nvidia-smi shows as "[N/A]"; that reading keeps the rest
# and leaves those empty.

SAMPLE_QUERY = "power.draw,memory.used,utilization.gpu,temperature.gpu,clocks.sm"


def parse_sample(text: str, time_s: float) -> GpuSample | None:
    try:
        power, memory_mib, utilization, temperature, sm_clock = (
            part.strip() for part in text.strip().split(",")
        )
        return GpuSample(
            time_s=time_s,
            power_w=None if power.startswith("[") else float(power),
            vram_used_gb=round(float(memory_mib) / 1024, 2),
            utilization_pct=float(utilization),
            temperature_c=float(temperature),
            sm_clock_mhz=None if sm_clock.startswith("[") else float(sm_clock),
        )
    except ValueError:
        return None


def read_sample(nvidia_smi: Path, gpu_index: int = 0) -> GpuSample | None:
    try:
        result = subprocess.run(
            [
                str(nvidia_smi),
                f"--id={gpu_index}",
                f"--query-gpu={SAMPLE_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_sample(result.stdout, time.monotonic())


# Boiling many readings down to one summary. Energy is power added up over time: each gap
# between two readings contributes its average power times its length. The "baseline" is a
# reading taken before the model loads, so the memory the model itself adds can be told apart
# from memory other programs were already using. "Busy" power averages only the readings where
# the GPU was actually working, leaving out quiet moments like loading a model from disk, so
# the energy of the work itself can be told apart from the whole run's (results.py uses it).
# The busy readings' average clock speed shows a GPU slowing itself down when it runs hot or
# hits its power limit, which is common in packed rental servers.

BUSY_UTILIZATION_PCT = 20


def summarize(samples: list[GpuSample], baseline: GpuSample | None = None) -> GpuSummary | None:
    if not samples:
        return None
    baseline = baseline or samples[0]

    powers = [s.power_w for s in samples if s.power_w is not None]
    energy_wh = None
    if len(powers) == len(samples):
        joules = sum(
            (a.power_w + b.power_w) / 2 * (b.time_s - a.time_s)
            for a, b in zip(samples, samples[1:], strict=False)
        )
        energy_wh = joules / 3600

    busy = [
        s.power_w
        for s in samples
        if s.power_w is not None and s.utilization_pct >= BUSY_UTILIZATION_PCT
    ]
    busy_clocks = [
        s.sm_clock_mhz
        for s in samples
        if s.sm_clock_mhz is not None and s.utilization_pct >= BUSY_UTILIZATION_PCT
    ]
    peak_vram = max(s.vram_used_gb for s in samples)
    return GpuSummary(
        duration_s=samples[-1].time_s - samples[0].time_s,
        sample_count=len(samples),
        avg_power_w=sum(powers) / len(powers) if powers else None,
        peak_power_w=max(powers) if powers else None,
        energy_wh=energy_wh,
        busy_avg_power_w=sum(busy) / len(busy) if busy else None,
        baseline_vram_gb=baseline.vram_used_gb,
        peak_vram_gb=peak_vram,
        added_vram_gb=max(0.0, peak_vram - baseline.vram_used_gb),
        avg_utilization_pct=sum(s.utilization_pct for s in samples) / len(samples),
        peak_temperature_c=max(s.temperature_c for s in samples),
        busy_avg_sm_clock_mhz=sum(busy_clocks) / len(busy_clocks) if busy_clocks else None,
    )


# What the GPU is already doing before a benchmark: a few quick readings of how busy it is,
# how much memory other programs hold and how much power it draws. background.py saves them
# with the run and warns when another program is in the way. How busy the GPU is comes from
# the quietest reading: right after a benchmark ends, NVIDIA still reports how busy the GPU
# was over its last moments, so only the first reading is high, while another program keeps
# every reading up.


def other_use(
    reader: Callable[[], GpuSample | None], readings: int = 3, interval_s: float = 0.5
) -> GpuSample | None:
    samples = []
    for number in range(readings):
        sample = reader()
        if sample is not None:
            samples.append(sample)
        if number < readings - 1:
            time.sleep(interval_s)
    if not samples:
        return None
    powers = [s.power_w for s in samples if s.power_w is not None]
    return GpuSample(
        time_s=samples[-1].time_s,
        power_w=sum(powers) / len(powers) if powers else None,
        vram_used_gb=min(s.vram_used_gb for s in samples),
        utilization_pct=min(s.utilization_pct for s in samples),
        temperature_c=max(s.temperature_c for s in samples),
    )


# The background watcher. Starting it takes the baseline reading and launches a helper thread
# that keeps reading until the watched work is done. Without an NVIDIA GPU it simply collects
# nothing, and summary() returns None.


class GpuMonitor:
    def __init__(
        self,
        interval_s: float = 0.5,
        gpu_index: int = 0,
        reader: Callable[[], GpuSample | None] | None = None,
    ):
        self.interval_s = interval_s
        if reader is None:
            nvidia_smi = find_nvidia_smi()
            reader = (lambda: read_sample(nvidia_smi, gpu_index)) if nvidia_smi else lambda: None
        self._read = reader
        self.baseline: GpuSample | None = None
        self.samples: list[GpuSample] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def __enter__(self) -> "GpuMonitor":
        self.baseline = self._read()
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        self._thread.join()

    def _watch(self) -> None:
        while True:
            sample = self._read()
            if sample is not None:
                self.samples.append(sample)
            if self._stop.wait(self.interval_s):
                break

    def summary(self) -> GpuSummary | None:
        return summarize(self.samples, self.baseline)
