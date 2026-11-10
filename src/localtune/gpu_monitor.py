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
    cards: list[int] = []
    card_peak_vram_gb: list[float] = []


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


# A computer can have several GPUs, and llama.cpp and Ollama spread a model over them by
# themselves. So every card is read at once (one line of nvidia-smi's answer each), and after
# the run the cards whose memory grew are the ones it used. Their readings are then added
# together into one, as if they were a single bigger GPU: power and memory add up, while load,
# temperature and clock speed are those of the busiest, hottest and slowest card.

USED_CARD_GB = 0.3


def parse_cards(text: str, time_s: float) -> dict[int, GpuSample]:
    cards = {}
    for line in text.strip().splitlines():
        index, _, rest = line.partition(",")
        sample = parse_sample(rest, time_s)
        if sample is not None and index.strip().isdigit():
            cards[int(index)] = sample
    return cards


def read_cards(nvidia_smi: Path) -> dict[int, GpuSample]:
    try:
        result = subprocess.run(
            [
                str(nvidia_smi),
                f"--query-gpu=index,{SAMPLE_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    return parse_cards(result.stdout, time.monotonic())


def together(samples: list[GpuSample]) -> GpuSample:
    powers = [s.power_w for s in samples]
    clocks = [s.sm_clock_mhz for s in samples if s.sm_clock_mhz is not None]
    return GpuSample(
        time_s=samples[0].time_s,
        power_w=None if None in powers else sum(powers),
        vram_used_gb=round(sum(s.vram_used_gb for s in samples), 2),
        utilization_pct=max(s.utilization_pct for s in samples),
        temperature_c=max(s.temperature_c for s in samples),
        sm_clock_mhz=min(clocks) if clocks else None,
    )


def cards_used(
    readings: list[dict[int, GpuSample]], baseline: dict[int, GpuSample], main: int
) -> list[int]:
    grew = [
        card
        for card, before in sorted(baseline.items())
        if max((r[card].vram_used_gb for r in readings if card in r), default=0)
        >= before.vram_used_gb + USED_CARD_GB
    ]
    return grew or [main]


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
        cards_reader: Callable[[], dict[int, GpuSample]] | None = None,
    ):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        if reader is not None:
            cards_reader = lambda: {gpu_index: s} if (s := reader()) is not None else {}  # noqa: E731
        elif cards_reader is None:
            nvidia_smi = find_nvidia_smi()
            cards_reader = (lambda: read_cards(nvidia_smi)) if nvidia_smi else dict
        self._read = cards_reader
        self.baseline: dict[int, GpuSample] = {}
        self.readings: list[dict[int, GpuSample]] = []
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
            reading = self._read()
            if reading:
                self.readings.append(reading)
            if self._stop.wait(self.interval_s):
                break

    @property
    def samples(self) -> list[GpuSample]:
        return [r[self.gpu_index] for r in self.readings if self.gpu_index in r]

    def summary(self) -> GpuSummary | None:
        used = cards_used(self.readings, self.baseline, self.gpu_index)
        whole = [r for r in self.readings if all(card in r for card in used)]
        before = self.baseline if all(card in self.baseline for card in used) else None
        summary = summarize(
            [together([r[card] for card in used]) for r in whole],
            together([before[card] for card in used]) if before else None,
        )
        if summary is None:
            return None
        peaks = [max(r[card].vram_used_gb for r in whole) for card in used]
        return summary.model_copy(update={"cards": used, "card_peak_vram_gb": peaks})
