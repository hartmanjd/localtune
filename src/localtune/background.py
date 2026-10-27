"""What else the computer was doing when a benchmark started."""

# Other programs share the computer with the model being measured: a game or a video call on
# the GPU, Ollama keeping a model loaded, a browser busy on the CPU. They take time and memory
# from the model, so the run comes out slower than the computer can really go. Their effect
# can't simply be subtracted from the speed, so instead a reading of them is saved with every
# run (results.py), like a note in a lab notebook: the dataset can then leave busy runs out, or
# study how much a busy computer slows a model down. Just before a run, the readings are:
#   - the GPU: how busy it is, how much memory is already in use, and how much power it draws
#     while nothing of ours runs on it (gpu_monitor.py), which the run's own power can be
#     compared against;
#   - the CPU: how busy it is, which matters most for models split between the GPU and RAM;
#   - the RAM already in use, which a model partly kept in RAM has to fit beside.

import platform
import time
from pathlib import Path

from pydantic import BaseModel

from localtune.gpu_monitor import BUSY_UTILIZATION_PCT, other_use, read_sample
from localtune.hardware import find_nvidia_smi, read_text, windows_memory


class Background(BaseModel):
    gpu_utilization_pct: float | None = None
    gpu_vram_used_gb: float | None = None
    gpu_power_w: float | None = None
    cpu_load_pct: float | None = None
    ram_used_gb: float | None = None


# CPU load is how much of the time between two readings the CPU spent working. Linux keeps the
# running totals in /proc/stat; Windows reports them through its own call. Either way they are
# (busy, total) time counters, and the load is the share of the difference that was busy.


def parse_cpu_times(stat: str) -> tuple[int, int] | None:
    for line in stat.splitlines():
        if line.startswith("cpu "):
            values = [int(value) for value in line.split()[1:9]]
            idle = values[3] + values[4]  # idle and waiting for the disk
            return sum(values) - idle, sum(values)
    return None


def windows_cpu_times() -> tuple[int, int] | None:
    import ctypes

    idle, kernel, user = (ctypes.c_ulonglong() for _ in range(3))
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
    total = kernel.value + user.value  # kernel time includes the idle time
    return total - idle.value, total


def cpu_times() -> tuple[int, int] | None:
    if platform.system() == "Windows":
        return windows_cpu_times()
    return parse_cpu_times(read_text("/proc/stat"))


def cpu_load_pct(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None or after[1] <= before[1]:
        return None
    return round((after[0] - before[0]) / (after[1] - before[1]) * 100, 1)


def parse_ram_used_gb(meminfo: str) -> float | None:
    values = {}
    for line in meminfo.splitlines():
        name, _, rest = line.partition(":")
        if name in ("MemTotal", "MemAvailable"):
            values[name] = int(rest.split()[0])
    if len(values) < 2:
        return None
    return round((values["MemTotal"] - values["MemAvailable"]) / 1024**2, 1)


def ram_used_gb() -> float | None:
    if platform.system() == "Windows":
        memory = windows_memory()
        return round((memory[0] - memory[1]) / 1024**3, 1) if memory else None
    return parse_ram_used_gb(read_text("/proc/meminfo"))


# Taking the readings. The GPU readings take about a second, and the CPU's load is measured over
# that same second.


def read_background(gpu_index: int = 0, seconds: float = 1.0) -> Background:
    nvidia_smi: Path | None = find_nvidia_smi()
    before = cpu_times()
    if nvidia_smi:
        gpu = other_use(lambda: read_sample(nvidia_smi, gpu_index), interval_s=seconds / 2)
    else:
        gpu = None
        time.sleep(seconds)
    return Background(
        gpu_utilization_pct=gpu.utilization_pct if gpu else None,
        gpu_vram_used_gb=gpu.vram_used_gb if gpu else None,
        gpu_power_w=round(gpu.power_w, 1) if gpu and gpu.power_w is not None else None,
        cpu_load_pct=cpu_load_pct(before, cpu_times()),
        ram_used_gb=ram_used_gb(),
    )


# When the computer counts as busy. The desktop alone keeps the GPU under BUSY_UTILIZATION_PCT
# and uses well under OTHER_VRAM_GB of its memory, and leaves the CPU mostly idle.

OTHER_VRAM_GB = 2.0
BUSY_CPU_PCT = 25.0


def busy_reasons(background: Background | None) -> list[str]:
    if background is None:
        return []
    b, reasons = background, []
    if b.gpu_utilization_pct is not None and b.gpu_utilization_pct >= BUSY_UTILIZATION_PCT:
        reasons.append(f"the GPU is already {b.gpu_utilization_pct:.0f}% busy")
    elif b.gpu_vram_used_gb is not None and b.gpu_vram_used_gb >= OTHER_VRAM_GB:
        reasons.append(f"{b.gpu_vram_used_gb:.1f} GB of the GPU's memory is already in use")
    if b.cpu_load_pct is not None and b.cpu_load_pct >= BUSY_CPU_PCT:
        reasons.append(f"the CPU is already {b.cpu_load_pct:.0f}% busy")
    return reasons
