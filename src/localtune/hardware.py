"""Detect the computer localtune is running on: OS, CPU, RAM and NVIDIA GPUs."""

# Every benchmark result is only meaningful next to the hardware that produced it, and later the
# prediction models learn from exactly these details. This module gathers them. It only reads,
# never changes anything, and when something can't be found (no NVIDIA card, an unusual OS) it
# leaves that field empty instead of crashing, so localtune still works on any machine.

import os
import platform
import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel

# The shape of the hardware information. Other parts of localtune (the terminal command now,
# saved benchmark runs later) rely on these exact fields, so they are defined once, here.


class GpuInfo(BaseModel):
    index: int
    name: str
    vram_gb: float
    driver_version: str


class HardwareInfo(BaseModel):
    os: str
    is_wsl: bool
    cpu_name: str | None
    cpu_threads: int | None
    ram_gb: float | None
    gpus: list[GpuInfo]


# NVIDIA GPUs are read through nvidia-smi, the tool that ships with NVIDIA's driver. On WSL it
# isn't always on the PATH, so the place Windows puts it inside WSL is checked as well.

WSL_NVIDIA_SMI = Path("/usr/lib/wsl/lib/nvidia-smi")

GPU_QUERY = "index,name,memory.total,driver_version"


def find_nvidia_smi() -> Path | None:
    found = shutil.which("nvidia-smi")
    if found:
        return Path(found)
    if WSL_NVIDIA_SMI.exists():
        return WSL_NVIDIA_SMI
    return None


def query_nvidia_smi(nvidia_smi: Path) -> list[GpuInfo]:
    try:
        result = subprocess.run(
            [str(nvidia_smi), f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gpu_csv(result.stdout)


def parse_gpu_csv(text: str) -> list[GpuInfo]:
    gpus = []
    for line in text.strip().splitlines():
        try:
            index, name, memory_mib, driver = (part.strip() for part in line.split(","))
            gpus.append(
                GpuInfo(
                    index=int(index),
                    name=name,
                    vram_gb=round(int(memory_mib) / 1024, 1),
                    driver_version=driver,
                )
            )
        except ValueError:
            continue
    return gpus


# On Linux (and WSL), CPU, RAM and WSL details come from files Linux keeps under /proc. Each
# helper takes that file's text, so the tests can feed in examples from machines they don't
# run on.


def is_wsl(proc_version: str) -> bool:
    return "microsoft" in proc_version.lower()


def parse_cpu_name(cpuinfo: str) -> str | None:
    for line in cpuinfo.splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def parse_ram_gb(meminfo: str) -> float | None:
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return round(kib / 1024 / 1024, 1)
    return None


def read_text(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


# Windows has no /proc. It reports RAM through its own memory status call, and keeps the CPU's
# full name in the registry. Both only exist on Windows, so they are only looked up there.


def windows_ram_gb() -> float | None:
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_physical", ctypes.c_ulonglong),
            ("available_physical", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("available_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("available_extended_virtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus(length=ctypes.sizeof(MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return round(status.total_physical / 1024**3, 1)


def windows_cpu_name() -> str | None:
    import winreg

    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip() or None
    except OSError:
        return None


# Puts all the pieces together into one HardwareInfo.


def detect_hardware() -> HardwareInfo:
    nvidia_smi = find_nvidia_smi()
    if platform.system() == "Windows":
        cpu_name, ram_gb = windows_cpu_name(), windows_ram_gb()
    else:
        cpu_name = parse_cpu_name(read_text("/proc/cpuinfo"))
        ram_gb = parse_ram_gb(read_text("/proc/meminfo"))
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}",
        is_wsl=is_wsl(read_text("/proc/version")),
        cpu_name=cpu_name or platform.processor() or None,
        cpu_threads=os.cpu_count(),
        ram_gb=ram_gb,
        gpus=query_nvidia_smi(nvidia_smi) if nvidia_smi else [],
    )
