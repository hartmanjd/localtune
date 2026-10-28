# Tests for hardware detection. GitHub's test machines have no GPU and aren't WSL, so these
# tests feed in sample text copied from real machines instead of reading the real hardware.
# They check two promises: the details are read correctly, and missing hardware never crashes.

import sys
from pathlib import Path

import pytest

from localtune import hardware

NVIDIA_SMI_OUTPUT = "0, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 581.57\n"

WSL_PROC_VERSION = "Linux version 6.6.87.2-microsoft-standard-WSL2 (root@host) (gcc) #1 SMP"

CPUINFO = """processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 7 7800X3D 8-Core Processor
"""

MEMINFO = """MemTotal:       33554432 kB
MemFree:         1000000 kB
"""


def test_parse_gpu_csv():
    [gpu] = hardware.parse_gpu_csv(NVIDIA_SMI_OUTPUT)
    assert gpu.name == "NVIDIA GeForce RTX 4070 Ti SUPER"
    assert gpu.vram_gb == 16.0
    assert gpu.driver_version == "581.57"


def test_parse_gpu_csv_with_two_gpus():
    text = NVIDIA_SMI_OUTPUT + "1, NVIDIA GeForce RTX 3060, 12288, 581.57\n"
    assert [gpu.index for gpu in hardware.parse_gpu_csv(text)] == [0, 1]


def test_parse_gpu_csv_with_no_output():
    assert hardware.parse_gpu_csv("") == []


def test_parse_gpu_csv_skips_unreadable_lines():
    text = "0, Some GPU, [N/A], 581.57\n" + NVIDIA_SMI_OUTPUT.replace("0,", "1,", 1)
    assert [gpu.index for gpu in hardware.parse_gpu_csv(text)] == [1]


def test_parse_limits_csv():
    limits = hardware.parse_limits_csv("0, 250.00, 285.00, 3105, 10501, 4, 16\n")
    assert limits[0] == {
        "power_limit_w": 250.0,
        "default_power_limit_w": 285.0,
        "max_sm_clock_mhz": 3105.0,
        "max_memory_clock_mhz": 10501.0,
        "pcie_gen": 4,
        "pcie_lanes": 16,
    }


def test_parse_limits_csv_leaves_unknown_values_empty():
    limits = hardware.parse_limits_csv("0, [N/A], [N/A], 1710, 7001, 3, 1\nnot a gpu line\n")
    assert limits[0]["power_limit_w"] is None
    assert limits[0]["pcie_lanes"] == 1
    assert list(limits) == [0]


def test_gpu_limits_are_added_to_the_right_gpu(monkeypatch):
    [gpu] = hardware.parse_gpu_csv(NVIDIA_SMI_OUTPUT)

    class Result:
        stdout = "0, 250.00, 285.00, 3105, 10501, 4, 16\n"

    monkeypatch.setattr(hardware.subprocess, "run", lambda *args, **kwargs: Result())
    [gpu] = hardware.add_gpu_limits(Path("nvidia-smi"), [gpu])
    assert gpu.power_limit_w == 250
    assert gpu.pcie_gen == 4


def test_gpus_are_kept_when_their_limits_cant_be_read():
    [gpu] = hardware.parse_gpu_csv(NVIDIA_SMI_OUTPUT)
    [kept] = hardware.add_gpu_limits(Path("/does/not/exist/nvidia-smi"), [gpu])
    assert kept == gpu


@pytest.mark.parametrize(
    ("cpu_max", "cpus"),
    [("800000 100000", 8.0), ("150000 100000", 1.5), ("max 100000", None), ("", None)],
)
def test_parse_cpu_quota(cpu_max, cpus):
    assert hardware.parse_cpu_quota(cpu_max) == cpus


@pytest.mark.parametrize(
    ("memory_max", "gb"),
    [("68719476736\n", 64.0), ("max\n", None), ("9223372036854771712", None), ("", None)],
)
def test_parse_memory_limit_gb(memory_max, gb):
    assert hardware.parse_memory_limit_gb(memory_max) == gb


def container_files(monkeypatch, files):
    monkeypatch.setattr(hardware, "read_text", lambda path: files.get(path, ""))


def test_a_container_reports_its_share_of_the_cpu(monkeypatch):
    container_files(monkeypatch, {"/sys/fs/cgroup/cpu.max": "800000 100000"})
    monkeypatch.setattr(
        hardware.os, "sched_getaffinity", lambda pid: set(range(128)), raising=False
    )
    assert hardware.cpu_threads_allowed() == 8


def test_an_older_container_reports_its_share_of_the_cpu(monkeypatch):
    container_files(
        monkeypatch,
        {
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us": "1200000",
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us": "100000",
        },
    )
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    assert hardware.cpu_threads_allowed() == 12


def test_cpus_pinned_to_the_program_count_without_a_quota(monkeypatch):
    container_files(monkeypatch, {"/sys/fs/cgroup/cpu.max": "max 100000"})
    monkeypatch.setattr(hardware.os, "sched_getaffinity", lambda pid: set(range(6)), raising=False)
    assert hardware.cpu_threads_allowed() == 6


def test_a_container_reports_its_share_of_the_ram(monkeypatch):
    container_files(monkeypatch, {"/sys/fs/cgroup/memory.max": "68719476736"})
    assert hardware.ram_allowed_gb(503.7) == 64.0


def test_ram_without_a_limit_is_the_whole_computers(monkeypatch):
    container_files(monkeypatch, {"/sys/fs/cgroup/memory.max": "max"})
    assert hardware.ram_allowed_gb(32.0) == 32.0


def test_is_wsl():
    assert hardware.is_wsl(WSL_PROC_VERSION)
    assert not hardware.is_wsl("Linux version 6.8.0-45-generic (buildd@ubuntu)")


def test_parse_cpu_name():
    assert hardware.parse_cpu_name(CPUINFO) == "AMD Ryzen 7 7800X3D 8-Core Processor"
    assert hardware.parse_cpu_name("") is None


def test_parse_ram_gb():
    assert hardware.parse_ram_gb(MEMINFO) == 32.0
    assert hardware.parse_ram_gb("") is None


def test_find_nvidia_smi_checks_the_wsl_location(monkeypatch, tmp_path):
    fake = tmp_path / "nvidia-smi"
    fake.touch()
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    monkeypatch.setattr(hardware, "WSL_NVIDIA_SMI", fake)
    assert hardware.find_nvidia_smi() == fake


def test_query_nvidia_smi_that_fails_returns_no_gpus():
    assert hardware.query_nvidia_smi(Path("/does/not/exist/nvidia-smi")) == []


def test_detect_hardware_without_a_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: None)
    info = hardware.detect_hardware()
    assert info.gpus == []
    assert info.os


@pytest.mark.skipif(sys.platform != "win32", reason="reads Windows' own RAM and CPU reports")
def test_detect_hardware_on_windows():
    info = hardware.detect_hardware()
    assert info.ram_gb and info.ram_gb > 0
    assert info.cpu_name


@pytest.mark.skipif(sys.platform != "linux", reason="reads Linux's /proc files")
def test_detect_hardware_on_linux():
    info = hardware.detect_hardware()
    assert info.ram_gb and info.ram_gb > 0
    assert info.cpu_name
