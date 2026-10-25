# Tests for hardware detection. GitHub's test machines have no GPU and aren't WSL, so these
# tests feed in sample text copied from real machines instead of reading the real hardware.
# They check two promises: the details are read correctly, and missing hardware never crashes.

from pathlib import Path

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
