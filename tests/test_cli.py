# Tests for the terminal command. They run the commands the way a person would, but with made-up
# hardware and GPU readings, so they check what gets shown without depending on the machine
# running them.

import itertools

from typer.testing import CliRunner

from localtune import cli
from localtune.doctor import Check
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo

runner = CliRunner()

FAKE_PC = HardwareInfo(
    os="Linux 6.6.87.2-microsoft-standard-WSL2",
    is_wsl=True,
    cpu_name="AMD Ryzen 7 7800X3D 8-Core Processor",
    cpu_threads=16,
    ram_gb=32.0,
    gpus=[
        GpuInfo(
            index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="581.57"
        )
    ],
)


def test_info_shows_the_hardware(monkeypatch):
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC)
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "RTX 4070 Ti SUPER" in result.output
    assert "(WSL)" in result.output
    assert "32.0 GB" in result.output


def test_info_without_a_gpu(monkeypatch):
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": []}))
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "No NVIDIA GPU found" in result.output


def fake_monitor(power_w):
    # A GPU that reports the given power every second of a pretend clock, using 8 GB of VRAM.
    clock = itertools.count()

    def read():
        return GpuSample(
            time_s=next(clock),
            power_w=power_w,
            vram_used_gb=8.0,
            utilization_pct=90,
            temperature_c=65,
        )

    return lambda gpu_index: GpuMonitor(interval_s=0.001, gpu_index=gpu_index, reader=read)


def test_monitor_shows_energy_and_cost(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05", "--price", "0.20"])
    assert result.exit_code == 0
    assert "250 W avg" in result.output
    assert "0.050 USD per hour" in result.output
    assert "your price" in result.output


def test_monitor_uses_the_default_price(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05"])
    assert "US average" in result.output


def test_monitor_when_power_is_not_reported(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=None))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05"])
    assert result.exit_code == 0
    assert "doesn't report power" in result.output


def test_monitor_without_a_gpu(monkeypatch):
    monkeypatch.setattr(
        cli, "GpuMonitor", lambda gpu_index: GpuMonitor(interval_s=0.001, reader=lambda: None)
    )
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.01"])
    assert result.exit_code == 1
    assert "No GPU readings" in result.output


def test_monitor_with_a_zip_code(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05", "--zip", "90210"])
    assert result.exit_code == 0
    assert "zip 90210 (CA)" in result.output


def fake_checks(*statuses):
    return lambda: [
        Check(name=name, status=status, detail="detail", fix="what to do")
        for name, status in zip(("NVIDIA GPU", "llama-bench"), statuses, strict=False)
    ]


def test_doctor_when_ready(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("ok", "ok"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "Ready to benchmark" in result.output


def test_doctor_shows_what_to_fix(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("fail", "ok"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "what to do" in result.output
    assert "fix the items marked" in result.output


def test_doctor_without_an_engine(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("ok", "warn"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "needs llama.cpp or a running local AI server" in result.output
