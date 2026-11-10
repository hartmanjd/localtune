# Tests for the measuring every runner shares. The engine is a pretend function that returns
# made-up speeds or fails, and the GPU readings are made up too, so these check how a run
# becomes a saved record: with its energy when it works, and with its error when not.

import itertools

import pytest

from localtune.background import Background
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import RunSettings, Speed
from localtune.runners import base
from localtune.runners.base import BackendError, EngineRun, measure

ASKED = RunSettings(engine="test", model="tiny")
USED = ASKED.model_copy(update={"quant": "Q4_0"})
PC_WITH_GPU = HardwareInfo(
    os="Windows 11",
    is_wsl=False,
    cpu_name="CPU",
    cpu_threads=16,
    ram_gb=64.0,
    gpus=[
        GpuInfo(index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="1")
    ],
)


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))


def busy_gpu(gpu_index):
    clock = itertools.count()

    def read():
        return GpuSample(
            time_s=next(clock), power_w=300, vram_used_gb=8, utilization_pct=95, temperature_c=70
        )

    return GpuMonitor(interval_s=0.001, gpu_index=gpu_index, reader=read)


def engine(busy_s):
    speed = Speed(prompt_tokens=900, generated_tokens=100, busy_s=busy_s)
    return lambda: EngineRun(settings=USED, speed=speed)


def test_a_working_run_is_measured():
    # 300 W for 12 s = 1 Wh = 0.001 kWh for 1,000 tokens.
    result = measure(engine(busy_s=12), ASKED, repetitions=3, monitor=busy_gpu)
    assert result.status == "ok"
    assert result.settings == USED
    assert result.repetitions == 3
    assert result.gpu.peak_vram_gb == 8
    assert result.energy.kwh == pytest.approx(0.001)
    assert result.energy.kwh_per_million_tokens == pytest.approx(1.0)


def test_a_failed_run_keeps_what_was_asked_and_why():
    def out_of_memory():
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    result = measure(out_of_memory, ASKED, repetitions=3, monitor=busy_gpu)
    assert result.status == "out_of_memory"
    assert result.error == "cudaMalloc failed"
    assert result.settings == ASKED
    assert result.speed is None
    assert result.energy.kwh_per_million_tokens is None


def test_a_run_keeps_the_facts_of_its_model_and_gpu(monkeypatch):
    monkeypatch.setattr(base, "detect_hardware", lambda: PC_WITH_GPU)
    asked = RunSettings(engine="llama.cpp", model="qwen3:14b")
    result = measure(
        lambda: EngineRun(settings=asked, speed=Speed(prompt_tokens=1, generated_tokens=1)),
        asked,
        repetitions=1,
        monitor=busy_gpu,
    )
    assert result.model_facts.params == 14_768_307_200
    assert result.model_facts.quant == "Q4_K_M"
    assert result.model_facts.file_bytes == 9_276_184_896
    assert result.gpu_facts.name == "RTX 4070 Ti SUPER"
    assert result.gpu_facts.memory_bandwidth_gbs > 0


def test_a_run_keeps_what_else_the_computer_was_doing():
    busy = Background(gpu_utilization_pct=60, cpu_load_pct=30)
    result = measure(engine(busy_s=1), ASKED, repetitions=1, monitor=busy_gpu, background=busy)
    assert result.background == busy
    quiet = measure(engine(busy_s=1), ASKED, repetitions=1, monitor=busy_gpu)
    assert quiet.background.cpu_load_pct == 3


def test_unknown_models_and_gpus_leave_their_facts_empty(monkeypatch):
    monkeypatch.setattr(
        base, "detect_hardware", lambda: PC_WITH_GPU.model_copy(update={"gpus": []})
    )
    result = measure(engine(busy_s=1), ASKED, repetitions=1, monitor=busy_gpu)
    assert result.model_facts is None and result.gpu_facts is None
