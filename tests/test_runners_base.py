# Tests for the measuring every runner shares. The engine is a pretend function that returns
# made-up speeds or fails, and the GPU readings are made up too, so these check how a run
# becomes a saved record: with its energy and cost when it works, and with its error when not.

import itertools

import pytest

from localtune.cost import ElectricityPrice
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.results import RunSettings, Speed
from localtune.runners.base import BackendError, EngineRun, measure

PRICE = ElectricityPrice(per_kwh=0.20, kind="yours", source="your price")
ASKED = RunSettings(engine="test", model="tiny")
USED = ASKED.model_copy(update={"quant": "Q4_0"})


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


def test_a_working_run_is_measured_and_priced():
    # 300 W for 12 s = 1 Wh = 0.001 kWh for 1,000 tokens.
    result = measure(engine(busy_s=12), ASKED, PRICE, repetitions=3, monitor=busy_gpu)
    assert result.status == "ok"
    assert result.settings == USED
    assert result.repetitions == 3
    assert result.gpu.peak_vram_gb == 8
    assert result.energy.kwh == pytest.approx(0.001)
    assert result.energy.kwh_per_million_tokens == pytest.approx(1.0)
    assert result.energy.cost == pytest.approx(0.0002)


def test_a_failed_run_keeps_what_was_asked_and_why():
    def out_of_memory():
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    result = measure(out_of_memory, ASKED, PRICE, repetitions=3, monitor=busy_gpu)
    assert result.status == "out_of_memory"
    assert result.error == "cudaMalloc failed"
    assert result.settings == ASKED
    assert result.speed is None
    assert result.energy.kwh_per_million_tokens is None
