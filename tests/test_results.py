# Tests for the saved run record. They build runs from made-up numbers chosen so the energy and
# cost answers are easy to check by hand, and save them into a temporary folder instead of the
# real one in your home directory.

import pytest

from localtune import results
from localtune.cost import ElectricityPrice
from localtune.gpu_monitor import GpuSummary
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import RunResult, RunSettings, Speed

PRICE = ElectricityPrice(per_kwh=0.20, kind="state", source="OH average, EIA 2026-08")

HARDWARE = HardwareInfo(
    os="Linux 6.6",
    is_wsl=True,
    cpu_name="Test CPU",
    cpu_threads=16,
    ram_gb=32.0,
    gpus=[GpuInfo(index=0, name="Test GPU", vram_gb=16.0, driver_version="580.00")],
)

SETTINGS = RunSettings(
    engine="llama.cpp", model="Qwen3-8B", quant="Q4_K_M", kv_cache_k="q8_0", kv_cache_v="q8_0"
)

# 250,000 tokens using 50 Wh: 0.05 kWh, so 0.2 kWh per million tokens and $0.04 per million.
SPEED = Speed(prompt_tokens=200_000, generated_tokens=50_000, prefill_tok_s=3000, decode_tok_s=90)


def gpu_summary(energy_wh: float | None) -> GpuSummary:
    return GpuSummary(
        duration_s=600,
        sample_count=1200,
        avg_power_w=300,
        peak_power_w=320,
        energy_wh=energy_wh,
        baseline_vram_gb=1.0,
        peak_vram_gb=9.0,
        added_vram_gb=8.0,
        avg_utilization_pct=95,
        peak_temperature_c=70,
    )


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))


def make_run(**changes) -> RunResult:
    gpu = gpu_summary(50)
    fields = dict(
        rig_id="test-rig",
        settings=SETTINGS,
        hardware=HARDWARE,
        speed=SPEED,
        gpu=gpu,
        energy=results.energy_and_cost(gpu, SPEED, PRICE),
    )
    return RunResult(**{**fields, **changes})


def test_energy_and_cost_per_million_tokens():
    energy = results.energy_and_cost(gpu_summary(50), SPEED, PRICE)
    assert energy.kwh == pytest.approx(0.05)
    assert energy.kwh_per_million_tokens == pytest.approx(0.2)
    assert energy.cost == pytest.approx(0.01)
    assert energy.cost_per_million_tokens == pytest.approx(0.04)
    assert energy.price == PRICE


def test_busy_time_leaves_out_model_loading():
    # 300 W busy for 60 s = 5 Wh, even though the whole run (loading included) used 50 Wh.
    gpu = gpu_summary(50).model_copy(update={"busy_avg_power_w": 300})
    speed = SPEED.model_copy(update={"busy_s": 60})
    assert results.energy_and_cost(gpu, speed, PRICE).kwh == pytest.approx(0.005)


def test_no_power_reading_keeps_the_price_but_no_cost():
    energy = results.energy_and_cost(gpu_summary(None), SPEED, PRICE)
    assert energy.kwh is None
    assert energy.cost is None
    assert energy.price == PRICE


def test_saved_runs_come_back_the_same():
    first, second = make_run(), make_run(kind="screening")
    results.save_result(first)
    results.save_result(second)
    assert results.load_results() == [first, second]


def test_failed_runs_are_saved_too():
    failed = make_run(
        status="out_of_memory",
        error="CUDA out of memory",
        speed=None,
        gpu=None,
        energy=results.energy_and_cost(None, None, PRICE),
    )
    results.save_result(failed)
    assert results.load_results() == [failed]


def test_a_broken_line_is_skipped():
    run = make_run()
    path = results.save_result(run)
    with path.open("a") as f:
        f.write('{"half written\n')
    assert results.load_results() == [run]


def test_no_results_yet():
    assert results.load_results() == []


def test_each_run_records_its_version_and_time():
    run = make_run()
    assert run.schema_version == results.SCHEMA_VERSION
    assert run.localtune_version
    assert run.created_at.tzinfo is not None
    assert make_run().run_id != run.run_id


def test_rig_id_stays_the_same_on_one_computer():
    assert results.rig_id() == results.rig_id()
