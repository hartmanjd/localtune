# Tests for the time left "localtune measure", "tune" and "sweep" show. They use the real model and
# GPU tables, with runs saved on this computer (the rig_id in LOCALTUNE_HOME) and on another.

import pytest

from localtune import eta
from localtune.gpu_monitor import GpuSummary
from localtune.gpus import find_gpu_spec
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import RunResult, RunSettings, Speed, energy_used, model_facts, rig_id

GPU = find_gpu_spec("RTX 4070 Ti SUPER", 16)
PC = HardwareInfo(
    os="Windows-11",
    is_wsl=False,
    cpu_name=None,
    cpu_threads=16,
    ram_gb=32.0,
    gpus=[
        GpuInfo(
            index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="581.57"
        )
    ],
)


@pytest.fixture(autouse=True)
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))


def saved_run(model: str, took_s: float, rig: str | None = None, **settings) -> RunResult:
    return RunResult(
        rig_id=rig or rig_id(),
        repetitions=3,
        settings=RunSettings(engine="llama.cpp", model=model, **settings),
        hardware=PC,
        model_facts=model_facts(model),
        gpu_facts=GPU,
        speed=Speed(prompt_tokens=3 * 512, generated_tokens=3 * 128, decode_tok_s=60),
        gpu=GpuSummary(
            duration_s=took_s,
            sample_count=1,
            avg_power_w=None,
            peak_power_w=None,
            energy_wh=None,
            baseline_vram_gb=0,
            peak_vram_gb=0,
            added_vram_gb=0,
            avg_utilization_pct=0,
            peak_temperature_c=0,
        ),
        energy=energy_used(None, None),
    )


def guess(runs, model="llama3.1:8b", **placement):
    return eta.guess_run(runs, model, PC, 0, 512, 128, 3, **placement)


def test_a_first_run_gets_a_rough_guess_from_the_formula():
    first = guess([])
    assert first.rough and first.seconds == pytest.approx(
        eta.raw_seconds(model_facts("llama3.1:8b"), GPU, PC, 512, 128, 3)
    )
    assert guess([], "not-a-model:7b") is None


def test_layers_kept_in_ram_are_guessed_much_slower():
    # --gpu-layers 8 of llama3.1:8b's 32 layers keeps three quarters of it in RAM.
    assert guess([], gpu_layers=8).seconds > 5 * guess([]).seconds


def test_the_guess_learns_from_this_computers_runs():
    raw = guess([]).seconds
    twice = saved_run("llama3.1:8b", 2 * raw)
    learned = guess([twice])
    assert not learned.rough and learned.seconds == pytest.approx(2 * raw)
    # The same pace carries over to a different placement of the same model.
    assert guess([twice], gpu_layers=8).seconds == pytest.approx(
        2 * guess([], gpu_layers=8).seconds
    )


def test_runs_of_the_same_model_count_first_and_other_computers_not_at_all():
    raw = guess([]).seconds
    other_model = saved_run("qwen3:14b", 3 * eta.raw_seconds_for(saved_run("qwen3:14b", 1)))
    elsewhere = saved_run("llama3.1:8b", 10 * raw, rig="another-computer")
    assert guess([other_model, elsewhere]).seconds == pytest.approx(3 * raw)
    same = saved_run("llama3.1:8b", 2 * raw)
    assert guess([other_model, same]).seconds == pytest.approx(2 * raw)


def test_a_sweep_learns_how_long_its_sessions_take():
    # Guessed 100 s, 100 s, 1,000 s (a model spilling into RAM) and one the tables don't know.
    clock = eta.SweepClock([100, 100, 1000, None])
    assert clock.rough and clock.session(1) == 100 and clock.after(0) is None
    clock.took.append(200)  # this computer took twice the guess
    assert not clock.rough
    assert clock.session(1) == 200
    assert clock.session(2) == 2000
    assert clock.session(3) == 200  # the average of the finished sessions
    assert clock.after(0) == 200 + 2000 + 200
    assert clock.left(1) == 200 + 2000 + 200


def test_a_sweep_starts_from_this_computers_pace_and_ignores_failed_runs():
    clock = eta.SweepClock([100, 100, 1000], prior=1.5)
    assert not clock.rough and clock.left(0) == 1800
    clock.took.append(None)  # ran out of memory after a few seconds
    assert clock.session(1) == 150
