"""What every runner shares: the error it raises, what it reports, and the measuring around it."""

# A runner only knows how to drive one engine and read its numbers. Everything else about a
# benchmark is the same for every engine and lives here: the GPU is watched while the engine
# runs (gpu_monitor.py), the energy is priced (cost.py), the computer is described
# (hardware.py), the model's and GPU's facts are looked up, and it all becomes one RunResult
# (results.py). A run that fails becomes a record too, with what went wrong.

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from localtune.cost import ElectricityPrice
from localtune.gpu_monitor import GpuMonitor
from localtune.hardware import detect_hardware
from localtune.results import (
    RunResult,
    RunSettings,
    Speed,
    energy_and_cost,
    gpu_facts,
    model_facts,
    rig_id,
)


class BackendError(Exception):
    def __init__(self, message: str, status: Literal["out_of_memory", "crashed"] = "crashed"):
        super().__init__(message)
        self.status = status


# What a runner hands back: the settings the engine actually used (which can differ from what
# was asked, for example a default filled in) and the speeds it measured.


class EngineRun(BaseModel):
    settings: RunSettings
    speed: Speed


def measure(
    run_engine: Callable[[], EngineRun],
    requested: RunSettings,
    price: ElectricityPrice,
    repetitions: int,
    gpu_index: int = 0,
    monitor: Callable[[int], GpuMonitor] = lambda gpu_index: GpuMonitor(gpu_index=gpu_index),
) -> RunResult:
    settings, speed, status, error = requested, None, "ok", None
    with monitor(gpu_index) as gpu_monitor:
        try:
            engine_run = run_engine()
            settings, speed = engine_run.settings, engine_run.speed
        except BackendError as failure:
            status, error = failure.status, str(failure)
    gpu = gpu_monitor.summary()
    hardware = detect_hardware()

    return RunResult(
        rig_id=rig_id(),
        repetitions=repetitions,
        status=status,
        error=error,
        settings=settings,
        hardware=hardware,
        gpu_index=gpu_index,
        model_facts=model_facts(settings.model),
        gpu_facts=gpu_facts(hardware, gpu_index),
        speed=speed,
        gpu=gpu,
        energy=energy_and_cost(gpu, speed, price),
    )
