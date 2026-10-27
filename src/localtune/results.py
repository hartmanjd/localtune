"""The saved record of each benchmark run, and where those records are kept."""

# Every benchmark run ends up as one RunResult: what was run (model, quantization, KV cache and
# the other settings), on what (hardware.py), how fast it went, what the GPU did while it ran
# (gpu_monitor.py), and the energy and money it took (cost.py, prices.py). All runs together
# form localtune's dataset: the results table shows it, tuning compares runs from it, and the
# prediction models are trained on it. Because of that, a run that failed (out of memory, a
# crash) is saved too, so the dataset also knows which settings don't work.

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from localtune import __version__
from localtune.background import Background
from localtune.cost import ElectricityPrice, energy_cost
from localtune.gpu_monitor import GpuSummary
from localtune.gpus import GpuSpec, find_gpu_spec
from localtune.hardware import HardwareInfo
from localtune.models import find_model

# Bumped whenever the shape of RunResult changes in a way older records don't match, so code
# reading the dataset later can tell old and new records apart.
SCHEMA_VERSION = 1


# What was run: the engine (llama.cpp, Ollama, ...), the model and the settings it was run
# with. These are the knobs tuning turns, so each one gets its own field. Anything an engine
# doesn't have, or a run didn't set, stays empty.


class RunSettings(BaseModel):
    engine: str
    engine_version: str | None = None
    model: str
    model_file: str | None = None
    quant: str | None = None
    context_size: int | None = None
    gpu_layers: int | None = None
    cpu_moe_layers: int | None = None
    kv_cache_k: str | None = None
    kv_cache_v: str | None = None
    flash_attention: bool | None = None
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None


# How fast it went: how many tokens were read and written, the speed of each, the wait
# before the first token appeared, and how long the engine spent on those tokens ("busy").


class Speed(BaseModel):
    prompt_tokens: int
    generated_tokens: int
    prefill_tok_s: float | None = None
    decode_tok_s: float | None = None
    ttft_ms: float | None = None
    busy_s: float | None = None


# What the electricity came to. Energy is measured by the GPU monitor; cost is that energy at
# the price used, and the price is saved with where it came from. "Per million tokens" counts
# every token the run handled (prompt and answer), the unit paid AI services also price in.
# When the engine says how long it spent on the tokens, the energy is the GPU's busy power over
# that time, so loading the model doesn't count against the tokens; otherwise it is the energy
# of the whole run.


class EnergyCost(BaseModel):
    kwh: float | None
    kwh_per_million_tokens: float | None
    cost: float | None
    cost_per_million_tokens: float | None
    price: ElectricityPrice


def energy_and_cost(
    gpu: GpuSummary | None, speed: Speed | None, price: ElectricityPrice
) -> EnergyCost:
    energy_wh = gpu.energy_wh if gpu else None
    if gpu and gpu.busy_avg_power_w is not None and speed and speed.busy_s:
        energy_wh = gpu.busy_avg_power_w * speed.busy_s / 3600
    tokens = speed.prompt_tokens + speed.generated_tokens if speed else 0
    if energy_wh is None:
        return EnergyCost(
            kwh=None,
            kwh_per_million_tokens=None,
            cost=None,
            cost_per_million_tokens=None,
            price=price,
        )
    cost = energy_cost(energy_wh, price)
    return EnergyCost(
        kwh=energy_wh / 1000,
        kwh_per_million_tokens=energy_wh / 1000 / tokens * 1_000_000 if tokens else None,
        cost=cost,
        cost_per_million_tokens=cost / tokens * 1_000_000 if tokens else None,
        price=price,
    )


# The facts about the model and the GPU that set how fast a run can be, saved with it so each
# run is a complete example for the prediction model: the model's size, how many of its
# parameters each word uses, its layers and attention heads and the exact file that ran
# (models.py), and the GPU's memory bandwidth and compute (gpus.py). They come from the tables
# shipped with localtune, so a model or GPU that isn't in them leaves these empty.


class ModelFacts(BaseModel):
    name: str
    family: str
    architecture: str
    params: int
    active_params: int
    layers: int
    kv_heads: int
    head_dim: int
    context_length: int | None
    experts: int | None
    experts_used: int | None
    quant: str
    file_bytes: int


def model_facts(name: str) -> ModelFacts | None:
    found = find_model(name)
    if found is None:
        return None
    model, file = found
    facts = model.model_dump(exclude={"files", "popularity_rank", "pulls", "latest"})
    return ModelFacts(**facts, quant=file.quant, file_bytes=file.size_bytes)


def gpu_facts(hardware: HardwareInfo, gpu_index: int) -> GpuSpec | None:
    if gpu_index >= len(hardware.gpus):
        return None
    gpu = hardware.gpus[gpu_index]
    return find_gpu_spec(gpu.name, gpu.vram_gb)


# One saved run. Besides the groups above it records when and by which localtune version it
# was measured, which machine ran it (rig_id, a random ID with nothing personal in it, so many
# runs from one computer can be told apart from runs on others), what kind of run it was
# (a quick screening try, a full measurement or a final confirmation), whether it worked, and,
# for runs made by "localtune tune", which tuning session it belonged to (tune_id), so the runs
# that only differ in their settings can be compared with each other. Runs made by "localtune
# sweep" share a sweep_id the same way, so one night's runs over many models stay together, and
# a run made for a use (uses.py: chat, code or documents) says which, so a stopped sweep knows
# which model and use it already finished. It also keeps what else the computer was doing when
# the run started (background.py), so busy runs can be told apart.


class RunResult(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    rig_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    localtune_version: str = __version__
    kind: Literal["screening", "full", "confirm"] = "full"
    tune_id: str | None = None
    sweep_id: str | None = None
    use: str | None = None
    repetitions: int = 1
    status: Literal["ok", "out_of_memory", "crashed"] = "ok"
    error: str | None = None
    settings: RunSettings
    hardware: HardwareInfo
    gpu_index: int = 0
    model_facts: ModelFacts | None = None
    gpu_facts: GpuSpec | None = None
    speed: Speed | None = None
    gpu: GpuSummary | None = None
    background: Background | None = None
    energy: EnergyCost


# Where results live: one folder in your home directory (or wherever LOCALTUNE_HOME points),
# holding results.jsonl, with one run per line, and rig_id, this computer's random ID.
# One line per run means a new run is simply added at the end, and the file opens directly in
# data tools like pandas.


def home_dir() -> Path:
    return Path(os.environ.get("LOCALTUNE_HOME") or Path.home() / ".localtune")


def results_file() -> Path:
    return home_dir() / "results.jsonl"


def rig_id() -> str:
    path = home_dir() / "rig_id"
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_id = uuid4().hex
    path.write_text(new_id + "\n")
    return new_id


def save_result(result: RunResult, path: Path | None = None) -> Path:
    path = path or results_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")
    return path


# Reading the dataset back. A line that can't be read (for example, half written when the
# computer lost power) is skipped, so one bad line never hides every other run.


def load_results(path: Path | None = None) -> list[RunResult]:
    path = path or results_file()
    if not path.exists():
        return []
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            results.append(RunResult.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError):
            continue
    return results


# Every run as one flat row, for spreadsheets and data tools like pandas: each group's fields
# become columns named group.field (speed.decode_tok_s, settings.kv_cache_k, ...). Only the GPU
# that ran it is kept from the computer's list of GPUs, so each row has one GPU.


def flat_record(run: RunResult) -> dict:
    data = run.model_dump(mode="json")
    gpus = data["hardware"].pop("gpus")
    data["hardware"]["gpu"] = next((g for g in gpus if g["index"] == run.gpu_index), None)
    return flatten(data)


def flatten(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat
