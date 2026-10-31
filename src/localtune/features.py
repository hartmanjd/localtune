"""The inputs the trained speed model sees for a run, worked out from one row of run data."""

# The trained prediction model (scripts/train.py) learns how far each run's real writing speed
# was from the formula in predict.py. These are the facts it learns from, and they have to be
# worked out the same way when it is trained and when it is asked about a new GPU or model, or
# its guesses mean nothing. So they live here, in one place: training reads them from rows of
# the combined table (scripts/combine_runs.py), and 'localtune predict' (speed_model.py) builds
# a row of the same shape for a run that hasn't happened yet. A row is a run flattened the way
# 'localtune results --csv' writes it (results.flat_record), with every value as text or a
# plain value.

import math

from localtune.tune import KV_CACHE_SIZE

# The inputs the model may see: only what is known before a run starts, about the settings,
# the computer, the GPU's spec sheet and the model. Nothing measured during the run (power,
# clock speeds, how busy the GPU was) is in here: those are only known afterwards, so a model
# that used them would look brilliant in testing and fail for anyone asking about a new GPU.
# Each one is a column of the combined table, used as a number. The CPU threads and RAM size
# aren't: every computer measured so far has its own, so they only told the model which computer
# a run came from, and it learned those computers instead of what makes runs fast. What the CPU
# and RAM do to the speed is in the estimate instead, through the CPU's RAM speed (cpus.py).

NUMBER_COLUMNS = [
    "settings.context_size",
    "settings.cpu_moe_layers",
    "settings.ubatch_size",
    "hardware.gpu.vram_gb",
    "hardware.gpu.power_limit_w",
    "hardware.gpu.max_sm_clock_mhz",
    "hardware.gpu.max_memory_clock_mhz",
    "hardware.gpu.pcie_gen",
    "hardware.gpu.pcie_lanes",
    "gpu_facts.memory_bandwidth_gbs",
    "gpu_facts.fp32_tflops",
    "gpu_facts.compute_capability",
    "gpu_facts.board_power_w",
    "model_facts.params",
    "model_facts.active_params",
    "model_facts.layers",
    "model_facts.kv_heads",
    "model_facts.head_dim",
    "model_facts.experts",
    "model_facts.experts_used",
    "model_facts.file_bytes",
    "estimate.decode_tok_s",
]

# A few inputs are worked out from columns, because the column alone means little: the share of
# the model's layers on the GPU (-1 means all of them), how much smaller the KV cache is than
# full size (tune.py), how compressed the model file is in bits per number, how deep into the
# conversation writing was measured, whether the GPU runs under its usual power, and which
# engine and operating system ran it. A value that isn't known is NaN, which the model treats
# as missing.


def number(row: dict, column: str) -> float:
    value = row.get(column)
    if value in (None, "", "None"):
        return math.nan
    if value in ("True", "False", True, False):
        return float(value in ("True", True))
    return float(value)


def worked_out(row: dict) -> dict[str, float]:
    layers = number(row, "model_facts.layers")
    on_gpu = number(row, "settings.gpu_layers")
    params = number(row, "model_facts.params")
    power = number(row, "hardware.gpu.power_limit_w")
    usual = number(row, "hardware.gpu.default_power_limit_w")
    return {
        "share of layers on GPU": 1.0 if math.isnan(on_gpu) or on_gpu < 0 else on_gpu / layers,
        "KV cache size": KV_CACHE_SIZE.get(row.get("settings.kv_cache_k") or "f16", math.nan),
        "flash attention": number(row, "settings.flash_attention"),
        "bits per number": number(row, "model_facts.file_bytes") * 8 / params,
        "decode depth": 0.0
        if row.get("settings.decode_depth") in (None, "")
        else number(row, "settings.decode_depth"),
        "power vs usual": power / usual,
        "Ollama": float(row.get("settings.engine") != "llama.cpp"),
        "Windows": float((row.get("hardware.os") or "").startswith("Windows")),
    }


FEATURES = NUMBER_COLUMNS + list(worked_out({}))


def inputs(row: dict) -> list[float]:
    return [number(row, c) for c in NUMBER_COLUMNS] + list(worked_out(row).values())
