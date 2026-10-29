"""Train an XGBoost model to predict writing speed, and score it against the baselines.

Run from the project folder, after scripts/combine_runs.py:
    uv run --group train python scripts/train.py
"""

# The formula in 'localtune predict' already gets the rough size of a model's writing speed
# right from physics: memory bandwidth divided by the bytes read for each word. What it misses
# is everything that makes real runs faster or slower than that: the settings, how much of the
# model sits in RAM, how the model is built, how long the conversation already is. So the
# model here doesn't learn the speed from scratch. It learns the gap: how many times faster or
# slower than the formula each run really was. Its guess is then the formula's estimate times
# that learned factor. Starting from physics this way needs far less data than learning speed
# alone, and a GPU the model never saw still gets a sensible answer from its spec sheet.
#
# It is scored exactly like the baselines (evaluate.py): each GPU in turn is left out, the model
# is trained on every other GPU's runs, and it predicts the one left out.

import math
import sys

import numpy as np
from evaluate import (
    ESTIMATE,
    GPU,
    RUNS_FILE,
    TARGET,
    Method,
    calibrated,
    console,
    formula,
    load_usable,
    score,
)
from rich.table import Table

# The inputs the model may see: only what is known before a run starts, about the settings,
# the computer, the GPU's spec sheet and the model. Nothing measured during the run (power,
# clock speeds, how busy the GPU was) is in here: those are only known afterwards, so a model
# that used them would look brilliant in testing and fail for anyone asking about a new GPU.
# Each one is a column of the combined table, used as a number.

NUMBER_COLUMNS = [
    "settings.context_size",
    "settings.cpu_moe_layers",
    "settings.threads",
    "settings.ubatch_size",
    "hardware.cpu_threads",
    "hardware.ram_gb",
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
# engine and operating system ran it.

KV_CACHE_SIZE = {"f16": 1.0, "q8_0": 34 / 64, "q4_0": 18 / 64}


def number(row: dict, column: str) -> float:
    value = row.get(column)
    if value in (None, "", "None"):
        return math.nan
    if value in ("True", "False"):
        return float(value == "True")
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


def features(rows: list[dict]) -> np.ndarray:
    return np.array(
        [[number(row, c) for c in NUMBER_COLUMNS] + list(worked_out(row).values()) for row in rows]
    )


# What the model learns for each run: how many times faster or slower than the formula it was,
# as a logarithm, so twice as fast and half as fast count as the same size of miss.


def gap(rows: list[dict]) -> np.ndarray:
    return np.array([math.log(float(r[TARGET]) / float(r[ESTIMATE])) for r in rows])


# The model's settings, kept small and fixed (no searching for the best ones, which would need
# a separate set of runs to judge them on): a few hundred shallow trees, each correcting the
# last a little. Shallow trees and the random sampling of runs and inputs keep it from simply
# memorising the few GPUs it has seen. The seed makes every training give the same result.

SETTINGS = {
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 0,
}
TREES = 300


def fit(train: list[dict]):
    import xgboost

    data = xgboost.DMatrix(features(train), label=gap(train), feature_names=FEATURES)
    return xgboost.train(SETTINGS, data, num_boost_round=TREES)


def xgboost(train: list[dict], test: list[dict]) -> list[float] | None:
    if not train:
        return None
    import xgboost

    data = xgboost.DMatrix(features(test), feature_names=FEATURES)
    factors = np.exp(fit(train).predict(data))
    return [float(row[ESTIMATE]) * float(f) for row, f in zip(test, factors, strict=True)]


# Which inputs the model leaned on most, from a model trained on every usable run: how much
# each one improved its guesses overall (XGBoost's "gain"), as a share of the total.


def importance(rows: list[dict], top: int = 10) -> list[tuple[str, float]]:
    gains = fit(rows).get_score(importance_type="total_gain")
    total = sum(gains.values()) or 1.0
    return sorted(((name, v / total) for name, v in gains.items()), key=lambda x: -x[1])[:top]


def main() -> None:
    if not RUNS_FILE.exists():
        console.print(
            "No combined table yet. Run this first: uv run python scripts/combine_runs.py"
        )
        sys.exit(1)
    rows, _ = load_usable(RUNS_FILE)
    if not rows:
        console.print("No usable runs with an estimate to train on yet.")
        sys.exit(1)
    methods: dict[str, Method] = {"Formula": formula, "Calibrated": calibrated, "XGBoost": xgboost}
    table = Table(title="Writing speed (decode tok/s), each GPU predicted without its own runs")
    table.add_column("GPU")
    for column in ["Runs"] + [f"{name} error" for name in methods]:
        table.add_column(column, justify="right")
    for line in score(rows, methods):
        table.add_row(*line)
    console.print(table)

    inputs = Table(title="What the model leaned on most, trained on every run")
    inputs.add_column("Input")
    inputs.add_column("Share", justify="right")
    for name, share in importance(rows):
        inputs.add_row(name, f"{share:.0%}")
    console.print(inputs)
    gpus = len({row[GPU] for row in rows})
    console.print(
        f"Trained on {len(rows)} runs from {gpus} GPUs. With few GPUs, the inputs that describe "
        "the GPU barely vary, so the model can't learn much about how GPUs differ yet."
    )


if __name__ == "__main__":
    main()
