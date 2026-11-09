"""Writing-speed guesses from the trained prediction model that comes with localtune."""

# The trained prediction model (scripts/train.py) ships inside the package, in
# src/localtune/data/speed-model.json, so everyone who installs localtune gets its guesses.
# A model saved in localtune's own folder (results.home_dir) is used instead when there is one,
# so a newly trained model can be tried on one computer before it ships.
#
# The model learned how many times faster or slower than the formula (predict.py) real runs
# were, from the inputs in features.py. It starts from the formula worked out with the
# computer's own RAM speed (START, predict.estimate_for). So a guess is that estimate times the
# learned factor, for a row of run data shaped like the ones it was trained on. A model file
# saved with different inputs than this version of localtune works out is ignored, because its
# guesses would mean nothing. XGBoost is loaded only when a guess is asked for, so the other
# commands start as fast as before. Where it can't load (on a Mac without the OpenMP library it
# needs, from 'brew install libomp'), there is no guess and 'localtune predict' shows the formula.

import math
from importlib.resources import files
from pathlib import Path

from localtune.features import FEATURES, inputs, number
from localtune.gpus import GpuSpec
from localtune.hardware import HardwareInfo
from localtune.predict import Prediction, estimate_for
from localtune.results import ModelFacts, RunResult, RunSettings, flat_record, flatten, home_dir
from localtune.runners.llamacpp import DEPTH_FROM
from localtune.tune import DEFAULT_UBATCH

START = "estimate.own_ram_decode_tok_s"


def model_file() -> Path:
    return home_dir() / "speed-model.json"


def bundled_model_file() -> Path:
    return Path(str(files("localtune") / "data" / "speed-model.json"))


def load_model():
    path = model_file() if model_file().exists() else bundled_model_file()
    if not path.exists():
        return None
    try:
        import xgboost
    except Exception:
        return None
    model = xgboost.Booster()
    model.load_model(str(path))
    return model if model.feature_names == FEATURES else None


def trained_estimate(row: dict) -> float | None:
    estimate = number(row, START)
    model = load_model()
    if model is None or math.isnan(estimate):
        return None
    import numpy as np
    import xgboost

    data = xgboost.DMatrix(np.array([inputs(row)]), feature_names=FEATURES)
    return estimate * math.exp(float(model.predict(data)[0]))


# A finished run's own guess, from the run as it was saved: the same row the model was trained on
# (scripts/combine_runs.py), so 'localtune run' can show it beside the measured speed.


def trained_for_run(run: RunResult) -> float | None:
    row = flat_record(run) | {
        "estimate.decode_tok_s": estimate_for(run),
        "estimate.own_ram_decode_tok_s": estimate_for(run, own_ram=True),
    }
    return trained_estimate(row)


# 'localtune predict' asks about a run that hasn't happened, so it is described as a row shaped
# like the saved ones (results.flat_record): llama.cpp with its usual settings (tune.DEFAULTS),
# on this computer (or, for a GPU named with --gpu-name, on a computer nothing else is known
# about), for the use's prompt and answer, writing after the prompt when it's long enough for
# that to be measured (DEPTH_FROM). own_ram is the estimate worked out with this computer's own
# RAM speed, the model's starting point. When the model doesn't all fit on the GPU, it is split the
# way the formula assumes: a dense model keeps as many of its layers on the GPU as fit, a
# mixture-of-experts model keeps every layer there and moves just enough layers' experts to RAM.


def planned_run(
    p: Prediction,
    facts: ModelFacts,
    gpu: GpuSpec,
    hardware: HardwareInfo,
    gpu_index: int,
    prompt_tokens: int,
    gen_tokens: int,
    own_ram: Prediction | None = None,
) -> dict:
    settings = RunSettings(
        engine="llama.cpp",
        model=facts.name,
        quant=facts.quant,
        context_size=prompt_tokens + gen_tokens,
        kv_cache_k="f16",
        kv_cache_v="f16",
        flash_attention=True,
        ubatch_size=DEFAULT_UBATCH,
        decode_depth=prompt_tokens if prompt_tokens >= DEPTH_FROM else None,
        **split(facts, p.in_ram_gb * 1024**3),
    )
    machine = hardware.model_dump(mode="json", exclude={"gpus"})
    machine["gpu"] = hardware.gpus[gpu_index].model_dump(mode="json")
    run = {
        "settings": settings.model_dump(mode="json"),
        "hardware": machine,
        "model_facts": facts.model_dump(mode="json"),
        "gpu_facts": gpu.model_dump(mode="json"),
        "estimate": {
            "decode_tok_s": p.decode_tok_s,
            "own_ram_decode_tok_s": (own_ram or p).decode_tok_s,
        },
    }
    return flatten(run)


def split(facts: ModelFacts, in_ram: float) -> dict:
    if in_ram <= 0:
        return {"gpu_layers": -1}
    if facts.experts and facts.experts_used:
        unused = 1 - facts.experts_used / facts.experts
        experts_per_layer = (
            facts.file_bytes * (facts.params - facts.active_params) / facts.params / unused
        ) / facts.layers
        moved = min(facts.layers, math.ceil(in_ram / experts_per_layer))
        return {"gpu_layers": -1, "cpu_moe_layers": moved}
    on_gpu = math.floor(facts.layers * (1 - in_ram / facts.file_bytes))
    return {"gpu_layers": max(0, on_gpu)}
