"""Writing-speed guesses from the trained prediction model, when one is on this computer."""

# The trained prediction model (scripts/train.py) is kept as one file in localtune's own folder
# (results.home_dir), never in the project, so it is only on the computers it was copied to.
# Where it isn't, or where XGBoost isn't installed (it's only in the train group, not in the
# normal install), nothing here does anything and 'localtune predict' shows the formula alone.
#
# The model learned how many times faster or slower than the formula (predict.py) real runs
# were, from the inputs in features.py. So a guess is the formula's estimate times that learned
# factor, for a row of run data shaped like the ones it was trained on. A model file saved with
# different inputs than this version of localtune works out is ignored, because its guesses
# would mean nothing.

import math
from pathlib import Path

from localtune.features import FEATURES, inputs, number
from localtune.gpus import GpuSpec
from localtune.hardware import HardwareInfo
from localtune.predict import Prediction
from localtune.results import ModelFacts, RunSettings, flatten, home_dir
from localtune.runners.llamacpp import DEPTH_FROM
from localtune.tune import DEFAULT_UBATCH


def model_file() -> Path:
    return home_dir() / "speed-model.json"


def load_model():
    path = model_file()
    if not path.exists():
        return None
    try:
        import xgboost
    except ImportError:
        return None
    model = xgboost.Booster()
    model.load_model(str(path))
    return model if model.feature_names == FEATURES else None


def trained_estimate(row: dict) -> float | None:
    estimate = number(row, "estimate.decode_tok_s")
    model = load_model()
    if model is None or math.isnan(estimate):
        return None
    import numpy as np
    import xgboost

    data = xgboost.DMatrix(np.array([inputs(row)]), feature_names=FEATURES)
    return estimate * math.exp(float(model.predict(data)[0]))


# 'localtune predict' asks about a run that hasn't happened, so it is described as a row shaped
# like the saved ones (results.flat_record): llama.cpp with its usual settings (tune.DEFAULTS),
# on this computer (or, for a GPU named with --gpu-name, on a computer nothing else is known
# about), for the use's prompt and answer, writing after the prompt when it's long enough for
# that to be measured (DEPTH_FROM). When the model doesn't all fit on the GPU, it is split the
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
        "estimate": {"decode_tok_s": p.decode_tok_s},
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
