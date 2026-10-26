"""Look up a model's facts and file sizes: parameters, layers, and each quantization's size."""

# Whether a model fits on a GPU, and how fast it can write there, comes down to a few facts
# about it: how big its file is in the quantization being used (every new word reads the
# whole file once), how many of its parameters are used per word (fewer than the total for
# mixture-of-experts models), and its layers and attention heads, which set how much memory
# the conversation (KV cache) takes on top. The prediction model will combine these with the
# GPU's specs (gpus.py) to estimate speeds nobody has measured yet.
#
# The facts come from two tables shipped inside the package, made by
# scripts/build_model_table.py from Hugging Face and Ollama (nothing in them is estimated):
# models.csv has one row per model, model_files.csv one row per downloadable file of it.
# A model can be looked up by its own name (Qwen3-14B) or by its Ollama name (qwen3:14b),
# which also says which file Ollama downloads for that name.

import csv
from functools import cache
from importlib.resources import files

from pydantic import BaseModel


class ModelFile(BaseModel):
    quant: str
    size_bytes: int
    ollama: str | None
    source: str


class ModelSpec(BaseModel):
    name: str
    hf_repo: str
    params: int
    active_params: int
    layers: int
    kv_heads: int
    head_dim: int
    context_length: int | None
    experts: int | None
    experts_used: int | None
    config_source: str
    files: list[ModelFile]

    def file(self, quant: str) -> ModelFile | None:
        return next((f for f in self.files if f.quant == quant.upper()), None)


def read_csv(name: str) -> list[dict]:
    with (files("localtune") / "data" / name).open(newline="") as f:
        return [{key: value or None for key, value in row.items()} for row in csv.DictReader(f)]


@cache
def load_model_table() -> tuple[ModelSpec, ...]:
    by_model: dict[str, list[ModelFile]] = {}
    for row in read_csv("model_files.csv"):
        by_model.setdefault(row["model"], []).append(ModelFile(**row))
    return tuple(
        ModelSpec(**row, files=by_model.get(row["name"], [])) for row in read_csv("models.csv")
    )


# Ollama leaves off ":latest" when people type a name, so "gemma3" and "gemma3:latest" are
# the same name.


def find_model(name: str) -> tuple[ModelSpec, ModelFile | None] | None:
    wanted = name.lower().removesuffix(":latest")
    for model in load_model_table():
        for f in model.files:
            if f.ollama and f.ollama.lower().removesuffix(":latest") == wanted:
                return model, f
        if model.name.lower() == wanted:
            return model, None
    return None
