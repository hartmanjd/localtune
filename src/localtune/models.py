"""Look up a model's facts and file sizes: parameters, layers, and each quantization's size."""

# Whether a model fits on a GPU, and how fast it can write there, comes down to a few facts
# about it: how big its file is in the quantization being used (every new word reads the
# whole file once), how many of its parameters are used per word (fewer than the total for
# mixture-of-experts models), and its layers and attention heads, which set how much memory
# the conversation (KV cache) takes on top. The prediction model will combine these with the
# GPU's specs (gpus.py) to estimate speeds nobody has measured yet.
#
# The facts come from two tables shipped inside the package, made by
# scripts/build_model_table.py from the most downloaded models on Ollama and read from their
# own files (nothing in them is estimated): models.csv has one row per model size, with its
# place in Ollama's ranking, and model_files.csv one row per quantization of it. A model is
# looked up by the name people type into Ollama: qwen3:14b, qwen3:14b-q8_0, or just qwen3 for
# the size Ollama gives by default.

import csv
from functools import cache
from importlib.resources import files

from pydantic import BaseModel


class ModelFile(BaseModel):
    quant: str
    size_bytes: int
    ollama: str
    source: str


class ModelSpec(BaseModel):
    name: str
    family: str
    popularity_rank: int
    pulls: int
    latest: bool
    architecture: str
    params: int
    active_params: int
    layers: int
    kv_heads: int
    head_dim: int
    context_length: int | None
    experts: int | None
    experts_used: int | None
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


# A name with no size, like "gemma3" or "gemma3:latest", means the size Ollama downloads by
# default for it. The file found is the one that exact name downloads.


def find_model(name: str) -> tuple[ModelSpec, ModelFile] | None:
    wanted = name.lower()
    family, _, tag = wanted.partition(":")
    for model in load_model_table():
        if tag in ("", "latest") and model.family.lower() == family and model.latest:
            return model, model.files[0]
        for f in model.files:
            if f.ollama.lower() == wanted:
                return model, f
    return None
