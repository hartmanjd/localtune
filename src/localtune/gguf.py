"""Read how a model is built from the start of its .gguf file."""

# A .gguf model file (the kind llama.cpp and Ollama use) begins with a list of facts about the
# model, then the name and shape of every block of weights in it; the weights themselves come
# after. Reading only that beginning tells localtune how many layers the model has and, for a
# mixture-of-experts model, how much of it is experts, without loading gigabytes. "localtune
# tune" (tune.py) uses this to work out how much of a model too big for the GPU can go on it.

import struct
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel


class ModelLayout(BaseModel):
    layers: int
    experts: int | None
    expert_share: float  # how much of the model's weights are experts, from 0 to 1


class GgufError(Exception):
    pass


# How each kind of value is stored (8 is text and 9 a list, handled below).
SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def take(f: BinaryIO, size: int) -> bytes:
    data = f.read(size)
    if len(data) != size:
        raise GgufError("the file ended early")
    return data


def number(f: BinaryIO, fmt: str):
    return struct.unpack("<" + fmt, take(f, struct.calcsize(fmt)))[0]


def text(f: BinaryIO) -> str:
    return take(f, number(f, "Q")).decode(errors="replace")


def value(f: BinaryIO, kind: int):
    if kind == 8:
        return text(f)
    if kind == 9:
        item_kind, count = number(f, "I"), number(f, "Q")
        return [value(f, item_kind) for _ in range(count)]
    if kind not in SCALARS:
        raise GgufError(f"unknown value type {kind}")
    return number(f, SCALARS[kind])


# Expert weights are the blocks named "_exps"; their share of all weights is counted by number
# of values, which is close to their share of the file since they use the same quantization.


def read_layout(path: Path) -> ModelLayout:
    with path.open("rb") as f:
        if take(f, 4) != b"GGUF":
            raise GgufError("not a GGUF model file")
        number(f, "I")
        tensor_count, kv_count = number(f, "Q"), number(f, "Q")
        meta = {}
        for _ in range(kv_count):
            key = text(f)
            meta[key] = value(f, number(f, "I"))
        total = experts = 0
        for _ in range(tensor_count):
            name = text(f)
            count = 1
            for _ in range(number(f, "I")):
                count *= number(f, "Q")
            number(f, "I"), number(f, "Q")
            total += count
            if "_exps" in name:
                experts += count

    arch = meta.get("general.architecture")
    layers = meta.get(f"{arch}.block_count")
    if not layers:
        raise GgufError("the file doesn't say how many layers the model has")
    return ModelLayout(
        layers=layers,
        experts=meta.get(f"{arch}.expert_count") or None,
        expert_share=experts / total if total else 0.0,
    )
