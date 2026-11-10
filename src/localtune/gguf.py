"""Read how a model is built from the start of its .gguf file."""

# A .gguf model file (the kind llama.cpp and Ollama use) begins with a list of facts about the
# model, then the name and shape of every block of weights in it; the weights themselves come
# after. Reading only that beginning tells localtune how many layers the model has and, for a
# mixture-of-experts model, how much of it is experts, without loading gigabytes. It also says
# how much memory each token of the conversation takes (the KV cache), which grows with the
# conversation's length. "localtune tune" (tune.py) uses this to work out how much of a model
# too big for the GPU can go on it.

import struct
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel


class ModelLayout(BaseModel):
    layers: int
    experts: int | None
    expert_share: float  # how much of the model's weights are experts, from 0 to 1
    kv_bytes_per_token: int = 0  # at full quality (f16); 0 if the file doesn't say
    context_length: int | None = None  # the longest conversation the model was built for


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


# Each token of the conversation keeps a key and a value per attention head that shares them
# ("KV heads") in every layer, at 2 bytes per number in an f16 cache. Some models give a
# different head count per layer. Hybrid models like Qwen3.5 only have attention in every
# few layers ("full_attention_interval"); the layers between keep a small fixed-size memory
# instead, which doesn't grow with the conversation.


def kv_bytes_per_token(meta: dict, arch: str, layers: int) -> int:
    heads = meta.get(f"{arch}.attention.head_count")
    kv_heads = meta.get(f"{arch}.attention.head_count_kv", heads)
    if isinstance(heads, list):
        heads = max(heads)
    key = meta.get(f"{arch}.attention.key_length")
    if key is None and heads and meta.get(f"{arch}.embedding_length"):
        key = meta[f"{arch}.embedding_length"] // heads
    value = meta.get(f"{arch}.attention.value_length", key)
    if not (kv_heads and key and value):
        return 0
    per_layer = kv_heads if isinstance(kv_heads, list) else [kv_heads] * layers
    interval = meta.get(f"{arch}.full_attention_interval")
    if interval:
        per_layer = [n for i, n in enumerate(per_layer) if (i + 1) % interval == 0]
    return sum(n * (key + value) * 2 for n in per_layer)


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
        kv_bytes_per_token=kv_bytes_per_token(meta, arch, layers),
        context_length=meta.get(f"{arch}.context_length") or None,
    )
