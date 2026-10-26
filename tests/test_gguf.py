# Tests for reading how a model is built from the start of its .gguf file.

import struct

import pytest

from localtune.gguf import GgufError, read_layout


def text(s: str) -> bytes:
    data = s.encode()
    return struct.pack("<Q", len(data)) + data


def gguf_file(path, facts: dict, tensors: dict) -> None:
    # A tiny .gguf file with the given facts (text or whole numbers) and blocks of weights by
    # shape. Only the beginning is written, which is all read_layout reads.
    body = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(facts) + 1)
    for key, value in facts.items():
        if isinstance(value, str):
            body += text(key) + struct.pack("<I", 8) + text(value)
        else:
            body += text(key) + struct.pack("<II", 4, value)
    body += text("general.tags") + struct.pack("<IIQ", 9, 8, 1) + text("chat")
    for name, shape in tensors.items():
        body += text(name) + struct.pack("<I", len(shape))
        body += b"".join(struct.pack("<Q", d) for d in shape) + struct.pack("<IQ", 2, 0)
    path.write_bytes(body)


def test_reads_the_layers_of_an_ordinary_model(tmp_path):
    model = tmp_path / "dense.gguf"
    facts = {"general.architecture": "llama", "llama.block_count": 32}
    gguf_file(model, facts, {"blk.0.attn_q.weight": (64, 64), "output.weight": (64, 10)})
    layout = read_layout(model)
    assert layout.layers == 32
    assert layout.experts is None and layout.expert_share == 0


def test_reads_how_much_of_a_mixture_of_experts_model_is_experts(tmp_path):
    model = tmp_path / "moe.gguf"
    facts = {"general.architecture": "qwen3moe", "qwen3moe.block_count": 48}
    facts["qwen3moe.expert_count"] = 128
    tensors = {"blk.0.attn_q.weight": (100, 10), "blk.0.ffn_up_exps.weight": (100, 10, 3)}
    gguf_file(model, facts, tensors)
    layout = read_layout(model)
    assert layout.layers == 48 and layout.experts == 128
    assert layout.expert_share == pytest.approx(0.75)


def test_other_files_are_refused(tmp_path):
    not_gguf = tmp_path / "notes.txt"
    not_gguf.write_text("hello")
    with pytest.raises(GgufError):
        read_layout(not_gguf)
    cut_short = tmp_path / "cut.gguf"
    cut_short.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(GgufError):
        read_layout(cut_short)
