# Tests for the estimates "localtune predict" gives before a model is downloaded. They use the
# real model and GPU tables, and check the estimates against speeds measured on an RTX 4070 Ti
# SUPER: qwen3:14b wrote 62 to 64 tok/s, and qwen3:30b, split between GPU and RAM, 134.

import pytest
from typer.testing import CliRunner

from localtune import cli
from localtune.gpus import find_gpu_spec
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.models import find_model
from localtune.predict import predict

runner = CliRunner()
GPU = find_gpu_spec("RTX 4070 Ti SUPER", 16)


def test_a_model_that_fits_writes_near_its_measured_speed():
    p = predict(*find_model("qwen3:14b"), GPU, tokens=768, ram_gb=32)
    assert p.in_ram_gb == 0
    assert p.fits_on_computer
    assert p.decode_tok_s == pytest.approx(63, rel=0.1)


def test_a_mixture_of_experts_model_too_big_for_the_gpu_is_split_with_ram():
    p = predict(*find_model("qwen3:30b"), GPU, tokens=768, ram_gb=32)
    assert p.in_ram_gb > 1
    assert p.decode_tok_s == pytest.approx(134, rel=0.15)


def test_a_longer_conversation_leaves_less_room_for_the_model():
    short = predict(*find_model("qwen3:30b"), GPU, tokens=768)
    long = predict(*find_model("qwen3:30b"), GPU, tokens=16256)
    assert long.conversation_gb > short.conversation_gb
    assert long.in_ram_gb > short.in_ram_gb
    assert long.decode_tok_s < short.decode_tok_s


def test_a_model_bigger_than_gpu_and_ram_together_cant_run():
    p = predict(*find_model("llama3.1:405b"), GPU, tokens=768, ram_gb=32)
    assert p.fits_on_computer is False
    assert p.decode_tok_s is None


def test_predict_for_a_named_gpu():
    result = runner.invoke(cli.app, ["predict", "qwen3:14b", "--gpu-name", "RTX 4070 Ti SUPER"])
    assert result.exit_code == 0
    assert "Yes, all of it" in result.output
    assert "about 62 tok/s" in result.output
    assert "localtune run qwen3:14b" in result.output


def test_predict_for_an_unknown_model():
    result = runner.invoke(cli.app, ["predict", "no-such-model:1b", "--gpu-name", "RTX 4090"])
    assert result.exit_code == 1
    assert "isn't in localtune's model table" in result.output


def test_predict_for_this_computers_gpu(monkeypatch):
    pc = HardwareInfo(
        os="Windows-11",
        is_wsl=False,
        cpu_name=None,
        cpu_threads=16,
        ram_gb=32.0,
        gpus=[
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 4070 Ti SUPER",
                vram_gb=16.0,
                driver_version="581.57",
            )
        ],
    )
    monkeypatch.setattr(cli, "detect_hardware", lambda: pc)
    result = runner.invoke(cli.app, ["predict", "llama3.1:405b"])
    assert result.exit_code == 0
    assert "Writing speed" not in result.output
    assert "localtune run" not in result.output
