# Tests for speed_model.py: the run 'localtune predict' describes to the trained model, the
# model that ships with localtune, and a model saved in localtune's folder taking its place.
# They use the real model and GPU tables, and a model trained on made-up runs where flash
# attention makes writing 1.5x faster than the formula (tests/test_train.py), so its guess for
# predict's usual settings is about 1.5x the estimate: 40 tok/s for qwen3:14b on an RTX 4070 Ti
# SUPER becomes about 60.

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from localtune import cli
from localtune.features import FEATURES, worked_out
from localtune.gpus import find_gpu_spec
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.models import find_model
from localtune.predict import predict
from localtune.results import model_facts
from localtune.speed_model import (
    bundled_model_file,
    load_model,
    planned_run,
    split,
    trained_estimate,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_train import made_up_runs, train  # noqa: E402

runner = CliRunner()
GPU = find_gpu_spec("RTX 4070 Ti SUPER", 16)
PC = HardwareInfo(
    os="Windows 11",
    is_wsl=False,
    cpu_name="AMD Ryzen 7 7800X3D",
    cpu_threads=16,
    ram_gb=32,
    gpus=[GpuInfo(index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16, driver_version="")],
)


def planned(model: str, prompt_tokens: int, gen_tokens: int) -> dict:
    p = predict(*find_model(model), GPU, prompt_tokens + gen_tokens, PC.ram_gb)
    return planned_run(p, model_facts(model), GPU, PC, 0, prompt_tokens, gen_tokens)


def test_a_planned_run_uses_llama_cpp_usual_settings_on_this_pc():
    row = planned("qwen3:14b", 512, 256)
    assert row["settings.engine"] == "llama.cpp"
    assert row["hardware.gpu.vram_gb"] == 16
    assert row["gpu_facts.memory_bandwidth_gbs"] == 672
    assert row["estimate.decode_tok_s"] == pytest.approx(39.8, abs=0.1)
    found = worked_out(row)
    assert found["share of layers on GPU"] == 1.0
    assert found["flash attention"] == 1.0
    assert found["KV cache size"] == 1.0
    assert found["decode depth"] == 0.0
    assert found["Windows"] == 1.0


def test_a_long_prompt_is_written_after():
    assert planned("qwen3:14b", 16000, 256)["settings.decode_depth"] == 16000


def test_a_model_too_big_for_the_gpu_is_split():
    moe = planned("qwen3:30b", 512, 256)
    assert moe["settings.gpu_layers"] == -1
    assert 0 < moe["settings.cpu_moe_layers"] < moe["model_facts.layers"]
    dense = model_facts("qwen3:14b")
    assert split(dense, 0) == {"gpu_layers": -1}
    assert split(dense, dense.file_bytes / 2) == {"gpu_layers": dense.layers // 2}


def test_the_shipped_model_matches_this_version(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    assert bundled_model_file().exists()
    assert load_model().feature_names == FEATURES


def test_without_a_saved_model_the_shipped_one_guesses(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    assert 30 < trained_estimate(planned("qwen3:14b", 512, 256)) < 100
    result = runner.invoke(cli.app, ["predict", "qwen3:14b", "--gpu-name", "RTX 4070 Ti SUPER"])
    assert "trained model guess" in result.output


def test_predict_shows_the_saved_model_beside_the_formula(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    train.save(made_up_runs())
    result = runner.invoke(cli.app, ["predict", "qwen3:14b", "--gpu-name", "RTX 4070 Ti SUPER"])
    assert result.exit_code == 0
    assert "Decode, formula guess" in result.output
    assert "about 40 tok/s" in result.output
    assert "trained model guess" in result.output
    assert "about 60 tok/s" in result.output
