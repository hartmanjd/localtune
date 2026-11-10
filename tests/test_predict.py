# Tests for the estimates "localtune predict" gives before a model is downloaded. They use the
# real model and GPU tables. On an RTX 4070 Ti SUPER (672 GB/s), qwen3:14b's 9,276,184,896-byte
# file gives 0.55 x 672 / 9.276 = 39.8 tok/s by hand.

import pytest
from typer.testing import CliRunner

from localtune import cli
from localtune.gpus import find_gpu_spec
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.models import find_model
from localtune.predict import estimate_for, kept_in_ram, predict
from localtune.results import RunResult, RunSettings, Speed, energy_used, model_facts

runner = CliRunner()
GPU = find_gpu_spec("RTX 4070 Ti SUPER", 16)


def test_a_model_that_fits_writes_at_bandwidth_over_size():
    p = predict(*find_model("qwen3:14b"), GPU, tokens=768, ram_gb=32)
    assert p.in_ram_gb == 0
    assert p.fits_on_computer
    assert p.decode_tok_s == pytest.approx(39.8, abs=0.1)


def test_a_mixture_of_experts_model_too_big_for_the_gpu_is_split_with_ram():
    model, file = find_model("qwen3:30b")
    p = predict(model, file, GPU, tokens=768, ram_gb=32)
    used_per_word = file.size_bytes * model.active_params / model.params
    all_on_gpu = 0.55 * 672e9 / used_per_word
    all_in_ram = 0.55 * 51.2e9 / used_per_word
    assert p.in_ram_gb > 1
    assert all_in_ram < p.decode_tok_s < all_on_gpu


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
    assert "Decode, formula guess" in result.output
    assert "about 40 tok/s" in result.output
    assert "localtune run qwen3:14b" in result.output
    assert "ollama pull" not in result.output


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
    assert "formula guess" not in result.output
    assert "localtune run" not in result.output


def saved_run(**changes) -> RunResult:
    fields = dict(
        rig_id="test-rig",
        repetitions=3,
        settings=RunSettings(engine="llama.cpp", model="qwen3:14b"),
        hardware=HardwareInfo(
            os="Windows-11", is_wsl=False, cpu_name=None, cpu_threads=16, ram_gb=32.0, gpus=[]
        ),
        model_facts=model_facts("qwen3:14b"),
        gpu_facts=GPU,
        speed=Speed(prompt_tokens=3 * 512, generated_tokens=3 * 256, decode_tok_s=63),
        energy=energy_used(None, None),
    )
    return RunResult(**{**fields, **changes})


def test_a_saved_run_gets_the_same_estimate_as_predict():
    assert estimate_for(saved_run()) == pytest.approx(39.8, abs=0.1)


def test_runs_saved_without_facts_have_no_estimate():
    assert estimate_for(saved_run(model_facts=None)) is None
    assert estimate_for(saved_run(gpu_facts=None)) is None


# A faster CPU memory makes the part of a model in RAM faster, and a model that fits on the GPU
# doesn't care.


def test_the_computers_ram_speed_is_used_for_the_part_in_ram():
    slow = predict(*find_model("qwen3:30b"), GPU, tokens=768, ram_gb=64)
    fast = predict(*find_model("qwen3:30b"), GPU, tokens=768, ram_gb=64, ram_bandwidth=204.8)
    assert fast.decode_tok_s > slow.decode_tok_s
    fits = predict(*find_model("qwen3:14b"), GPU, tokens=768, ram_gb=64, ram_bandwidth=204.8)
    assert fits.decode_tok_s == pytest.approx(39.8, abs=0.1)


# The estimate a measured run shows stays the plain formula; only the trained model's starting
# point uses the run's own CPU's RAM speed.


def test_only_the_starting_point_uses_the_runs_own_ram_speed():
    hardware = HardwareInfo(
        os="Linux",
        is_wsl=False,
        cpu_name="AMD EPYC 7402 24-Core Processor",
        cpu_threads=24,
        ram_gb=128.0,
        gpus=[],
    )
    run = saved_run(
        settings=RunSettings(engine="llama.cpp", model="qwen3:30b"),
        hardware=hardware,
        model_facts=model_facts("qwen3:30b"),
    )
    plain = predict(*find_model("qwen3:30b"), GPU, tokens=768, ram_gb=128).decode_tok_s
    assert estimate_for(run) == pytest.approx(plain)
    assert estimate_for(run, own_ram=True) > plain


# Layers a run kept in RAM on purpose count in the starting point even when the model would fit
# on the GPU; the plain formula still assumes the usual placement.


def test_layers_kept_in_ram_count_in_the_starting_point():
    facts = model_facts("llama3.1:8b")
    assert kept_in_ram(facts, None, None) == 0
    assert kept_in_ram(facts, -1, None) == 0
    assert kept_in_ram(facts, facts.layers + 1, None) == 0
    assert kept_in_ram(facts, facts.layers // 2, None) == pytest.approx(facts.file_bytes / 2)

    def run(gpu_layers):
        return saved_run(
            settings=RunSettings(engine="llama.cpp", model="llama3.1:8b", gpu_layers=gpu_layers),
            model_facts=facts,
        )

    assert estimate_for(run(16)) == pytest.approx(estimate_for(run(None)))
    assert estimate_for(run(16), own_ram=True) < estimate_for(run(None), own_ram=True) / 3


def test_experts_kept_in_ram_are_part_of_the_file():
    facts = model_facts("qwen3:30b")
    all_experts = kept_in_ram(facts, None, facts.layers)
    assert 0.8 * facts.file_bytes < all_experts < facts.file_bytes
    assert kept_in_ram(facts, None, facts.layers // 2) == pytest.approx(all_experts / 2)
    assert kept_in_ram(model_facts("qwen3:14b"), None, 10) == 0


# "localtune models" and "localtune gpus" print the tables predict reads, optionally only the
# rows whose name contains a word.


def test_models_lists_the_model_table():
    result = runner.invoke(cli.app, ["models", "qwen3"])
    assert result.exit_code == 0
    assert "qwen3:14b" in result.output
    assert "llama3.1:8b" not in result.output


def test_gpus_lists_the_gpu_table():
    result = runner.invoke(cli.app, ["gpus", "4070 ti super"])
    assert result.exit_code == 0
    assert "RTX 4070 Ti SUPER" in result.output
    assert "RTX 4090" not in result.output


def test_a_search_with_no_match_says_so():
    assert runner.invoke(cli.app, ["models", "no-such-model"]).exit_code == 1
    assert runner.invoke(cli.app, ["gpus", "no-such-gpu"]).exit_code == 1
