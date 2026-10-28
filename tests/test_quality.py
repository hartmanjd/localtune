# Tests for measuring how much a smaller KV cache changes a model's answers.

import subprocess
import zipfile
from pathlib import Path

from localtune import quality
from localtune.hardware import HardwareInfo
from localtune.quality import (
    QualityResult,
    build_command,
    get_text,
    loss_pct,
    measure,
    parse_output,
    verdict,
)

PC = HardwareInfo(os="Linux", is_wsl=False, cpu_name="CPU", cpu_threads=8, ram_gb=32.0, gpus=[])
BASE = {"rig_id": "rig", "model": "qwen3:14b", "hardware": PC}

# What llama-perplexity prints, shortened: its build first, then a running estimate per chunk
# and the final one.
EXAMPLE_OUTPUT = """build: 11669 (6b1d2e9) with cc (Ubuntu 13.3.0) for x86_64-linux-gnu
llama_model_loader: loaded meta data with 28 key-value pairs
perplexity: calculating perplexity over 8 chunks, n_ctx=4096, batch_size=2048, n_seq=1
[1]5.9120,[2]6.0034,[3]6.1502,[4]6.0981,[5]6.1123,[6]6.1290,[7]6.1188,[8]6.1234,
Final estimate: PPL = 6.1234 +/- 0.04567
"""


def test_reads_the_final_perplexity_and_the_build():
    assert parse_output(EXAMPLE_OUTPUT) == (6.1234, 0.04567, "b11669 (6b1d2e9)")
    assert parse_output("error: failed to load model") is None


def test_every_cache_type_runs_with_flash_attention_on_the_given_split():
    command = build_command(
        "llama-perplexity", Path("m.gguf"), Path("wiki.test.raw"), "q4_0",
        {"gpu_layers": 49, "cpu_moe_layers": 6},
    )  # fmt: skip
    joined = " ".join(command)
    assert "-c 4096 --chunks 8 -fa on -ctk q4_0 -ctv q4_0" in joined
    assert "-ngl 49 --n-cpu-moe 6" in joined
    assert "-ngl" not in " ".join(build_command("p", Path("m"), Path("t"), "f16", {}))


def fake_perplexity(monkeypatch, output, returncode=0):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")

    monkeypatch.setattr(quality.subprocess, "run", run)


def test_a_measurement_saves_the_perplexity(monkeypatch):
    fake_perplexity(monkeypatch, EXAMPLE_OUTPUT)
    result = measure("p", Path("m.gguf"), Path("t"), "q8_0", {"gpu_layers": 99}, BASE)
    assert (result.status, result.perplexity, result.kv_cache) == ("ok", 6.1234, "q8_0")
    assert result.engine_version == "b11669 (6b1d2e9)"
    assert result.gpu_layers == 99


def test_running_out_of_memory_is_saved_as_such(monkeypatch):
    fake_perplexity(monkeypatch, "ggml_backend_cuda: cudaMalloc failed: out of memory\n", 1)
    result = measure("p", Path("m.gguf"), Path("t"), "f16", {}, BASE)
    assert result.status == "out_of_memory" and result.perplexity is None


def test_loss_is_measured_against_f16_in_plain_words():
    f16 = QualityResult(kv_cache="f16", model_file="m", perplexity=6.0, **BASE)
    q4 = QualityResult(kv_cache="q4_0", model_file="m", perplexity=6.12, **BASE)
    assert round(loss_pct(q4, f16), 2) == 2.0
    assert verdict(0.4) == "too small to notice"
    assert verdict(2.0) == "small, but there"
    assert verdict(5.0) == "noticeable"


def test_the_text_is_downloaded_once_and_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path / "home"))
    archive = tmp_path / "wikitext.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("wikitext-2-raw/wiki.test.raw", " = Robert Boulter = \n")
    path = get_text(archive.as_uri())
    assert path.read_text() == " = Robert Boulter = \n"
    assert get_text("https://unreachable.invalid/x.zip") == path
