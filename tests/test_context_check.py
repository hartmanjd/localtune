# Tests for proving the biggest context. No model is loaded: a made-up GPU answers each try from
# simple arithmetic (a 9 GB model, a quarter of a megabyte per token, a 16 GB card), so the
# search's steps can be checked by hand.

import time

import pytest
from typer.testing import CliRunner

from localtune import cli, context_check
from localtune.context_check import ContextProof, ContextTry, latest_proof, prove, save_proof
from localtune.gguf import ModelLayout
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.runners.base import BackendError

GIB = 1024**3
TOKEN_BYTES = 262_144
GPU = GpuInfo(index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="1")
runner = CliRunner()


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))


def a_gpu(model_gb=9.0, others_gb=0.5, token_bytes=TOKEN_BYTES, limit_gb=16.0):
    """Answers a try the way the card would: the model, the conversation and other programs."""
    asked = []

    def try_context(context):
        asked.append(context)
        peak = model_gb + others_gb + context * token_bytes / GIB
        if peak > limit_gb:
            return ContextTry(context=context, loaded=False)
        return ContextTry(
            context=context, loaded=True, peak_vram_gb=peak, free_gb=round(limit_gb - peak, 2)
        )

    return try_context, asked


def test_the_formulas_size_is_proven_when_the_gpu_keeps_room():
    try_context, asked = a_gpu(others_gb=0.5)
    proven, tries = prove(try_context, 22_528, TOKEN_BYTES)
    assert proven == 22_528
    assert asked == [22_528]
    assert tries[0].free_gb == 1.0


def test_a_gpu_other_programs_are_using_gets_a_smaller_size():
    # 1.5 GB held by other programs leaves no margin at the formula's size: 1 GB short, which is
    # 4,096 tokens, and a thousand more to be clear of it.
    try_context, asked = a_gpu(others_gb=1.5)
    proven, tries = prove(try_context, 22_528, TOKEN_BYTES)
    assert asked == [22_528, 17_408]
    assert proven == 17_408
    assert not tries[0].proven and tries[1].proven


def test_a_model_that_needs_less_memory_per_token_gets_one_bigger_try():
    try_context, asked = a_gpu(token_bytes=TOKEN_BYTES // 2)
    proven, _ = prove(try_context, 22_528, TOKEN_BYTES)
    assert len(asked) == 2 and asked[1] > asked[0]
    assert proven == asked[1]


def test_the_bigger_try_stops_at_what_the_model_was_built_for():
    try_context, asked = a_gpu(token_bytes=TOKEN_BYTES // 2)
    proven, _ = prove(try_context, 22_528, TOKEN_BYTES, model_limit=24_000)
    assert asked == [22_528, 24_000]
    assert proven == 24_000


def test_a_bigger_try_that_overfills_keeps_the_size_already_proven():
    def try_context(context):
        free = 3.0 if context == 22_528 else 0.2
        return ContextTry(context=context, loaded=True, peak_vram_gb=16 - free, free_gb=free)

    proven, tries = prove(try_context, 22_528, TOKEN_BYTES)
    assert proven == 22_528
    assert len(tries) == 2


def test_a_model_that_doesnt_load_is_tried_a_tenth_smaller():
    try_context, asked = a_gpu(limit_gb=14.5)
    prove(try_context, 22_528, TOKEN_BYTES)
    assert asked[1] == 19_456


def test_nothing_is_proven_when_no_size_leaves_room():
    try_context, asked = a_gpu(model_gb=14.9)
    proven, tries = prove(try_context, 4_096, TOKEN_BYTES)
    assert proven is None
    assert len(tries) <= context_check.MAX_TRIES


# One try, with a made-up llama-server and made-up GPU readings.


class FakeServer:
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def readings(loaded_gb):
    count = iter(range(1000))

    def read():
        t = next(count)
        return {
            0: GpuSample(
                time_s=t,
                power_w=100.0,
                vram_used_gb=loaded_gb if t else 0.5,
                utilization_pct=10.0,
                temperature_c=50.0,
            )
        }

    return lambda gpu_index: GpuMonitor(interval_s=0.01, cards_reader=read)


def measure(tmp_path, ready, loaded_gb=14.0):
    commands = []

    def start(command, **options):
        commands.append(command)
        return FakeServer()

    tried = context_check.load_and_measure(
        "llama-server",
        tmp_path / "model.gguf",
        20_480,
        "q8_0",
        [GPU],
        start=start,
        monitor=readings(loaded_gb),
        ready=ready,
        sleep=lambda seconds: time.sleep(0.05),
    )
    return tried, commands[0]


def test_one_try_loads_the_whole_model_with_that_much_room_and_reads_the_memory(tmp_path):
    tried, command = measure(tmp_path, ready=lambda *args: None)
    assert command[command.index("-c") + 1] == "20480"
    assert command[command.index("-ngl") + 1] == "999"
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert tried.loaded
    assert tried.peak_vram_gb == 14.0
    assert tried.free_gb == 2.0
    assert tried.proven


def test_a_try_that_runs_out_of_memory_didnt_load(tmp_path):
    def ready(*args):
        raise BackendError("llama-server didn't start: out of memory", status="out_of_memory")

    tried, _ = measure(tmp_path, ready)
    assert not tried.loaded and not tried.proven


# Saved proofs, and the command itself.


def a_proof(model="qwen3:14b", proven=30_720):
    tried = ContextTry(context=proven, loaded=True, peak_vram_gb=14.5, free_gb=1.5)
    return ContextProof(
        rig_id="rig",
        model=model,
        kv_cache="f16",
        formula_context=37_888,
        proven_context=proven,
        tries=[tried],
    )


def test_the_latest_proof_for_a_model_is_read_back():
    save_proof(a_proof(proven=28_672))
    save_proof(a_proof(model="llama3.1:8b"))
    save_proof(a_proof(proven=30_720))
    assert latest_proof("rig", "qwen3:14b").proven_context == 30_720
    assert latest_proof("rig", "qwen3:14b", "q8_0") is None
    assert latest_proof("another-rig", "qwen3:14b") is None


def on_this_pc(monkeypatch, tmp_path, answer):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 1024)
    layout = ModelLayout(
        layers=40,
        experts=None,
        expert_share=0.0,
        kv_bytes_per_token=TOKEN_BYTES,
        context_length=40_960,
    )
    pc = HardwareInfo(os="Linux", is_wsl=False, cpu_name=None, cpu_threads=8, ram_gb=32, gpus=[GPU])
    monkeypatch.setattr(cli, "find_tool", lambda name: "/llama/llama-server")
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    monkeypatch.setattr(cli, "detect_hardware", lambda: pc)
    monkeypatch.setattr(cli, "rig_id", lambda: "rig")
    monkeypatch.setattr(cli, "load_and_measure", answer)
    return model


def test_the_command_proves_and_saves_a_size(monkeypatch, tmp_path):
    def answer(server, model_file, context, kv_cache, gpus, gpu):
        return ContextTry(context=context, loaded=True, peak_vram_gb=14.8, free_gb=1.2)

    model = on_this_pc(monkeypatch, tmp_path, answer)
    result = runner.invoke(cli.app, ["context", str(model)])
    assert result.exit_code == 0
    assert "Proven:" in result.output
    assert "40,960 tokens, all the model can take" in result.output
    assert latest_proof("rig", "model.gguf").proven_context == 40_960


def test_the_command_says_when_nothing_could_be_proven(monkeypatch, tmp_path):
    def answer(server, model_file, context, kv_cache, gpus, gpu):
        return ContextTry(context=context, loaded=False)

    model = on_this_pc(monkeypatch, tmp_path, answer)
    result = runner.invoke(cli.app, ["context", str(model)])
    assert result.exit_code == 1
    assert "No size could be proven" in result.output


def test_predict_shows_a_size_proven_on_this_computer(monkeypatch):
    pc = HardwareInfo(os="Linux", is_wsl=False, cpu_name=None, cpu_threads=8, ram_gb=32, gpus=[GPU])
    monkeypatch.setattr(cli, "detect_hardware", lambda: pc)
    monkeypatch.setattr(cli, "rig_id", lambda: "rig")
    save_proof(a_proof())
    result = runner.invoke(cli.app, ["predict", "qwen3:14b"])
    assert "Proven on this computer" in result.output
    assert "30,720 tokens" in result.output
