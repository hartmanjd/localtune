# Tests for the llama.cpp runner. llama-bench's answer is read from a saved example
# (tests/data/llama_bench.json: its real output format, with the numbers of a Llama 2 7B Q4_0
# run on an RTX 4070 Ti SUPER), and llama-bench itself is replaced by small pretend programs, so
# no GPU, model file or llama.cpp install is needed.

import sys
from pathlib import Path

import pytest

from localtune.runners import llamacpp
from localtune.runners.base import BackendError
from localtune.runners.llamacpp import BenchOptions

EXAMPLE = (Path(__file__).parent / "data" / "llama_bench.json").read_text()
MODEL = Path("models/llama-2-7b.Q4_0.gguf")


def test_command_passes_only_the_settings_given():
    command = llamacpp.build_command("llama-bench", MODEL, BenchOptions(gpu_layers=99))
    assert command[:3] == ["llama-bench", "-m", str(MODEL)]
    assert command[command.index("-ngl") + 1] == "99"
    assert "-ctk" not in command
    assert "-fa" not in command


def test_command_sets_kv_cache_and_flash_attention():
    options = BenchOptions(kv_cache="q8_0", flash_attention=False)
    command = llamacpp.build_command("llama-bench", MODEL, options)
    assert command[command.index("-ctk") + 1] == "q8_0"
    assert command[command.index("-ctv") + 1] == "q8_0"
    assert command[command.index("-fa") + 1] == "off"


def test_reads_speeds_and_the_settings_used():
    run = llamacpp.parse_output(EXAMPLE, MODEL, repetitions=3)
    assert run.speed.prefill_tok_s == 7612.4
    assert run.speed.decode_tok_s == 133.2
    assert run.speed.prompt_tokens == 512 * 3
    assert run.speed.generated_tokens == 128 * 3
    assert run.speed.ttft_ms == pytest.approx(512 / 7612.4 * 1000)
    assert run.speed.busy_s == pytest.approx(3.084562)
    assert run.settings.quant == "Q4_0"
    assert run.settings.gpu_layers == 99
    assert run.settings.kv_cache_k == "f16"
    assert run.settings.engine_version == "b11669 (620d2225c)"
    assert run.settings.model == "llama-2-7b.Q4_0"


def test_unreadable_output_is_a_backend_error():
    with pytest.raises(BackendError, match="couldn't read"):
        llamacpp.parse_output("[", MODEL, repetitions=3)


def test_out_of_memory_is_recognised():
    error = llamacpp.describe_failure(
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9000 MiB on device 0: "
        "cudaMalloc failed: out of memory\n"
        "llama_bench: error: failed to create context with model 'big.gguf'\n"
    )
    assert error.status == "out_of_memory"
    assert str(error) == "llama_bench: error: failed to create context with model 'big.gguf'"


def test_other_failures_keep_the_error_line():
    error = llamacpp.describe_failure(
        "load_backend: loaded CPU backend\nllama_bench: error: failed to load model 'x.gguf'\n"
    )
    assert error.status == "crashed"
    assert "failed to load model" in str(error)


def pretend_llama_bench(tmp_path, script):
    path = tmp_path / "llama-bench"
    path.write_text(f"#!{sys.executable}\n{script}")
    path.chmod(0o755)
    return str(path)


def test_runs_llama_bench(tmp_path):
    (tmp_path / "out.json").write_text(EXAMPLE)
    llama_bench = pretend_llama_bench(
        tmp_path, f"print(open({str(tmp_path / 'out.json')!r}).read())"
    )
    run = llamacpp.run_llama_bench(llama_bench, MODEL, BenchOptions())
    assert run.speed.decode_tok_s == 133.2


def test_a_failed_run_raises(tmp_path):
    llama_bench = pretend_llama_bench(
        tmp_path,
        "import sys\nprint('llama_bench: error: out of memory', file=sys.stderr)\nsys.exit(1)",
    )
    with pytest.raises(BackendError) as failure:
        llamacpp.run_llama_bench(llama_bench, MODEL, BenchOptions())
    assert failure.value.status == "out_of_memory"


def test_requested_settings_keep_what_was_asked():
    settings = llamacpp.requested_settings(MODEL, BenchOptions(kv_cache="q4_0", gpu_layers=20))
    assert settings.kv_cache_k == settings.kv_cache_v == "q4_0"
    assert settings.gpu_layers == 20
    assert settings.model_file == "llama-2-7b.Q4_0.gguf"
