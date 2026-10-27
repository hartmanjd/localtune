# Tests for the terminal command. They run the commands the way a person would, but with made-up
# hardware and GPU readings, so they check what gets shown without depending on the machine
# running them.

import itertools
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from localtune import cli
from localtune.doctor import Check
from localtune.gguf import ModelLayout
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import Speed, load_results
from localtune.runners.base import BackendError, EngineRun
from localtune.runners.llamacpp import parse_output
from localtune.runners.web_api import Server
from localtune.runners.web_api import requested_settings as web_api_settings
from localtune.tune import RESERVED_GB

runner = CliRunner()

EXAMPLE_RUN = (Path(__file__).parent / "data" / "llama_bench.json").read_text()

FAKE_PC = HardwareInfo(
    os="Linux 6.6.87.2-microsoft-standard-WSL2",
    is_wsl=True,
    cpu_name="AMD Ryzen 7 7800X3D 8-Core Processor",
    cpu_threads=16,
    ram_gb=32.0,
    gpus=[
        GpuInfo(
            index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="581.57"
        )
    ],
)


def test_info_shows_the_hardware(monkeypatch):
    wide_terminal(monkeypatch)
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC)
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "RTX 4070 Ti SUPER" in result.output
    assert "(WSL)" in result.output
    assert "32.0 GB" in result.output
    assert "672 GB/s" in result.output


def test_info_without_a_gpu(monkeypatch):
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": []}))
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "No NVIDIA GPU found" in result.output


def fake_monitor(power_w):
    # A GPU that reports the given power every second of a pretend clock, using 8 GB of VRAM.
    clock = itertools.count()

    def read():
        return GpuSample(
            time_s=next(clock),
            power_w=power_w,
            vram_used_gb=8.0,
            utilization_pct=90,
            temperature_c=65,
        )

    return lambda gpu_index: GpuMonitor(interval_s=0.001, gpu_index=gpu_index, reader=read)


def test_monitor_shows_energy_and_cost(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05", "--price", "0.20"])
    assert result.exit_code == 0
    assert "250 W avg" in result.output
    assert "0.050 USD per hour" in result.output
    assert "your price" in result.output


def test_monitor_uses_the_default_price(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05"])
    assert "US average" in result.output


def test_monitor_when_power_is_not_reported(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=None))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05"])
    assert result.exit_code == 0
    assert "doesn't report power" in result.output


def test_monitor_without_a_gpu(monkeypatch):
    monkeypatch.setattr(
        cli, "GpuMonitor", lambda gpu_index: GpuMonitor(interval_s=0.001, reader=lambda: None)
    )
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.01"])
    assert result.exit_code == 1
    assert "No GPU readings" in result.output


def test_monitor_with_a_zip_code(monkeypatch):
    monkeypatch.setattr(cli, "GpuMonitor", fake_monitor(power_w=250))
    result = runner.invoke(cli.app, ["monitor", "--seconds", "0.05", "--zip", "90210"])
    assert result.exit_code == 0
    assert "zip 90210 (CA)" in result.output


def fake_checks(*statuses):
    return lambda: [
        Check(name=name, status=status, detail="detail", fix="what to do")
        for name, status in zip(("NVIDIA GPU", "llama-bench"), statuses, strict=False)
    ]


def test_doctor_when_ready(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("ok", "ok"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 0
    assert "Ready to benchmark" in result.output


def test_doctor_shows_what_to_fix(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("fail", "ok"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "what to do" in result.output
    assert "fix the items marked" in result.output


def test_doctor_without_an_engine(monkeypatch):
    monkeypatch.setattr(cli, "run_checks", fake_checks("ok", "warn"))
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code == 1
    assert "needs llama.cpp" in result.output


def missing_llama_cpp(monkeypatch, installed):
    # A computer with a GPU but no llama.cpp, until the (pretend) install runs.
    def checks():
        bench = "ok" if installed else "warn"
        return fake_checks("ok", bench)()

    def fake_install(build, progress):
        installed.append(build)
        progress("llama.tar.gz", 1, 1)
        return "/home/you/.localtune/llama.cpp"

    monkeypatch.setattr(cli, "run_checks", checks)
    monkeypatch.setattr(cli, "choose_build", lambda: "linux-cuda-13.4")
    monkeypatch.setattr(cli, "install_llama_cpp", fake_install)


def test_doctor_installs_llama_cpp_when_asked(monkeypatch):
    installed = []
    missing_llama_cpp(monkeypatch, installed)
    result = runner.invoke(cli.app, ["doctor", "--install"])
    assert installed == ["linux-cuda-13.4"]
    assert "Installed llama.cpp" in result.output
    assert result.exit_code == 0


def test_doctor_asks_before_installing(monkeypatch):
    installed = []
    missing_llama_cpp(monkeypatch, installed)
    monkeypatch.setattr(cli, "can_ask", lambda: True)
    result = runner.invoke(cli.app, ["doctor"], input="n\n")
    assert "Download and install it now" in result.output
    assert installed == []
    assert result.exit_code == 1


def test_doctor_explains_when_it_cant_install(monkeypatch):
    installed = []
    missing_llama_cpp(monkeypatch, installed)

    def too_old():
        raise cli.InstallError("Update the driver")

    monkeypatch.setattr(cli, "choose_build", too_old)
    result = runner.invoke(cli.app, ["doctor", "--install"])
    assert "can't be installed automatically: Update the driver" in result.output
    assert installed == []


def fake_run(monkeypatch, tmp_path, outcome):
    # 'localtune run' with a pretend llama-bench that gives back the saved example run, or fails.
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "find_tool", lambda name: "llama-bench")
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC)
    monkeypatch.setattr(cli, "run_llama_bench", outcome)
    model = tmp_path / "llama-2-7b.Q4_0.gguf"
    model.write_text("")
    return model


def example_run(llama_bench, model, options, name=None):
    return parse_output(EXAMPLE_RUN, model, options.repetitions, name)


def test_run_shows_and_saves_the_result(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["run", str(model), "--price", "0.20"])
    assert result.exit_code == 0
    assert "7,612 tok/s" in result.output
    assert "133.2 tok/s" in result.output
    assert "Saved to" in result.output
    assert load_results()[0].speed.decode_tok_s == 133.2


def test_run_tests_the_lengths_of_a_use(monkeypatch, tmp_path):
    asked = []

    def remembers_options(llama_bench, model, options, name=None):
        asked.append(options)
        return example_run(llama_bench, model, options, name)

    model = fake_run(monkeypatch, tmp_path, remembers_options)
    runner.invoke(cli.app, ["run", str(model), "--use", "code", "--gen-tokens", "64"])
    assert (asked[0].prompt_tokens, asked[0].gen_tokens) == (8000, 64)


def test_run_refuses_an_unknown_use(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["run", str(model), "--use", "poems"])
    assert result.exit_code == 1
    assert "Unknown use poems" in result.output


def test_run_warns_when_another_program_is_using_the_gpu(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    busy = GpuSample(time_s=0, power_w=None, vram_used_gb=9.3, utilization_pct=0, temperature_c=50)
    monkeypatch.setattr(cli, "find_nvidia_smi", lambda: Path("nvidia-smi"))
    monkeypatch.setattr(cli, "other_use", lambda reader: busy)
    wide_terminal(monkeypatch)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert "9.3 GB of the GPU's memory is already in use" in result.output
    assert "ollama stop" in result.output
    assert result.exit_code == 0 and len(load_results()) == 1


def test_run_saves_a_failed_run(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options, name=None):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert result.exit_code == 1
    assert "ran out of GPU memory" in result.output
    assert load_results()[0].status == "out_of_memory"


def test_run_benchmarks_an_ollama_model_by_name(monkeypatch, tmp_path):
    fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: tmp_path / "sha256-abc")
    result = runner.invoke(cli.app, ["run", "qwen3:14b", "--price", "0.20"])
    assert result.exit_code == 0
    saved = load_results()[0].settings
    assert saved.model == "qwen3:14b"
    assert saved.model_file == "sha256-abc"


def test_run_with_an_unknown_model(monkeypatch, tmp_path):
    fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: None)
    result = runner.invoke(cli.app, ["run", "qwen3:8b"])
    assert result.exit_code == 1
    assert "ollama pull qwen3:8b" in result.output


def test_run_with_a_missing_model_file(monkeypatch, tmp_path):
    fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["run", "missing.gguf"])
    assert result.exit_code == 1
    assert "Model file not found: missing.gguf" in result.output
    assert "ollama pull" not in result.output


def test_run_without_llama_cpp(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "find_tool", lambda name: None)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert result.exit_code == 1
    assert "localtune doctor" in result.output


def test_run_refuses_an_unknown_kv_cache_type(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["run", str(model), "--kv-cache", "q9"])
    assert result.exit_code == 1
    assert "Unknown KV cache type" in result.output
    assert load_results() == []


# 'localtune run -b api' with a pretend server found on this computer, and a pretend runner that
# gives back made-up speeds.

OLLAMA = Server(name="Ollama", version="0.12.6", url="http://127.0.0.1:11434", models=["a", "b"])


def fake_api_run(monkeypatch, tmp_path, server=OLLAMA):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC)
    monkeypatch.setattr(cli, "find_server", lambda url: server)

    def run_web_api(server, model, options):
        speed = Speed(
            prompt_tokens=1500, generated_tokens=384, prefill_tok_s=3100, decode_tok_s=98.5
        )
        return EngineRun(settings=web_api_settings(server, model, options), speed=speed)

    monkeypatch.setattr(cli, "run_web_api", run_web_api)


def test_run_benchmarks_a_server(monkeypatch, tmp_path):
    fake_api_run(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["run", "b", "-b", "api", "--price", "0.20"])
    assert result.exit_code == 0
    assert "98.5 tok/s" in result.output
    assert "Ollama 0.12.6" in result.output
    saved = load_results()[0]
    assert (saved.settings.engine, saved.settings.model) == ("Ollama", "b")


def test_run_asks_which_model_when_the_server_has_several(monkeypatch, tmp_path):
    fake_api_run(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["run", "-b", "api"])
    assert result.exit_code == 1
    assert "Its models: a, b" in result.output
    assert load_results() == []


def test_run_without_a_server(monkeypatch, tmp_path):
    fake_api_run(monkeypatch, tmp_path, server=None)
    result = runner.invoke(cli.app, ["run", "-b", "api"])
    assert result.exit_code == 1
    assert "No local AI server found" in result.output


def test_run_refuses_llama_cpp_settings_for_a_server(monkeypatch, tmp_path):
    fake_api_run(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["run", "a", "-b", "api", "--kv-cache", "q8_0"])
    assert result.exit_code == 1
    assert "--kv-cache only work with llama.cpp" in result.output


def wide_terminal(monkeypatch):
    # The results table has many columns; a wide terminal keeps each value on one line.
    monkeypatch.setattr(cli, "console", Console(width=200))


def test_results_shows_saved_runs(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, example_run)
    runner.invoke(cli.app, ["run", str(model), "--price", "0.20"])
    result = runner.invoke(cli.app, ["results"])
    assert result.exit_code == 0
    assert "Saved runs (1 of 1)" in result.output
    assert "7,612" in result.output
    assert "133.2" in result.output
    assert "GPU layers 99, KV f16" in result.output
    assert "llama.cpp" in result.output


def test_results_shows_all_gpu_layers(monkeypatch, tmp_path):
    # llama.cpp reports -1 GPU layers when every layer is on the GPU.
    def all_layers(llama_bench, model, options, name=None):
        output = EXAMPLE_RUN.replace('"n_gpu_layers": 99', '"n_gpu_layers": -1')
        return parse_output(output, model, options.repetitions)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, all_layers)
    runner.invoke(cli.app, ["run", str(model)])
    result = runner.invoke(cli.app, ["results"])
    assert "GPU layers all" in result.output


def test_results_lists_failed_runs(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options, name=None):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    runner.invoke(cli.app, ["run", str(model)])
    result = runner.invoke(cli.app, ["results"])
    assert "out of memory" in result.output


def test_results_shows_only_the_last_runs(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    for _ in range(3):
        runner.invoke(cli.app, ["run", str(model)])
    result = runner.invoke(cli.app, ["results", "--last", "2"])
    assert "Saved runs (2 of 3)" in result.output


def test_results_filters_by_model(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    runner.invoke(cli.app, ["run", str(model)])
    result = runner.invoke(cli.app, ["results", "--model", "qwen"])
    assert "No saved runs yet" in result.output


def test_results_when_nothing_is_saved(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    result = runner.invoke(cli.app, ["results"])
    assert result.exit_code == 0
    assert "localtune run" in result.output


def tuned_run(llama_bench, model, options, name=None):
    # A pretend llama-bench where the settings change the speed: q8_0 and bigger micro-batches
    # are faster, and a q4_0 cache doesn't fit in memory.
    if options.kv_cache == "q4_0":
        raise BackendError("cudaMalloc failed", status="out_of_memory")
    run = example_run(llama_bench, model, options, name)
    run.settings.flash_attention = options.flash_attention
    run.settings.kv_cache_k = run.settings.kv_cache_v = options.kv_cache
    run.settings.ubatch_size = options.ubatch_size
    run.speed.decode_tok_s = (
        100 + options.ubatch_size / 256 + (10 if options.kv_cache == "q8_0" else 0)
    )
    return run


def test_tune_tries_every_combination_and_shows_the_fastest(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, tuned_run)
    result = runner.invoke(cli.app, ["tune", str(model), "--price", "0.20"])
    assert result.exit_code == 0
    assert "Trying 16 combinations" in result.output
    assert "out of GPU memory" in result.output
    assert "Fastest on this computer: flash attn on, KV q8_0, micro-batch 2048." in result.output
    assert f"localtune run {model} --flash-attn --kv-cache q8_0 --ubatch-size 2048" in result.output
    assert "Tuned is 15% faster than llama.cpp's usual settings" in result.output
    saved = load_results()
    assert len(saved) == 16
    assert len({run.tune_id for run in saved}) == 1 and saved[0].tune_id


def test_tune_says_when_tuning_made_no_real_difference(monkeypatch, tmp_path):
    def all_the_same(llama_bench, model, options, name=None):
        run = tuned_run(llama_bench, model, options, name)
        run.speed.decode_tok_s = 100 + (1 if options.ubatch_size == 2048 else 0)
        return run

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, all_the_same)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert "other settings are within 3% of it" in result.output
    assert "Tuned is within 3% of llama.cpp's usual settings" in result.output


def test_tune_explains_when_nothing_fits(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options, name=None):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert result.exit_code == 1
    assert "None of the settings worked" in result.output
    assert "--gpu-layers" in result.output


def test_tune_keeps_the_finished_runs_when_stopped(monkeypatch, tmp_path):
    calls = itertools.count()

    def stopped_on_the_third(llama_bench, model, options, name=None):
        if next(calls) == 2:
            raise KeyboardInterrupt
        return tuned_run(llama_bench, model, options, name)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, stopped_on_the_third)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert "Stopped" in result.output
    assert "Fastest so far: flash attn off, KV f16, micro-batch 512." in result.output
    assert "Only 2 of 16 combinations were tried" in result.output
    assert len(load_results()) == 2


def test_tune_keeps_settings_that_arent_tuned(monkeypatch, tmp_path):
    asked = []

    def remembers_options(llama_bench, model, options, name=None):
        asked.append(options)
        return tuned_run(llama_bench, model, options, name)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, remembers_options)
    result = runner.invoke(cli.app, ["tune", str(model), "--gpu-layers", "30"])
    assert f"localtune run {model} --gpu-layers 30 --flash-attn" in result.output
    assert {options.gpu_layers for options in asked} == {30}


def test_tune_first_splits_a_model_too_big_for_the_gpu(monkeypatch, tmp_path):
    # A mixture-of-experts model twice the GPU's free memory, where keeping the experts of
    # fewer than 25 layers in RAM runs out of GPU memory and more in RAM is slower.
    def split_run(llama_bench, model, options, name=None):
        if options.cpu_moe_layers < 25:
            raise BackendError("cudaMalloc failed", status="out_of_memory")
        if options.ubatch_size is None:
            run = example_run(llama_bench, model, options, name)
            run.speed.decode_tok_s = 100 - options.cpu_moe_layers
            return run
        return tuned_run(llama_bench, model, options, name)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, split_run)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=48, experts=128, expert_share=0.9)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 2400 / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert result.exit_code == 0
    assert "don't fit in the GPU's memory" in result.output
    assert "Fastest split: experts of 25 of 48 layers in RAM." in result.output
    assert f"localtune run {model} --gpu-layers 49 --cpu-moe 25 --flash-attn" in result.output
    assert len(load_results()) == 4 + 16


def test_tune_leaves_room_for_a_long_prompt(monkeypatch, tmp_path):
    # A model that fits beside a short prompt but not beside a 16,000-token one.
    asked = []

    def remembers_options(llama_bench, model, options, name=None):
        asked.append(options)
        if options.ubatch_size is None:
            return example_run(llama_bench, model, options, name)
        return tuned_run(llama_bench, model, options, name)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, remembers_options)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=40, experts=None, expert_share=0.0, kv_bytes_per_token=1)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 10000 / 1024**3})
    pc = FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    monkeypatch.setattr(cli, "detect_hardware", lambda: pc)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert "don't fit" not in result.output
    result = runner.invoke(cli.app, ["tune", str(model), "--prompt-tokens", "16000"])
    assert "a 16,128-token conversation don't fit" in result.output
    assert f"localtune run {model} --prompt-tokens 16000 --gpu-layers" in result.output
    assert asked[-1].prompt_tokens == 16000
    assert "Trying 13 combinations" in result.output
    result = runner.invoke(cli.app, ["tune", str(model), "--use", "documents"])
    assert f"localtune run {model} --use documents --gpu-layers" in result.output
    assert asked[-1].prompt_tokens == 16000


def test_tune_stops_when_no_split_fits(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options, name=None):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=40, experts=None, expert_share=0.0)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 2400 / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert result.exit_code == 1
    assert "No split fit in GPU memory" in result.output


def test_tune_without_llama_cpp(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "find_tool", lambda name: None)
    result = runner.invoke(cli.app, ["tune", "qwen3:14b"])
    assert result.exit_code == 1
    assert "localtune doctor" in result.output


def test_tune_says_how_long_is_left():
    assert cli.time_left(0, 0, 16) == ""
    assert cli.time_left(60, 2, 16) == " (about 7 min left)"
    assert cli.time_left(10, 15, 16) == " (under a minute left)"
