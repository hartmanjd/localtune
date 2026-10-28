# Tests for the terminal command. They run the commands the way a person would, but with made-up
# hardware and GPU readings, so they check what gets shown without depending on the machine
# running them.

import csv
import itertools
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from localtune import cli, tune
from localtune.background import Background
from localtune.doctor import Check
from localtune.gguf import ModelLayout
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import RunResult, RunSettings, Speed, load_results
from localtune.runners import base
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
    monkeypatch.setattr(cli, "find_tool", lambda name: name)
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


def test_run_warns_and_notes_when_another_program_is_busy(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    busy = Background(gpu_utilization_pct=0, gpu_vram_used_gb=9.3, cpu_load_pct=40)
    monkeypatch.setattr(cli, "read_background", lambda gpu_index=0: busy)
    wide_terminal(monkeypatch)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert "9.3 GB of the GPU's memory is already in use and the CPU is already 40% busy" in (
        result.output
    )
    assert "ollama stop" in result.output
    assert result.exit_code == 0
    assert load_results()[0].background == busy


def test_run_starts_on_a_fresh_reading_once_the_gpu_is_freed(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    quiet = Background(gpu_utilization_pct=0, gpu_vram_used_gb=0.6)
    readings = iter([Background(gpu_utilization_pct=0, gpu_vram_used_gb=9.3), quiet])
    monkeypatch.setattr(cli, "read_background", lambda gpu_index=0: next(readings))
    monkeypatch.setattr(cli, "show_memory_meter", lambda gpu: "free")
    wide_terminal(monkeypatch)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert "9.3 GB of the GPU's memory is already in use" in result.output
    assert "The GPU is free now." in result.output
    assert load_results()[0].background == quiet


def test_the_memory_meter_only_shows_for_someone_watching(monkeypatch):
    monkeypatch.setattr(cli, "can_ask", lambda: False)
    assert cli.show_memory_meter(0) is None


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
    server = f"llama-server -m {model} -c 4096 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -ub 2048"
    assert server in result.output
    assert "room for a 4,096-token conversation" in result.output
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


def test_tune_passes_over_settings_that_leave_the_gpu_nearly_full(monkeypatch, tmp_path):
    def q8_0_fills_it(run):
        return run.settings.kv_cache_k == "q8_0"

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, tuned_run)
    monkeypatch.setattr(cli, "too_full", q8_0_fills_it)
    monkeypatch.setattr(tune, "too_full", q8_0_fills_it)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert "GPU memory nearly full" in result.output
    assert "'full' means the run left less than 1 GB" in result.output
    assert "KV f16, micro-batch 2048." in result.output


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
    assert "-ngl 49 --n-cpu-moe 25 -fa on" in result.output
    assert len(load_results()) == 4 + 16


def test_tune_keeps_a_split_clear_of_a_nearly_full_gpu(monkeypatch, tmp_path):
    # Like above, but keeping the experts of fewer than 27 layers in RAM leaves the GPU too full.
    def split_run(llama_bench, model, options, name=None):
        if options.ubatch_size is None:
            run = example_run(llama_bench, model, options, name)
            run.settings.cpu_moe_layers = options.cpu_moe_layers
            run.speed.decode_tok_s = 100 - options.cpu_moe_layers
            return run
        return tuned_run(llama_bench, model, options, name)

    def nearly_full(run):
        return (run.settings.cpu_moe_layers or 99) < 27

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, split_run)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=48, experts=128, expert_share=0.9)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 2400 / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    monkeypatch.setattr(cli, "too_full", nearly_full)
    monkeypatch.setattr(tune, "too_full", nearly_full)
    result = runner.invoke(cli.app, ["tune", str(model)])
    assert result.exit_code == 0
    assert "Fastest split: experts of 27 of 48 layers in RAM." in result.output


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
    assert "room for a 16,384-token conversation" in result.output
    result = runner.invoke(cli.app, ["tune", str(model), "--use", "documents"])
    assert f"localtune run {model} --use documents --gpu-layers" in result.output
    assert asked[-1].prompt_tokens == 16000


def test_tune_gives_a_smaller_kv_cache_its_own_split(monkeypatch, tmp_path):
    # A mixture-of-experts model beside a 16,000-token conversation: with an f16 cache only 15
    # of its 48 layers' experts fit on the GPU, but a q8_0 cache is half the size and leaves
    # room for all of them, which makes it the fastest.
    budget, cache_size = 18000, {"f16": 1.0, "q8_0": 34 / 64, "q4_0": 18 / 64}

    def memory_run(llama_bench, model, options, name=None):
        cache = options.kv_cache or "f16"
        on_gpu = 48 - (options.cpu_moe_layers or 0)
        if on_gpu * 90 + 480 + 16128 * cache_size[cache] > budget:
            raise BackendError("cudaMalloc failed", status="out_of_memory")
        run = example_run(llama_bench, model, options, name)
        run.settings.flash_attention = options.flash_attention
        run.settings.kv_cache_k = run.settings.kv_cache_v = cache
        run.settings.ubatch_size = options.ubatch_size
        run.settings.cpu_moe_layers = options.cpu_moe_layers
        run.speed.decode_tok_s = (
            100 - (options.cpu_moe_layers or 0) + (options.ubatch_size or 0) / 256
        )
        return run

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, memory_run)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=48, experts=128, expert_share=0.9, kv_bytes_per_token=1)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + budget / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    result = runner.invoke(cli.app, ["tune", str(model), "--use", "documents"])
    assert result.exit_code == 0
    assert "Fastest split: experts of 33 of 48 layers in RAM." in result.output
    assert "Finding its split too." in result.output
    assert "Fastest split with a q8_0 KV cache: the whole model on the GPU." in result.output
    assert "experts of 33 of 48 layers in RAM" in result.output  # the table's split column
    assert "whole model on the GPU" in result.output
    assert "--use documents --gpu-layers 49 --cpu-moe 0 --flash-attn --kv-cache q8_0" in (
        result.output
    )


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


def test_paths_with_spaces_are_quoted_for_pasting(monkeypatch):
    monkeypatch.setattr(cli.sys, "platform", "win32")
    assert cli.quoted(r"C:\My Models\a.gguf") == r'"C:\My Models\a.gguf"'
    assert cli.quoted(r"C:\Program Files\llama-server.exe", program=True).startswith('& "')
    assert cli.quoted("llama-server", program=True) == "llama-server"


def test_sweep_benchmarks_each_model_for_each_use(monkeypatch, tmp_path):
    asked = []

    def remembers_options(llama_bench, model, options, name=None):
        asked.append(options)
        return example_run(llama_bench, model, options, name)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, remembers_options)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: None)
    result = runner.invoke(
        cli.app, ["sweep", str(model), "gone:7b", "--use", "chat", "--use", "code"]
    )
    assert result.exit_code == 0
    assert "Benchmarking 2 models for chat, code: 4 runs" in result.output
    assert "Skipping gone:7b" in result.output
    assert [(o.prompt_tokens, o.gen_tokens) for o in asked] == [(512, 256), (8000, 512)]
    assert "Sweep results" in result.output
    saved = load_results()
    assert len(saved) == 2 and saved[0].sweep_id and saved[0].sweep_id == saved[1].sweep_id


def test_sweep_takes_every_ollama_model_when_none_are_named(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "downloaded_models", lambda: ["qwen3:8b"])
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: model)
    result = runner.invoke(cli.app, ["sweep", "--use", "chat"])
    assert "Benchmarking 1 model for chat" in result.output
    assert load_results()[0].settings.model == "qwen3:8b"


def test_sweep_without_any_models(monkeypatch, tmp_path):
    fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "downloaded_models", lambda: [])
    result = runner.invoke(cli.app, ["sweep"])
    assert result.exit_code == 1
    assert "ollama pull" in result.output


def test_sweep_starts_a_model_too_big_for_the_gpu_from_a_split(monkeypatch, tmp_path):
    asked = []

    def remembers_options(llama_bench, model, options, name=None):
        asked.append(options)
        return example_run(llama_bench, model, options, name)

    model = fake_run(monkeypatch, tmp_path, remembers_options)
    model.write_bytes(b"x" * 4800)
    layout = ModelLayout(layers=48, experts=128, expert_share=0.9)
    monkeypatch.setattr(cli, "read_layout", lambda path: layout)
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 2400 / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    runner.invoke(cli.app, ["sweep", str(model), "--use", "chat"])
    assert (asked[0].gpu_layers, asked[0].cpu_moe_layers) == (49, 27)


def test_sweep_keeps_the_finished_runs_when_stopped(monkeypatch, tmp_path):
    calls = itertools.count()

    def stopped_on_the_second(llama_bench, model, options, name=None):
        if next(calls) == 1:
            raise KeyboardInterrupt
        return example_run(llama_bench, model, options, name)

    model = fake_run(monkeypatch, tmp_path, stopped_on_the_second)
    result = runner.invoke(cli.app, ["sweep", str(model)])
    assert "Stopped" in result.output
    assert len(load_results()) == 1


def test_a_stopped_sweep_resumes_where_it_stopped(monkeypatch, tmp_path):
    calls = itertools.count()
    asked = []

    def stopped_on_the_second(llama_bench, model, options, name=None):
        if next(calls) == 1:
            raise KeyboardInterrupt
        asked.append(options.prompt_tokens)
        return example_run(llama_bench, model, options, name)

    model = fake_run(monkeypatch, tmp_path, stopped_on_the_second)
    runner.invoke(cli.app, ["sweep", str(model)])
    result = runner.invoke(cli.app, ["sweep", str(model), "--resume"])
    assert "1 model and use pairs are done" in result.output
    assert "2 runs" in result.output
    assert asked == [512, 8000, 16000]
    saved = load_results()
    assert [run.use for run in saved] == ["chat", "code", "documents"]
    assert len({run.sweep_id for run in saved}) == 1


def test_resume_without_an_earlier_sweep_starts_a_new_one(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["sweep", str(model), "--use", "chat", "--resume"])
    assert "no earlier sweep to resume" in result.output
    assert len(load_results()) == 1


def test_a_resumed_tuning_sweep_redoes_the_session_that_was_cut_off(monkeypatch, tmp_path):
    calls = itertools.count()

    def stopped_in_the_second_session(llama_bench, model, options, name=None):
        if next(calls) == 20:
            raise KeyboardInterrupt
        return tuned_run(llama_bench, model, options, name)

    model = fake_run(monkeypatch, tmp_path, stopped_in_the_second_session)
    sweep = ["sweep", str(model), "--use", "chat", "--use", "code", "--tune"]
    runner.invoke(cli.app, sweep)
    result = runner.invoke(cli.app, [*sweep, "--resume"])
    assert "Session 1 of 1" in result.output
    saved = load_results()
    assert len({run.sweep_id for run in saved}) == 1
    assert [run.use for run in saved].count("chat") == 16


def test_runs_saved_before_uses_were_kept_are_matched_by_length():
    settings = RunSettings(engine="llama.cpp", model="qwen3:14b", context_size=8000 + 512)
    run = RunResult.model_construct(use=None, settings=settings)
    assert cli.use_of(run) == "code"


def test_sweep_can_tune_every_model_for_every_use(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, tuned_run)
    result = runner.invoke(
        cli.app, ["sweep", str(model), "--use", "chat", "--use", "code", "--tune"]
    )
    assert result.exit_code == 0
    assert "Tuning 1 model for chat, code: 2 tuning sessions" in result.output
    assert "Session 2 of 2" in result.output
    assert "Fastest settings for each model and use" in result.output
    assert "flash attn on, KV q8_0, micro-batch 2048" in result.output
    assert "+15%" in result.output
    saved = load_results()
    assert len(saved) == 16 + 13  # code's long prompt tries flash attention off only once
    assert len({run.sweep_id for run in saved}) == 1 and saved[0].sweep_id
    assert len({run.tune_id for run in saved}) == 2


def test_a_split_counts_layers_out_of_the_models_own():
    # llama.cpp counts the output layer as one more, so 66 of a 65-layer model is all of it.
    assert cli.describe_placement({"gpu_layers": 66}, layers=65) == "whole model on the GPU"
    assert cli.describe_placement({"gpu_layers": 40}, layers=65) == "40 of 65 layers on the GPU"
    assert cli.describe_placement({"gpu_layers": 40}) == "40 layers on the GPU"
    moe = {"gpu_layers": 49, "cpu_moe_layers": 6}
    assert cli.describe_placement(moe, layers=48) == "experts of 6 of 48 layers in RAM"


def test_a_tuning_sweep_moves_on_when_nothing_fits(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options, name=None):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    model.write_bytes(b"x" * 4800)
    monkeypatch.setattr(
        cli, "read_layout", lambda path: ModelLayout(layers=48, experts=128, expert_share=0.9)
    )
    small_gpu = FAKE_PC.gpus[0].model_copy(update={"vram_gb": RESERVED_GB + 2400 / 1024**3})
    monkeypatch.setattr(
        cli, "detect_hardware", lambda: FAKE_PC.model_copy(update={"gpus": [small_gpu]})
    )
    result = runner.invoke(
        cli.app, ["sweep", str(model), "--use", "chat", "--use", "code", "--tune"]
    )
    assert result.exit_code == 0
    assert result.output.count("moving on") == 2
    assert "nothing worked" in result.output


def test_localtune_on_its_own_lists_the_commands():
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert "sweep" in result.output and "LOCALTUNE_NO_ANIMATION" in result.output


def test_run_shows_the_estimate_beside_the_measured_speed(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(base, "detect_hardware", lambda: FAKE_PC)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: tmp_path / "sha256-abc")
    result = runner.invoke(cli.app, ["run", "qwen3:14b", "--price", "0.20"])
    assert "Decode estimate" in result.output
    assert "from 'localtune predict'" in result.output


def test_results_saves_every_run_as_a_csv_row(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(base, "detect_hardware", lambda: FAKE_PC)
    runner.invoke(cli.app, ["run", str(model), "--price", "0.20"])
    runner.invoke(cli.app, ["run", str(model), "--price", "0.20"])
    path = tmp_path / "runs.csv"
    result = runner.invoke(cli.app, ["results", "--csv", str(path)])
    assert "Saved 2 runs" in result.output
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["speed.decode_tok_s"] == "133.2"
    assert rows[0]["hardware.gpu.name"] == "NVIDIA GeForce RTX 4070 Ti SUPER"
    assert "estimate.decode_tok_s" in rows[0]


def fake_quality(monkeypatch, tmp_path, perplexities):
    # 'localtune quality' with a pretend llama-perplexity giving each cache type a perplexity.
    model = fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "get_text", lambda: tmp_path / "wiki.test.raw")

    def measured(tool, model_file, text, cache, placement, base, gpu, chunks, context):
        base = base | {"context_size": context}
        if perplexities[cache] is None:
            return cli.QualityResult(
                kv_cache=cache, model_file=model_file.name, status="out_of_memory",
                error="cudaMalloc failed", **base,
            )  # fmt: skip
        return cli.QualityResult(
            kv_cache=cache, model_file=model_file.name, perplexity=perplexities[cache],
            perplexity_error=0.04, **base,
        )  # fmt: skip

    monkeypatch.setattr(cli, "measure_quality", measured)
    return model


def test_quality_compares_each_kv_cache_with_f16(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_quality(monkeypatch, tmp_path, {"f16": 6.0, "q8_0": 6.012, "q4_0": 6.24})
    result = runner.invoke(cli.app, ["quality", str(model)])
    assert result.exit_code == 0
    assert "Perplexity is how surprised the model was" in result.output
    assert "+4.0%" in result.output
    assert "A q8_0 KV cache: 0.2% worse, too small to notice." in result.output
    assert "A q4_0 KV cache: 4.0% worse, noticeable." in result.output
    saved = (tmp_path / "home" / "quality.jsonl").read_text().splitlines()
    assert len(saved) == 3


def test_quality_lists_a_cache_type_that_failed(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_quality(monkeypatch, tmp_path, {"f16": None, "q8_0": 6.0, "q4_0": 6.1})
    result = runner.invoke(cli.app, ["quality", str(model)])
    assert "out of GPU memory" in result.output
    assert "KV cache: " not in result.output
    assert "f16 failed: cudaMalloc failed" in result.output


def test_quality_without_llama_cpp(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    monkeypatch.setattr(cli, "find_tool", lambda name: None)
    result = runner.invoke(cli.app, ["quality", str(model)])
    assert result.exit_code == 1
    assert "localtune doctor" in result.output


def test_quality_reads_chunks_as_long_as_a_uses_conversation(monkeypatch, tmp_path):
    wide_terminal(monkeypatch)
    model = fake_quality(monkeypatch, tmp_path, {"f16": 6.0, "q8_0": 6.0, "q4_0": 6.3})
    result = runner.invoke(cli.app, ["quality", str(model), "--use", "documents"])
    assert "16,384-token chunks" in result.output
    result = runner.invoke(cli.app, ["quality", str(model), "--context", "8192"])
    assert "8,192-token chunks" in result.output
    result = runner.invoke(cli.app, ["quality", str(model), "--use", "poems"])
    assert result.exit_code == 1
