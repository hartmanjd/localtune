# Tests for the terminal command. They run the commands the way a person would, but with made-up
# hardware and GPU readings, so they check what gets shown without depending on the machine
# running them.

import itertools
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from localtune import cli
from localtune.doctor import Check
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import load_results
from localtune.runners.base import BackendError
from localtune.runners.llamacpp import parse_output

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
    monkeypatch.setattr(cli, "detect_hardware", lambda: FAKE_PC)
    result = runner.invoke(cli.app, ["info"])
    assert result.exit_code == 0
    assert "RTX 4070 Ti SUPER" in result.output
    assert "(WSL)" in result.output
    assert "32.0 GB" in result.output


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
    monkeypatch.setattr(cli, "choose_build", lambda: "cuda-13.4")
    monkeypatch.setattr(cli, "install_llama_cpp", fake_install)


def test_doctor_installs_llama_cpp_when_asked(monkeypatch):
    installed = []
    missing_llama_cpp(monkeypatch, installed)
    result = runner.invoke(cli.app, ["doctor", "--install"])
    assert installed == ["cuda-13.4"]
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


def example_run(llama_bench, model, options):
    return parse_output(EXAMPLE_RUN, model, options.repetitions)


def test_run_shows_and_saves_the_result(monkeypatch, tmp_path):
    model = fake_run(monkeypatch, tmp_path, example_run)
    result = runner.invoke(cli.app, ["run", str(model), "--price", "0.20"])
    assert result.exit_code == 0
    assert "7,612 tok/s" in result.output
    assert "133.2 tok/s" in result.output
    assert "Saved to" in result.output
    assert load_results()[0].speed.decode_tok_s == 133.2


def test_run_saves_a_failed_run(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options):
        raise BackendError("cudaMalloc failed", status="out_of_memory")

    model = fake_run(monkeypatch, tmp_path, out_of_memory)
    result = runner.invoke(cli.app, ["run", str(model)])
    assert result.exit_code == 1
    assert "ran out of GPU memory" in result.output
    assert load_results()[0].status == "out_of_memory"


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


def test_results_shows_automatic_gpu_layers(monkeypatch, tmp_path):
    # llama.cpp reports -1 GPU layers when it picks the number itself.
    def automatic_layers(llama_bench, model, options):
        output = EXAMPLE_RUN.replace('"n_gpu_layers": 99', '"n_gpu_layers": -1')
        return parse_output(output, model, options.repetitions)

    wide_terminal(monkeypatch)
    model = fake_run(monkeypatch, tmp_path, automatic_layers)
    runner.invoke(cli.app, ["run", str(model)])
    result = runner.invoke(cli.app, ["results"])
    assert "GPU layers auto" in result.output


def test_results_lists_failed_runs(monkeypatch, tmp_path):
    def out_of_memory(llama_bench, model, options):
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
