"""The localtune terminal command."""

# Everything a person types in the terminal ("localtune info", "localtune doctor",
# "localtune run", "localtune tune", "localtune results" and "localtune monitor") starts here.
# This file only handles the conversation with the person: it asks other modules for the
# actual work (hardware.py for the computer's details, gpu_monitor.py, cost.py and prices.py
# for GPU readings and their cost, doctor.py for whether the computer is ready, runners/ for
# benchmarks, tune.py for the settings tuning tries, uses.py for the prompt lengths of a use)
# and shows the answers.

import sys
import time
from functools import partial
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from localtune import __version__
from localtune.cost import energy_cost, hourly_cost
from localtune.doctor import is_ready, needs_llama_cpp, run_checks
from localtune.gguf import GgufError, read_layout
from localtune.gpu_monitor import GpuMonitor, busy_reason, other_use, read_sample
from localtune.gpus import find_gpu_spec
from localtune.hardware import detect_hardware, find_nvidia_smi
from localtune.install import (
    DOWNLOAD_MB,
    InstallError,
    choose_build,
    find_tool,
    install_llama_cpp,
)
from localtune.ollama_models import find_ollama_model
from localtune.prices import find_price
from localtune.results import RunResult, RunSettings, load_results, results_file, save_result
from localtune.runners.base import measure
from localtune.runners.llamacpp import (
    KV_CACHE_TYPES,
    BenchOptions,
    requested_settings,
    run_llama_bench,
)
from localtune.runners.web_api import ApiOptions, find_server, pick_model, run_web_api
from localtune.runners.web_api import requested_settings as web_api_settings
from localtune.tune import (
    DEFAULTS,
    NOISE,
    answer_seconds,
    as_good_as,
    combinations,
    default_run,
    describe,
    describe_split,
    fastest,
    find_best_split,
    first_guess,
    run_flags,
    server_context,
    server_flags,
    split_settings,
)
from localtune.uses import UnknownUseError, token_lengths

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """Benchmark local LLMs on your own hardware."""


# "localtune info": shows what localtune can see about this computer, the same details that
# will be saved with every benchmark run, plus each GPU's specs from the GPU table (gpus.py).


@app.command()
def info() -> None:
    """Show this computer's OS, CPU, RAM and GPUs."""
    hw = detect_hardware()

    system = Table(title=f"localtune {__version__}", show_header=False)
    system.add_row("OS", hw.os + (" (WSL)" if hw.is_wsl else ""))
    system.add_row("CPU", hw.cpu_name or "unknown")
    system.add_row("CPU threads", str(hw.cpu_threads or "unknown"))
    system.add_row("RAM", f"{hw.ram_gb} GB" if hw.ram_gb else "unknown")
    console.print(system)

    if not hw.gpus:
        console.print("No NVIDIA GPU found (nvidia-smi not available).")
        return

    gpus = Table(title="GPUs")
    gpus.add_column("#")
    gpus.add_column("Name")
    gpus.add_column("VRAM")
    gpus.add_column("Bandwidth")
    gpus.add_column("Compute")
    gpus.add_column("Driver")
    for gpu in hw.gpus:
        spec = find_gpu_spec(gpu.name, gpu.vram_gb)
        gpus.add_row(
            str(gpu.index),
            gpu.name,
            f"{gpu.vram_gb} GB",
            f"{spec.memory_bandwidth_gbs:,.0f} GB/s" if spec else "unknown",
            f"{spec.fp32_tflops} TFLOPS" if spec else "unknown",
            gpu.driver_version,
        )
    console.print(gpus)


# "localtune doctor": checks everything a benchmark needs and shows what to fix. When llama.cpp
# is missing on a computer with a working GPU, it offers to install it (install.py), then checks
# again. It ends with an error code when the computer isn't ready, so scripts (like a
# rented-GPU setup) can stop early instead of benchmarking on a broken setup.

STATUS_MARKS = {"ok": "[green]✓[/green]", "warn": "[yellow]![/yellow]", "fail": "[red]✗[/red]"}


@app.command()
def doctor(
    install: bool = typer.Option(
        False, "--install", help="Install llama.cpp without asking, if it's missing."
    ),
) -> None:
    """Check that this computer is ready to benchmark, and what to fix if not."""
    checks = run_checks()
    show_checks(checks)

    if needs_llama_cpp(checks) and offer_llama_cpp(ask=not install):
        checks = run_checks()
        show_checks(checks)

    if is_ready(checks):
        console.print("Ready to benchmark.")
    elif any(check.status == "fail" for check in checks):
        console.print("Not ready to benchmark: fix the items marked ✗.")
        raise typer.Exit(1)
    else:
        console.print("Not ready to benchmark: needs llama.cpp or a running local AI server.")
        raise typer.Exit(1)


def show_checks(checks) -> None:
    table = Table(title="localtune doctor", show_header=False)
    for check in checks:
        detail = check.detail + (f"\n[dim]{check.fix}[/dim]" if check.fix else "")
        table.add_row(STATUS_MARKS[check.status], check.name, detail)
    console.print(table)


# Installing llama.cpp from doctor: says what will be downloaded and asks first (unless
# --install was given, or nobody is there to answer), then shows each download's progress.
# Returns whether llama.cpp got installed.


def can_ask() -> bool:
    return sys.stdin.isatty()


def offer_llama_cpp(ask: bool) -> bool:
    try:
        build = choose_build()
    except InstallError as error:
        console.print(f"llama.cpp can't be installed automatically: {error}")
        return False

    if ask:
        question = (
            f"llama.cpp isn't installed. Download and install it now "
            f"(about {DOWNLOAD_MB[build]} MB)?"
        )
        if not can_ask() or not typer.confirm(question, default=True):
            return False

    columns = (
        TextColumn("{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        tasks = {}

        def update(name: str, done: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = progress.add_task(f"Downloading {name}", total=total or None)
            progress.update(tasks[name], completed=done)

        try:
            folder = install_llama_cpp(build, update)
        except (InstallError, OSError) as error:
            console.print(f"Installing llama.cpp failed: {error}")
            return False
    console.print(f"Installed llama.cpp in {folder}")
    return True


# "localtune run": benchmarks a model and saves the result. With llama.cpp (the default) it
# benchmarks a model file; with "-b api" it benchmarks a model in a local AI server that's
# already running (runners/web_api.py), found by itself or at --url. While the engine runs, the
# GPU is watched and the energy priced (runners/base.py); afterwards the speeds, energy and cost
# are shown, and the run is added to the results file (results.py). A failed run is saved too,
# and ends with an error code.

BACKENDS = ("llama.cpp", "api")


@app.command()
def run(
    model: Annotated[
        str | None,
        typer.Argument(
            help="The model file (.gguf) or a model downloaded with Ollama, like qwen3:14b. "
            "With -b api, the model's name in the server."
        ),
    ] = None,
    backend: str = typer.Option(
        "llama.cpp", "--backend", "-b", help="llama.cpp, or api for a running local AI server."
    ),
    url: str | None = typer.Option(
        None, help="-b api: the server's address, like http://localhost:11434, if not found."
    ),
    use: str | None = typer.Option(
        None, help="What the model is for: chat, code or documents. Sets the lengths below."
    ),
    prompt_tokens: int | None = typer.Option(
        None, help="Length of the prompt the model reads (512 unless --use says otherwise)."
    ),
    gen_tokens: int | None = typer.Option(
        None, help="How many tokens the model writes (128 unless --use says otherwise)."
    ),
    repetitions: int = typer.Option(3, help="How many times each test is repeated."),
    gpu_layers: int | None = typer.Option(None, help="How many layers go on the GPU."),
    cpu_moe: int | None = typer.Option(
        None, help="Mixture-of-experts models: how many layers keep their experts in RAM."
    ),
    kv_cache: str | None = typer.Option(None, help="KV cache type, like f16, q8_0 or q4_0."),
    flash_attn: bool | None = typer.Option(
        None, "--flash-attn/--no-flash-attn", help="Turn flash attention on or off."
    ),
    threads: int | None = typer.Option(None, help="CPU threads to use."),
    batch_size: int | None = typer.Option(None, help="Prompt batch size."),
    ubatch_size: int | None = typer.Option(None, help="Prompt micro-batch size."),
    gpu: int = typer.Option(0, help="Which GPU to use, by its # from 'localtune info'."),
    zip_code: str | None = typer.Option(
        None, "--zip", help="Your US zip code, for your local electricity price."
    ),
    state: str | None = typer.Option(None, help="Your US state, like OH, if no zip is given."),
    country: str | None = typer.Option(None, help="Your country, like DE, outside the US."),
    price: float | None = typer.Option(
        None, help="Your own electricity price per kWh, from your bill. Overrides the rest."
    ),
) -> None:
    """Benchmark a model with llama.cpp or a local AI server, and save the result."""
    prompt_tokens, gen_tokens = lengths_for(use, prompt_tokens, gen_tokens)
    if backend not in BACKENDS:
        console.print(f"Unknown backend {backend}. Choose from: {', '.join(BACKENDS)}")
        raise typer.Exit(1)

    if backend == "api":
        engine_only = {
            "--gpu-layers": gpu_layers,
            "--cpu-moe": cpu_moe,
            "--kv-cache": kv_cache,
            "--flash-attn": flash_attn,
            "--threads": threads,
            "--batch-size": batch_size,
            "--ubatch-size": ubatch_size,
        }
        given = [flag for flag, value in engine_only.items() if value is not None]
        if given:
            console.print(
                f"{', '.join(given)} only work with llama.cpp. A server keeps the settings "
                "it was started with."
            )
            raise typer.Exit(1)
        options = ApiOptions(
            prompt_tokens=prompt_tokens, gen_tokens=gen_tokens, repetitions=repetitions
        )
        server, model_name = choose_server_and_model(url, model)
        engine = partial(run_web_api, server, model_name, options)
        requested = web_api_settings(server, model_name, options)
        label = f"Benchmarking {model_name} in {server.name} at {server.url}..."
    else:
        llama_bench, model_file, model_name = prepare_llama_cpp(model, kv_cache)
        warn_if_gpu_busy(gpu)
        options = BenchOptions(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            repetitions=repetitions,
            gpu_layers=gpu_layers,
            cpu_moe_layers=cpu_moe,
            kv_cache=kv_cache,
            flash_attention=flash_attn,
            threads=threads,
            batch_size=batch_size,
            ubatch_size=ubatch_size,
            gpu_index=gpu,
        )
        engine = partial(run_llama_bench, llama_bench, model_file, options, model_name)
        requested = requested_settings(model_file, options, model_name)
        label = f"Benchmarking {model_name or model_file.name} with llama.cpp..."

    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)
    with console.status(label):
        result = measure(engine, requested, electricity, repetitions, gpu_index=gpu)
    path = save_result(result)

    if result.status != "ok":
        reason = "ran out of GPU memory" if result.status == "out_of_memory" else "failed"
        console.print(f"The benchmark {reason}: {result.error}")
        console.print(f"Saved the failed run to {path}")
        raise typer.Exit(1)
    show_run(result)
    console.print(f"Saved to {path}")


# --use picks the prompt and answer lengths for a kind of use (uses.py); --prompt-tokens and
# --gen-tokens still win when given.


def lengths_for(use: str | None, prompt_tokens: int | None, gen_tokens: int | None):
    try:
        return token_lengths(use, prompt_tokens, gen_tokens)
    except UnknownUseError as error:
        console.print(str(error))
        raise typer.Exit(1) from None


# Before llama.cpp benchmarks anything, a quick look at whether another program is already using
# the GPU (gpu_monitor.py), since that makes every result slower than it should be. It only
# warns: the person may have a good reason, and the run still goes ahead.


def warn_if_gpu_busy(gpu: int) -> None:
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return
    reason = busy_reason(other_use(lambda: read_sample(nvidia_smi, gpu)))
    if reason:
        console.print(
            f"[yellow]Heads-up: {reason}.[/yellow] Another program is using the GPU (like a "
            "game, a video or video call, or Ollama keeping a model loaded), so these results "
            "may be slower than they should be. For trustworthy numbers, close it first. "
            "'ollama ps' shows a model Ollama has loaded, and 'ollama stop MODEL' frees it."
        )


# For llama.cpp: the model is a file, or a model Ollama downloaded (ollama_models.py), which
# is then benchmarked from Ollama's copy and saved under its Ollama name. Before anything runs,
# llama.cpp, the model and the settings are checked, so a mistake is explained straight away.


def prepare_llama_cpp(model: str | None, kv_cache: str | None = None):
    llama_bench = find_tool("llama-bench")
    if llama_bench is None:
        console.print("llama.cpp isn't installed. Run 'localtune doctor' to install it.")
        raise typer.Exit(1)
    model_file, model_name = find_model_file(model)
    if model_file is None:
        explain_missing_model(model)
        raise typer.Exit(1)
    if kv_cache is not None and kv_cache not in KV_CACHE_TYPES:
        console.print(f"Unknown KV cache type {kv_cache}. Choose from: {', '.join(KV_CACHE_TYPES)}")
        raise typer.Exit(1)
    return llama_bench, model_file, model_name


def find_model_file(model: str | None) -> tuple[Path | None, str | None]:
    if model is None:
        return None, None
    if Path(model).is_file():
        return Path(model), None
    ollama_file = find_ollama_model(model)
    return (ollama_file, model) if ollama_file else (None, None)


def explain_missing_model(model: str | None) -> None:
    if model is None:
        console.print(
            "Name a model file (.gguf), or a model downloaded with Ollama, like qwen3:14b."
        )
    elif model.lower().endswith(".gguf"):
        console.print(f"Model file not found: {model}")
    else:
        console.print(f"{model} isn't a model file or a model downloaded with Ollama.")
        console.print("If it's an Ollama model, download it first, then run this again:")
        console.print(f"  ollama pull {model}", markup=False, highlight=False)


# For "-b api": finds the running server and the model in it to benchmark, or says what's
# missing. When the server has several models and none was named, it lists them to choose from.


def choose_server_and_model(url: str | None, wanted: str | None):
    server = find_server(url)
    if server is None:
        where = url or "the usual ports"
        console.print(
            f"No local AI server found at {where}. Start Ollama, llama-server or LM Studio, "
            "or give its address with --url."
        )
        raise typer.Exit(1)
    model = pick_model(server.models, wanted)
    if model is None:
        if wanted:
            console.print(f"{server.name} doesn't have a model called {wanted}.")
        else:
            console.print(f"{server.name} has several models. Name the one to benchmark.")
        console.print("Its models: " + (", ".join(server.models) or "none"))
        raise typer.Exit(1)
    return server, model


def show_run(result: RunResult) -> None:
    speed, gpu, energy = result.speed, result.gpu, result.energy
    unit = energy.price.currency
    settings = result.settings
    title = settings.model + (f" ({settings.quant})" if settings.quant else "")
    table = Table(title=title, show_header=False)
    table.add_row("Engine", " ".join(filter(None, [settings.engine, settings.engine_version])))
    if speed.prefill_tok_s is not None:
        table.add_row("Prefill", f"{speed.prefill_tok_s:,.0f} tok/s")
    if speed.decode_tok_s is not None:
        table.add_row("Decode", f"{speed.decode_tok_s:,.1f} tok/s")
    if speed.ttft_ms is not None:
        table.add_row("Time to first token", f"{speed.ttft_ms:,.0f} ms")
    if gpu is not None:
        table.add_row("Peak VRAM", f"{gpu.peak_vram_gb:.1f} GB")
        if gpu.avg_power_w is not None:
            table.add_row("Power", f"{gpu.avg_power_w:.0f} W avg, {gpu.peak_power_w:.0f} W peak")
    if energy.kwh is not None:
        table.add_row(
            "Energy",
            f"{energy.kwh * 1000:.2f} Wh ({energy.kwh_per_million_tokens:.3f} kWh per 1M tokens)",
        )
        table.add_row(
            "Cost",
            f"{energy.cost:.5f} {unit} ({energy.cost_per_million_tokens:.3f} {unit} per 1M tokens)",
        )
        table.add_row("Price used", f"{energy.price.per_kwh} {unit}/kWh ({energy.price.source})")
    else:
        table.add_row("Energy and cost", "unknown (no GPU power readings)")
    console.print(table)


# "localtune tune": benchmarks one model with every combination of settings in tune.py, one
# run after another, and shows which was fastest on this computer, with the options that use
# it in "localtune run". Each run is saved like a "localtune run" (with the same tune_id), so
# the whole session becomes data. It says how long is left, estimated from the runs so far,
# and Ctrl+C stops it with the finished runs kept. Settings that aren't tuned, like how many
# layers go on the GPU, can be fixed for every run, which is how a model too big for the GPU
# gets tuned.


@app.command()
def tune(
    model: Annotated[
        str | None,
        typer.Argument(help="The model file (.gguf) or a model downloaded with Ollama."),
    ] = None,
    use: str | None = typer.Option(
        None, help="What the model is for: chat, code or documents. Sets the lengths below."
    ),
    prompt_tokens: int | None = typer.Option(
        None, help="Length of the prompt the model reads (512 unless --use says otherwise)."
    ),
    gen_tokens: int | None = typer.Option(
        None, help="How many tokens the model writes (128 unless --use says otherwise)."
    ),
    repetitions: int = typer.Option(3, help="How many times each test is repeated."),
    gpu_layers: int | None = typer.Option(None, help="How many layers go on the GPU."),
    cpu_moe: int | None = typer.Option(
        None, help="Mixture-of-experts models: how many layers keep their experts in RAM."
    ),
    threads: int | None = typer.Option(None, help="CPU threads to use."),
    gpu: int = typer.Option(0, help="Which GPU to use, by its # from 'localtune info'."),
    zip_code: str | None = typer.Option(
        None, "--zip", help="Your US zip code, for your local electricity price."
    ),
    state: str | None = typer.Option(None, help="Your US state, like OH, if no zip is given."),
    country: str | None = typer.Option(None, help="Your country, like DE, outside the US."),
    price: float | None = typer.Option(
        None, help="Your own electricity price per kWh, from your bill. Overrides the rest."
    ),
) -> None:
    """Try llama.cpp's main settings on a model and find the fastest on this computer."""
    given_prompt, given_gen = prompt_tokens, gen_tokens
    prompt_tokens, gen_tokens = lengths_for(use, prompt_tokens, gen_tokens)
    llama_bench, model_file, model_name = prepare_llama_cpp(model)
    warn_if_gpu_busy(gpu)
    name = model_name or model_file.name
    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)
    tune_id, started = uuid4().hex, time.monotonic()
    placement = {"gpu_layers": gpu_layers, "cpu_moe_layers": cpu_moe}

    def bench(settings: dict, label: str) -> RunResult:
        options = BenchOptions(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            repetitions=repetitions,
            threads=threads,
            gpu_index=gpu,
            **(placement | settings),
        )
        engine = partial(run_llama_bench, llama_bench, model_file, options, model_name)
        requested = requested_settings(model_file, options, model_name)
        with console.status(label):
            result = measure(engine, requested, electricity, repetitions, gpu_index=gpu)
        result.tune_id = tune_id
        save_result(result)
        return result

    combos, runs = combinations(prompt_tokens), []
    try:
        if gpu_layers is None and cpu_moe is None:
            tokens = prompt_tokens + gen_tokens
            placement = find_split(model_file, gpu, bench, tokens) or placement
        console.print(f"Trying {len(combos)} combinations of settings on {name}, one at a time.")
        started = time.monotonic()
        for number, settings in enumerate(combos, 1):
            left = time_left(time.monotonic() - started, number - 1, len(combos))
            result = bench(settings, f"Run {number} of {len(combos)}: {describe(settings)}{left}")
            runs.append(result)
            console.print(f"{number:>3}. {describe(settings)}: {outcome(result)}")
    except KeyboardInterrupt:
        console.print("Stopped. The runs that finished are saved.")

    fixed = {
        "--use": use,
        "--prompt-tokens": given_prompt,
        "--gen-tokens": given_gen,
        "--gpu-layers": placement["gpu_layers"],
        "--cpu-moe": placement.get("cpu_moe_layers"),
        "--threads": threads,
    }
    fixed_flags = "".join(f" {flag} {value}" for flag, value in fixed.items() if value is not None)
    best = show_tune(runs, f"{model}{fixed_flags}", planned=len(combos))
    if best is None:
        raise typer.Exit(1)
    show_server_command(best, model_file, placement, prompt_tokens + gen_tokens, threads, gpu)
    console.print(f"All runs saved to {results_file()}")


# The winner as a llama-server command (tune.py), for chatting with the model at these settings
# in the browser, or from any app that talks to a local AI server. A path with spaces is quoted
# so it can be pasted as it is, in PowerShell as well.


def show_server_command(
    best: RunResult, model_file: Path, placement: dict, tokens: int, threads, gpu: int
) -> None:
    server = find_tool("llama-server") or "llama-server"
    flags = server_flags(best, placement, tokens, threads, gpu)
    command = f"{quoted(server, program=True)} -m {quoted(str(model_file))} {flags}"
    console.print("To chat with the model at these settings, start llama.cpp's server:")
    console.print(f"  {command}", markup=False, highlight=False, soft_wrap=True)
    console.print(
        f"Then open http://localhost:8080 in your browser. It has room for a "
        f"{server_context(tokens):,}-token conversation."
    )


def quoted(text: str, program: bool = False) -> str:
    if " " not in text:
        return text
    return f'& "{text}"' if program and sys.platform == "win32" else f'"{text}"'


# A model bigger than the GPU's memory is first split between the GPU and RAM (tune.py says
# how), at llama.cpp's default settings, so the settings tried after it are tried on the
# fastest split. Models that fit, files localtune can't read and computers without a GPU skip
# this step.


def find_split(model_file: Path, gpu: int, bench, tokens: int) -> dict | None:
    gpus = detect_hardware().gpus
    if gpu >= len(gpus):
        return None
    try:
        layout = read_layout(model_file)
    except (GgufError, OSError):
        return None
    size = model_file.stat().st_size
    guess = first_guess(size, layout, gpus[gpu].vram_gb, tokens)
    if guess is None:
        return None

    console.print(
        f"The model ({size / 1024**3:.1f} GB) and a {tokens:,}-token conversation don't fit "
        f"in the GPU's memory ({gpus[gpu].vram_gb:.0f} GB), so part of the model has to stay "
        "in RAM. First finding how much fits on the GPU."
    )

    def try_split(on_gpu: int) -> float | None:
        label = describe_split(on_gpu, layout)
        result = bench(split_settings(on_gpu, layout), f"Trying {label}")
        console.print(f"  Split: {label}: {outcome(result)}")
        return answer_seconds(result)

    best = find_best_split(try_split, guess, layout.layers)
    if best is None:
        console.print(
            "[red]No split fit in GPU memory.[/red] Try a smaller version of the model, "
            "like one with fewer parameters or a smaller quantization."
        )
        raise typer.Exit(1)
    console.print(f"Fastest split: {describe_split(best, layout)}.")
    return {"cpu_moe_layers": None} | split_settings(best, layout)


def time_left(elapsed_s: float, done: int, total: int) -> str:
    if done == 0:
        return ""
    minutes = round(elapsed_s / done * (total - done) / 60)
    return f" (about {minutes} min left)" if minutes >= 1 else " (under a minute left)"


def outcome(result: RunResult) -> str:
    if result.status == "out_of_memory":
        return "[red]out of GPU memory[/red]"
    if result.status != "ok" or not result.speed:
        return f"[red]failed[/red] ({result.error})"
    speed = result.speed
    seconds = answer_seconds(result)
    answer = f"{seconds:,.1f} s per answer, " if seconds is not None else ""
    return f"{answer}{speed.decode_tok_s:,.1f} tok/s decode, {speed.prefill_tok_s:,.0f} prefill"


def show_tune(runs: list[RunResult], model_and_fixed_flags: str, planned: int) -> RunResult | None:
    best = fastest(runs)
    if best is None:
        if runs:
            console.print("None of the settings worked.")
            if any(run.status == "out_of_memory" for run in runs):
                console.print(
                    "The model didn't fit in GPU memory. Try again with fewer layers on the GPU, "
                    "like --gpu-layers 20."
                )
        return None

    worked = sorted((run for run in runs if answer_seconds(run) is not None), key=answer_seconds)
    table = Table(title=f"{best.settings.model}, fastest first")
    for column in ("Flash attn", "KV cache", "Micro-batch"):
        table.add_column(column)
    for column in ("Answer time", "Decode tok/s", "Prefill tok/s", "Peak VRAM"):
        table.add_column(column, justify="right")
    for run in worked:
        s, gpu = run.settings, run.gpu
        table.add_row(
            "on" if s.flash_attention else "off",
            s.kv_cache_k or "",
            str(s.ubatch_size or ""),
            f"{answer_seconds(run):,.1f} s",
            f"{run.speed.decode_tok_s:,.1f}",
            f"{run.speed.prefill_tok_s:,.0f}",
            f"{gpu.peak_vram_gb:.1f} GB" if gpu and gpu.peak_vram_gb is not None else "",
        )
    console.print(table)

    s = best.settings
    winner = {"flash_attention": s.flash_attention, "kv_cache": s.kv_cache_k}
    winner["ubatch_size"] = s.ubatch_size
    if len(runs) < planned:
        console.print(f"Fastest so far: {describe(winner)}.")
        console.print(
            f"Only {len(runs)} of {planned} combinations were tried, so a faster one may be "
            "among the rest."
        )
    else:
        console.print(f"Fastest on this computer: {describe(winner)}.")
    show_trust(best, worked, runs)
    console.print("To use these settings:")
    command = f"  localtune run {model_and_fixed_flags} {run_flags(best)}"
    console.print(command, markup=False, highlight=False, soft_wrap=True)
    if s.kv_cache_k in ("q8_0", "q4_0"):
        console.print(
            f"A {s.kv_cache_k} KV cache saves memory but can make answers slightly worse than f16."
        )
    return best


# How much the winner can be trusted (tune.py): settings within measurement noise of it are just
# as good, and the gain only counts against llama.cpp's usual settings.


def show_trust(best: RunResult, worked: list[RunResult], runs: list[RunResult]) -> None:
    percent = f"{NOISE:.0%}"
    ties = as_good_as(best, worked)
    if ties:
        others = "1 other setting is" if len(ties) == 1 else f"{len(ties)} other settings are"
        console.print(
            f"{others} within {percent} of it, about the difference between two runs of the "
            "same settings, so they are just as good."
        )
    usual, best_s = default_run(runs), answer_seconds(best)
    usual_s = answer_seconds(usual) if usual else None
    if usual is best:
        console.print("These are llama.cpp's usual settings, so there's nothing to change.")
    elif usual_s is not None and usual_s <= best_s * (1 + NOISE):
        console.print(
            f"Tuned is within {percent} of llama.cpp's usual settings ({describe(DEFAULTS)}), "
            "so tuning these made no real difference."
        )
    elif usual_s is not None:
        console.print(
            f"Tuned is {usual_s / best_s - 1:.0%} faster than llama.cpp's usual settings "
            f"({best_s:,.1f} s per answer instead of {usual_s:,.1f} s)."
        )


# "localtune results": shows the runs saved by "localtune run" (results.py) as one table, so
# runs with different models, engines and settings can be compared side by side. Failed runs
# are listed too, because knowing which settings don't work is part of tuning.


@app.command()
def results(
    last: int = typer.Option(20, help="How many of the most recent runs to show."),
    model: str | None = typer.Option(None, help="Only show runs whose model name contains this."),
) -> None:
    """Show your saved benchmark runs."""
    runs = load_results()
    if model:
        runs = [r for r in runs if model.lower() in r.settings.model.lower()]
    if not runs:
        console.print("No saved runs yet. Benchmark a model with 'localtune run'.")
        return
    shown = runs[-last:]

    table = Table(title=f"Saved runs ({len(shown)} of {len(runs)})")
    table.add_column("When", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Engine")
    for column in ("Quant", "Settings", "Result"):
        table.add_column(column)
    for column in ("Prefill tok/s", "Decode tok/s", "Peak VRAM", "Cost/1M tok"):
        table.add_column(column, justify="right")
    for run in shown:
        table.add_row(*result_row(run))
    console.print(table)
    console.print(f"From {results_file()}")


RESULT_LABELS = {
    "ok": "ok",
    "out_of_memory": "[red]out of memory[/red]",
    "crashed": "[red]crashed[/red]",
}


def result_row(run: RunResult) -> list[str]:
    speed, gpu, energy = run.speed, run.gpu, run.energy
    cost = energy.cost_per_million_tokens
    return [
        run.created_at.astimezone().strftime("%m-%d %H:%M"),
        run.settings.model,
        run.settings.engine,
        run.settings.quant or "",
        settings_summary(run.settings),
        RESULT_LABELS[run.status],
        f"{speed.prefill_tok_s:,.0f}" if speed and speed.prefill_tok_s else "",
        f"{speed.decode_tok_s:,.1f}" if speed and speed.decode_tok_s else "",
        f"{gpu.peak_vram_gb:.1f} GB" if gpu and gpu.peak_vram_gb is not None else "",
        f"{cost:.3f} {energy.price.currency}" if cost is not None else "",
    ]


def settings_summary(settings: RunSettings) -> str:
    parts = []
    if settings.gpu_layers is not None:
        layers = "all" if settings.gpu_layers < 0 else settings.gpu_layers
        parts.append(f"GPU layers {layers}")
    if settings.cpu_moe_layers:
        parts.append(f"{settings.cpu_moe_layers} expert layers in RAM")
    if settings.kv_cache_k is not None:
        parts.append(f"KV {settings.kv_cache_k}")
    if settings.flash_attention is not None:
        parts.append("flash attn " + ("on" if settings.flash_attention else "off"))
    return ", ".join(parts) or "defaults"


# "localtune monitor": watches the GPU for a few seconds and shows what it used and what that
# cost, the same measurements that will be taken while every benchmark runs. Handy for seeing
# the readings work, or for watching the GPU while something else (a game, a model) is running.


@app.command()
def monitor(
    seconds: float = typer.Option(10, help="How long to watch the GPU."),
    zip_code: str | None = typer.Option(
        None, "--zip", help="Your US zip code, for your local electricity price."
    ),
    state: str | None = typer.Option(None, help="Your US state, like OH, if no zip is given."),
    country: str | None = typer.Option(None, help="Your country, like DE, outside the US."),
    price: float | None = typer.Option(
        None, help="Your own electricity price per kWh, from your bill. Overrides the rest."
    ),
    gpu: int = typer.Option(0, help="Which GPU to watch, by its # from 'localtune info'."),
) -> None:
    """Watch the GPU's power, memory, load and temperature, and what the power costs."""
    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)

    with GpuMonitor(gpu_index=gpu) as gpu_monitor:
        watch_with_countdown(gpu_monitor, gpu, seconds)
    summary = gpu_monitor.summary()

    if summary is None:
        console.print("No GPU readings (is an NVIDIA GPU and nvidia-smi available?).")
        raise typer.Exit(1)

    table = Table(title=f"GPU {gpu} over {summary.duration_s:.1f} s", show_header=False)
    if summary.avg_power_w is not None:
        table.add_row(
            "Power", f"{summary.avg_power_w:.0f} W avg, {summary.peak_power_w:.0f} W peak"
        )
    table.add_row(
        "VRAM",
        f"{summary.peak_vram_gb:.1f} GB peak ({summary.added_vram_gb:.1f} GB more than at start)",
    )
    table.add_row("Load", f"{summary.avg_utilization_pct:.0f}% avg")
    table.add_row("Temperature", f"{summary.peak_temperature_c:.0f} °C peak")
    if summary.energy_wh is not None:
        cost = energy_cost(summary.energy_wh, electricity)
        per_hour = hourly_cost(summary.avg_power_w, electricity)
        unit = electricity.currency
        table.add_row("Energy", f"{summary.energy_wh:.3f} Wh")
        table.add_row("Cost", f"{cost:.5f} {unit} ({per_hour:.3f} {unit} per hour at this power)")
        table.add_row("Price used", f"{electricity.per_kwh} {unit}/kWh ({electricity.source})")
    else:
        table.add_row("Energy and cost", "unknown (this GPU doesn't report power)")
    console.print(table)


# The live view while monitor runs: a bar that fills up as the seconds pass, the time left, and
# the GPU's latest power reading, so it's clear the watcher is working.


def watch_with_countdown(gpu_monitor: GpuMonitor, gpu: int, seconds: float) -> None:
    columns = (
        TextColumn(f"Watching GPU {gpu}"),
        BarColumn(),
        TextColumn("{task.fields[left]}"),
        TextColumn("{task.fields[power]}"),
    )
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task("watch", total=seconds, left="", power="")
        start = time.monotonic()
        while (elapsed := time.monotonic() - start) < seconds:
            latest = gpu_monitor.samples[-1] if gpu_monitor.samples else None
            power = f"{latest.power_w:.0f} W" if latest and latest.power_w is not None else ""
            left = f"{seconds - elapsed:.0f} s left"
            progress.update(task, completed=elapsed, left=left, power=power)
            time.sleep(0.1)
        progress.update(task, completed=seconds)
