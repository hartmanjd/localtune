"""The localtune terminal command."""

# Everything a person types in the terminal ("localtune info", "localtune doctor",
# "localtune predict", "localtune run", "localtune tune", "localtune sweep",
# "localtune results", "localtune share" and "localtune monitor") starts here.
# This file only handles the conversation with the person: it asks other modules for the
# actual work (hardware.py for the computer's details, gpu_monitor.py, cost.py and prices.py
# for GPU readings and their cost, doctor.py for whether the computer is ready, runners/ for
# benchmarks, predict.py for estimates before a model is downloaded, tune.py for the settings
# tuning tries, uses.py for the prompt lengths of a use, share.py for sharing runs) and shows the
# answers.

import csv
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
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
from localtune.animations import play_logo, working
from localtune.awake import keep_awake
from localtune.background import Background, busy_reasons, read_background
from localtune.cost import energy_cost, hourly_cost
from localtune.doctor import is_ready, needs_llama_cpp, run_checks
from localtune.gguf import GgufError, read_layout
from localtune.gpu_monitor import GpuMonitor, GpuSample, read_sample
from localtune.gpus import find_gpu_spec
from localtune.hardware import detect_hardware, find_nvidia_smi
from localtune.install import (
    DOWNLOAD_MB,
    InstallError,
    choose_build,
    find_tool,
    install_llama_cpp,
)
from localtune.meter import gpu_programs, is_busy, wait_for_room
from localtune.models import find_model
from localtune.ollama_models import downloaded_models, find_ollama_model
from localtune.predict import Prediction, estimate_for, predict
from localtune.prices import find_price
from localtune.quality import (
    CHUNKS,
    CONTEXT,
    KV_CACHES,
    TEXT_NAME,
    QualityResult,
    get_text,
    loss_pct,
    quality_file,
    save_quality,
    verdict,
)
from localtune.quality import measure as measure_quality
from localtune.results import (
    RunResult,
    RunSettings,
    flat_record,
    load_results,
    results_file,
    rig_id,
    save_result,
)
from localtune.runners.base import measure
from localtune.runners.llamacpp import (
    KV_CACHE_TYPES,
    BenchOptions,
    requested_settings,
    run_llama_bench,
)
from localtune.runners.web_api import ApiOptions, find_server, pick_model, run_web_api
from localtune.runners.web_api import requested_settings as web_api_settings
from localtune.share import (
    ShareError,
    mark_shared,
    not_yet_shared,
    payload,
    send,
    share_url,
    write_preview,
)
from localtune.tune import (
    DEFAULTS,
    KV_CACHE_SIZE,
    NOISE,
    SPILL_MARGIN_GB,
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
    too_full,
)
from localtune.uses import USES, UnknownUseError, token_lengths

app = typer.Typer()
console = Console()


# "localtune" on its own plays the logo (animations.py) and lists the commands. While the
# commands below work, a small loading animation runs beside what they're doing. The commands
# that benchmark keep the computer from going to sleep until they end (awake.py).

BENCHMARK_COMMANDS = {"run", "tune", "sweep"}


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Benchmark local LLMs on your own hardware.

    Set LOCALTUNE_NO_ANIMATION=1 to turn off the logo and loading animations.
    """
    if ctx.invoked_subcommand in BENCHMARK_COMMANDS:
        ctx.with_resource(keep_awake())
    if ctx.invoked_subcommand is None:
        play_logo(console)
        console.print(ctx.get_help(), markup=False, highlight=False)


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


# "localtune predict": estimates whether a model fits on this computer's GPU (or one named with
# --gpu-name) and how fast it will write, from the model and GPU tables alone (predict.py), so
# it works before the model is downloaded. It ends with how to measure the real speed.


@app.command("predict")
def predict_command(
    model: Annotated[str, typer.Argument(help="A model's Ollama name, like qwen3:14b.")],
    use: str | None = typer.Option(
        None, help="What the model is for: chat, code or documents. Longer uses need more memory."
    ),
    gpu: int = typer.Option(0, help="Which GPU to use, by its # from 'localtune info'."),
    gpu_name: str | None = typer.Option(
        None, help='Estimate for another GPU instead, like "RTX 4090".'
    ),
) -> None:
    """Estimate whether a model fits and how fast it will write, before downloading it."""
    prompt_tokens, gen_tokens = lengths_for(use, None, None)
    found = find_model(model)
    if found is None:
        console.print(
            f"{model} isn't in localtune's model table, which has the 100 most downloaded "
            "models on Ollama. Check the name on ollama.com/library.",
            markup=False,
        )
        raise typer.Exit(1)

    ram_gb = None
    if gpu_name:
        spec = find_gpu_spec(gpu_name)
        if spec is None:
            console.print(f"{gpu_name} isn't in localtune's GPU table.", markup=False)
            raise typer.Exit(1)
    else:
        hw = detect_hardware()
        if gpu >= len(hw.gpus):
            console.print(
                'No NVIDIA GPU found. Name one to estimate for, like --gpu-name "RTX 4090".'
            )
            raise typer.Exit(1)
        spec = find_gpu_spec(hw.gpus[gpu].name, hw.gpus[gpu].vram_gb)
        if spec is None:
            console.print(f"{hw.gpus[gpu].name} isn't in localtune's GPU table yet.", markup=False)
            raise typer.Exit(1)
        ram_gb = hw.ram_gb

    show_prediction(predict(*found, spec, prompt_tokens + gen_tokens, ram_gb), use)


def show_prediction(p: Prediction, use: str | None) -> None:
    table = Table(title=f"Estimate for {p.model} on {p.gpu}", show_header=False)
    table.add_row("Model file", f"{p.file_gb:.1f} GB ({p.quant})")
    table.add_row(f"Conversation ({use or 'short chat'})", f"{p.conversation_gb:.1f} GB")
    table.add_row("GPU memory", f"{p.vram_gb:g} GB, {p.memory_bandwidth_gbs:,.0f} GB/s")
    if p.fits_on_computer is False:
        fits = (
            f"[red]No[/red]: about {p.in_ram_gb:.1f} GB would have to go in RAM, more than is free"
        )
    elif p.in_ram_gb > 0:
        fits = f"Partly: about {p.in_ram_gb:.1f} GB goes in RAM, which is slower"
    else:
        fits = "Yes, all of it"
    table.add_row("Fits on the GPU", fits)
    if p.decode_tok_s is not None:
        table.add_row("Writing speed", f"about {p.decode_tok_s:,.0f} tok/s (estimate)")
    console.print(table)
    if p.decode_tok_s is None:
        return
    console.print(
        "This is an estimate from the GPU's memory speed and the model's size, not a "
        "measurement."
        + (" It assumes a good GPU/RAM split, which 'localtune tune' finds." if p.in_ram_gb else "")
    )
    console.print("To measure it for real:")
    if find_ollama_model(p.model) is None:
        console.print(f"  ollama pull {p.model}", markup=False, highlight=False)
    console.print(f"  localtune run {p.model}", markup=False, highlight=False)


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
    with working(console, "doctor", "Checking this computer..."):
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
        background = None  # the server holds its model on the GPU, so no warning
    else:
        llama_bench, model_file, model_name = prepare_llama_cpp(model, kv_cache)
        background = warn_if_busy(gpu)
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
    with working(console, "run", label):
        result = measure(
            engine, requested, electricity, repetitions, gpu_index=gpu, background=background
        )
    result.use = use
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


# Before llama.cpp benchmarks anything, a quick look at what else the computer is doing
# (background.py), since another program on the GPU or CPU makes every result slower than it
# should be. It only warns: the person may have a good reason, and the run still goes ahead.
# The same reading is saved with the run.


def warn_if_busy(gpu: int) -> Background:
    background = read_background(gpu)
    reasons = busy_reasons(background)
    if reasons:
        console.print(
            f"[yellow]Heads-up: {' and '.join(reasons)}.[/yellow] Another program is using the "
            "computer (like a game, a video or video call, or Ollama keeping a model loaded), "
            "so these results may be slower than they should be. For trustworthy numbers, "
            "close it first. 'ollama ps' shows a model Ollama has loaded, and 'ollama stop "
            "MODEL' frees it."
        )
        if show_memory_meter(gpu) == "free":
            console.print("[green]The GPU is free now.[/green]")
            background = read_background(gpu)
    return background


# When the GPU is the busy part and someone is watching the terminal, the live gauge of its
# memory (meter.py) lets them close programs and see the room come back before the run starts.


def show_memory_meter(gpu: int) -> str | None:
    if not (can_ask() and console.is_terminal):
        return None
    nvidia_smi, gpus = find_nvidia_smi(), detect_hardware().gpus
    if nvidia_smi is None or gpu >= len(gpus):
        return None

    def read() -> GpuSample | None:
        return read_sample(nvidia_smi, gpu)

    first = read()
    if first is None or not is_busy(first):
        return None
    return wait_for_room(read, gpus[gpu].vram_gb, console, gpu_programs(nvidia_smi, gpu))


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
        estimate = estimate_for(result)
        if estimate is not None:
            table.add_row("Decode estimate", f"{estimate:,.0f} tok/s (from 'localtune predict')")
    if speed.ttft_ms is not None:
        table.add_row("Time to first token", f"{speed.ttft_ms:,.0f} ms")
    if gpu is not None:
        table.add_row("Peak VRAM", f"{gpu.peak_vram_gb:.1f} GB")
        if gpu.avg_power_w is not None:
            power = f"{gpu.avg_power_w:.0f} W avg, {gpu.peak_power_w:.0f} W peak"
            idle = result.background.gpu_power_w if result.background else None
            if idle is not None:
                power += f" ({idle:.0f} W before the run)"
            table.add_row("Power", power)
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
    warn_if_busy(gpu)
    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)
    session = TuneSession(given={"gpu_layers": gpu_layers, "cpu_moe_layers": cpu_moe})
    try:
        tune_model(
            session,
            llama_bench,
            model_file,
            model_name,
            prompt_tokens,
            gen_tokens,
            repetitions,
            electricity,
            gpu,
            threads,
            use=use,
        )
    except KeyboardInterrupt:
        console.print("Stopped. The runs that finished are saved.")
    runs, placements = session.runs, session.placements

    placement_of = session.placement_of

    def command_for(run: RunResult) -> str:
        placement = placement_of(run)
        fixed = {
            "--use": use,
            "--prompt-tokens": given_prompt,
            "--gen-tokens": given_gen,
            "--gpu-layers": placement["gpu_layers"],
            "--cpu-moe": placement.get("cpu_moe_layers"),
            "--threads": threads,
        }
        flags = "".join(f" {flag} {value}" for flag, value in fixed.items() if value is not None)
        return f"{model}{flags}"

    def split_of(run: RunResult) -> str:
        return describe_placement(placement_of(run), session.layers)

    splits_differ = len({str(p) for p in placements.values() if p is not None}) > 1
    best = show_tune(
        runs, command_for, planned=session.planned, split_of=split_of if splits_differ else None
    )
    if best is None:
        raise typer.Exit(1)
    tokens = prompt_tokens + gen_tokens
    show_server_command(best, model_file, placement_of(best), tokens, threads, gpu)
    console.print(f"All runs saved to {results_file()}")


# One tuning session: finding the GPU/RAM split a model too big for the GPU needs, then
# trying every combination of settings (tune.py) on it, one run after another. "localtune tune"
# runs one session; "localtune sweep --tune" runs one per model and use. Each run is saved as
# soon as it finishes, with the session's tune_id (and the sweep's sweep_id), and lands in the
# session as it goes, so whatever finished is kept when Ctrl+C stops it. It also knows how many
# layers the model has, so a split can say "40 of 65 layers on the GPU" (llama.cpp counts the
# output layer as one more, so 66 of them means the whole model).


@dataclass
class TuneSession:
    given: dict
    runs: list[RunResult] = field(default_factory=list)
    planned: int = 0
    placements: dict[str, dict | None] = field(default_factory=dict)
    layers: int | None = None

    def placement_of(self, run: RunResult) -> dict:
        return self.placements.get(run.settings.kv_cache_k) or self.given


def describe_placement(placement: dict, layers: int | None = None) -> str:
    of = f" of {layers}" if layers else ""
    on_gpu, in_ram = placement.get("gpu_layers"), placement.get("cpu_moe_layers")
    if in_ram:
        return f"experts of {in_ram}{of} layers in RAM"
    if in_ram == 0 or on_gpu is None or on_gpu < 0 or (layers and on_gpu > layers):
        return "whole model on the GPU"
    return f"{on_gpu}{of} layers on the GPU"


def tune_model(
    session: TuneSession,
    llama_bench: str,
    model_file: Path,
    model_name: str | None,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    electricity,
    gpu: int,
    threads: int | None = None,
    sweep_id: str | None = None,
    use: str | None = None,
) -> None:
    tune_id = uuid4().hex
    session.placements = {cache: session.given for cache in KV_CACHE_SIZE}
    try:
        session.layers = read_layout(model_file).layers
    except (GgufError, OSError):
        session.layers = None

    def bench(
        settings: dict, label: str, placement: dict | None = None, progress: float | None = None
    ) -> RunResult:
        options = BenchOptions(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            repetitions=repetitions,
            threads=threads,
            gpu_index=gpu,
            **((placement or {}) | settings),
        )
        engine = partial(run_llama_bench, llama_bench, model_file, options, model_name)
        requested = requested_settings(model_file, options, model_name)
        with working(console, "tune", label, progress):
            result = measure(engine, requested, electricity, repetitions, gpu_index=gpu)
        result.tune_id, result.sweep_id, result.use = tune_id, sweep_id, use
        save_result(result)
        return result

    combos = combinations(prompt_tokens)
    session.planned = len(combos)
    if all(value is None for value in session.given.values()):
        tokens = prompt_tokens + gen_tokens
        session.placements = find_splits(model_file, gpu, bench, tokens) or session.placements
    combos = [c for c in combos if session.placements[c["kv_cache"]] is not None]
    session.planned = len(combos)
    name = model_name or model_file.name
    console.print(f"Trying {len(combos)} combinations of settings on {name}, one at a time.")
    started = time.monotonic()
    for number, settings in enumerate(combos, 1):
        left = time_left(time.monotonic() - started, number - 1, len(combos))
        label = f"Run {number} of {len(combos)}: {describe(settings)}{left}"
        progress = (number - 1) / len(combos)
        result = bench(settings, label, session.placements[settings["kv_cache"]], progress)
        session.runs.append(result)
        console.print(f"{number:>3}. {describe(settings)}: {outcome(result)}")


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
# how), so the settings tried after it are tried on the fastest split. Each KV cache type gets
# its own split: a smaller cache leaves room for more of the model on the GPU, which with a long
# conversation can be worth more than the cache costs. Cache types whose first guess is the same
# share one search, so a short conversation only needs one. Models that fit, files localtune
# can't read and computers without a GPU skip this step.


def find_splits(model_file: Path, gpu: int, bench, tokens: int) -> dict[str, dict | None] | None:
    gpus = detect_hardware().gpus
    if gpu >= len(gpus):
        return None
    try:
        layout = read_layout(model_file)
    except (GgufError, OSError):
        return None
    size, vram_gb = model_file.stat().st_size, gpus[gpu].vram_gb
    guesses = {cache: first_guess(size, layout, vram_gb, tokens, cache) for cache in KV_CACHE_SIZE}
    if guesses["f16"] is None:
        return None

    console.print(
        f"The model ({size / 1024**3:.1f} GB) and a {tokens:,}-token conversation don't fit "
        f"in the GPU's memory ({vram_gb:.0f} GB), so part of the model has to stay "
        "in RAM. First finding how much fits on the GPU."
    )

    def try_split(on_gpu: int, cache_settings: dict, cache: str) -> float | None:
        label = describe_split(on_gpu, layout) + ("" if cache == "f16" else f", KV {cache}")
        result = bench(split_settings(on_gpu, layout) | cache_settings, f"Trying {label}")
        console.print(f"  Split: {label}: {outcome(result)}")
        return None if too_full(result) else answer_seconds(result)

    splits: dict[str, dict | None] = {}
    searched: dict[int, dict | None] = {}
    for cache, guess in guesses.items():
        start = layout.layers if guess is None else guess
        if start in searched:
            splits[cache] = searched[start]
            continue
        cache_settings = {} if cache == "f16" else {"flash_attention": True, "kv_cache": cache}
        if cache != "f16":
            console.print(
                f"A {cache} KV cache takes less memory, so more of the model may fit beside it. "
                "Finding its split too."
            )
        best = find_best_split(
            partial(try_split, cache_settings=cache_settings, cache=cache), start, layout.layers
        )
        split = None if best is None else {"cpu_moe_layers": None} | split_settings(best, layout)
        if best is not None:
            cache_note = "" if cache == "f16" else f" with a {cache} KV cache"
            console.print(f"Fastest split{cache_note}: {describe_split(best, layout)}.")
        searched[start] = splits[cache] = split

    if all(split is None for split in splits.values()):
        console.print(
            "[red]No split fit in GPU memory.[/red] Try a smaller version of the model, "
            "like one with fewer parameters or a smaller quantization."
        )
        raise typer.Exit(1)
    return splits


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
    full = ", [yellow]GPU memory nearly full[/yellow]" if too_full(result) else ""
    return (
        f"{answer}{speed.decode_tok_s:,.1f} tok/s decode, {speed.prefill_tok_s:,.0f} prefill{full}"
    )


def show_tune(
    runs: list[RunResult],
    command_for: Callable[[RunResult], str],
    planned: int,
    split_of: Callable[[RunResult], str] | None = None,
) -> RunResult | None:
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
    for column in ("Flash attn", "KV cache", "Micro-batch") + (("Split",) if split_of else ()):
        table.add_column(column)
    for column in ("Answer time", "Decode tok/s", "Prefill tok/s", "Peak VRAM"):
        table.add_column(column, justify="right")
    for run in worked:
        s = run.settings
        table.add_row(
            "on" if s.flash_attention else "off",
            s.kv_cache_k or "",
            str(s.ubatch_size or ""),
            *([split_of(run)] if split_of else []),
            f"{answer_seconds(run):,.1f} s",
            f"{run.speed.decode_tok_s:,.1f}",
            f"{run.speed.prefill_tok_s:,.0f}",
            peak_vram(run),
        )
    console.print(table)
    if any(too_full(run) for run in worked):
        console.print(
            f"'full' means the run left less than {SPILL_MARGIN_GB:.0f} GB of the GPU's memory "
            "free. Those settings can slow down a lot when another program uses the GPU, so "
            "they only win when nothing else left room."
        )
    if too_full(best):
        console.print(
            "[yellow]Every setting left the GPU's memory nearly full.[/yellow] Closing other "
            "programs, or a smaller version of the model, leaves room to spare."
        )
    estimate = estimate_for(best)
    if estimate is not None:
        console.print(f"'localtune predict' estimated about {estimate:,.0f} tok/s decode.")

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
    command = f"  localtune run {command_for(best)} {run_flags(best)}"
    console.print(command, markup=False, highlight=False, soft_wrap=True)
    if s.kv_cache_k in ("q8_0", "q4_0"):
        console.print(
            f"A {s.kv_cache_k} KV cache saves memory but can make answers slightly worse than f16."
        )
    return best


def peak_vram(run: RunResult) -> str:
    gpu = run.gpu
    if gpu is None or gpu.peak_vram_gb is None:
        return ""
    return f"{gpu.peak_vram_gb:.1f} GB" + (" (full)" if too_full(run) else "")


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
    elif usual is not None and too_full(usual) and not too_full(best):
        console.print(
            "llama.cpp's usual settings left the GPU's memory nearly full, so these leave room "
            "for other programs instead."
        )
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


# "localtune sweep": benchmarks many models one after another, each for every chosen use
# (uses.py), so a computer left alone overnight turns into a lot of data. With no models named
# it takes every model downloaded with Ollama. Each run is llama.cpp's usual settings, like
# "localtune run" without options, except that a model too big for the GPU starts from tune's
# first guess at a GPU/RAM split (tune.py), so it doesn't crawl. The runs share a sweep_id and
# each is saved as soon as it finishes; Ctrl+C stops the sweep with the finished runs kept, and
# a model that fails is noted and skipped. A table at the end shows every run. With --tune,
# every model and use gets a whole tuning session instead (tune_sweep, below). With --resume,
# a sweep that stopped (Ctrl+C, a crash, a restart) carries on under the same sweep_id,
# skipping the models and uses it already finished (last_sweep, below).


@app.command()
def sweep(
    models: Annotated[
        list[str] | None,
        typer.Argument(
            help="Models to benchmark. Leave out for every model downloaded with Ollama."
        ),
    ] = None,
    use: Annotated[
        list[str] | None,
        typer.Option(help="What the models are for; repeat to test several. All three if none."),
    ] = None,
    tune: bool = typer.Option(
        False, "--tune", help="Tune every model for every use, like 'localtune tune'. Takes hours."
    ),
    resume: bool = typer.Option(
        False, "--resume", help="Carry on the last sweep that stopped, skipping what it finished."
    ),
    repetitions: int = typer.Option(3, help="How many times each test is repeated."),
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
    """Benchmark many models one after another, like every model you have in Ollama."""
    lengths = {name: lengths_for(name, None, None) for name in use or USES}
    llama_bench = find_tool("llama-bench")
    if llama_bench is None:
        console.print("llama.cpp isn't installed. Run 'localtune doctor' to install it.")
        raise typer.Exit(1)
    models = models or downloaded_models()
    if not models:
        console.print("No models to benchmark. Name some, or download one with Ollama first, like:")
        console.print("  ollama pull qwen3:8b", markup=False, highlight=False)
        raise typer.Exit(1)

    warn_if_busy(gpu)
    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)
    sweep_id, finished = uuid4().hex, set()
    if resume:
        last = last_sweep(tuned=tune)
        if last is None:
            console.print("There's no earlier sweep to resume, so this is a new one.")
        else:
            sweep_id, finished = last
            console.print(f"Resuming the last sweep; {len(finished)} model and use pairs are done.")
    if tune:
        tune_sweep(models, lengths, llama_bench, repetitions, electricity, gpu, sweep_id, finished)
        return
    planned = max(1, len(models) * len(lengths) - len(finished))
    console.print(
        f"Benchmarking {len(models)} model{'s' if len(models) > 1 else ''} for "
        f"{', '.join(lengths)}: {planned} run{'s' if planned > 1 else ''}, "
        "one at a time. Ctrl+C stops it and keeps the finished runs."
    )
    runs: list[tuple[str, RunResult]] = []
    started, done = time.monotonic(), 0
    try:
        for model in models:
            model_file, model_name = find_model_file(model)
            if model_file is None:
                console.print(f"Skipping {model}: it isn't a model file or an Ollama model.")
                done += len(lengths)
                continue
            for use_name, (prompt_tokens, gen_tokens) in lengths.items():
                if (model_name or model_file.stem, use_name) in finished:
                    continue
                placement = guessed_split(model_file, gpu, prompt_tokens + gen_tokens)
                options = BenchOptions(
                    prompt_tokens=prompt_tokens,
                    gen_tokens=gen_tokens,
                    repetitions=repetitions,
                    gpu_index=gpu,
                    **placement,
                )
                engine = partial(run_llama_bench, llama_bench, model_file, options, model_name)
                requested = requested_settings(model_file, options, model_name)
                left = time_left(time.monotonic() - started, done, planned)
                label = f"Run {done + 1} of {planned}: {model}, {use_name}{left}"
                with working(console, "sweep", label, done / planned):
                    result = measure(engine, requested, electricity, repetitions, gpu_index=gpu)
                result.sweep_id, result.use = sweep_id, use_name
                save_result(result)
                runs.append((use_name, result))
                done += 1
                console.print(f"{done:>3}. {model}, {use_name}: {outcome(result)}")
    except KeyboardInterrupt:
        console.print("Stopped. The runs that finished are saved.")

    show_sweep(runs)
    console.print(f"All runs saved to {results_file()}")


# Resuming: the last sweep's saved runs say which models and uses it finished. Each run says
# which use it was for; runs saved before that are matched to a use by their prompt and answer
# lengths. A plain sweep has one run per model and use, so each saved one is finished. A tuning
# sweep has a whole session per model and use, and the last one it started may have been cut
# off, so that one is done again.


def use_of(run: RunResult) -> str | None:
    if run.use:
        return run.use
    return next((name for name, (p, g) in USES.items() if p + g == run.settings.context_size), None)


def last_sweep(tuned: bool) -> tuple[str, set[tuple[str, str]]] | None:
    runs = [r for r in load_results() if r.sweep_id and (r.tune_id is not None) == tuned]
    if not runs:
        return None
    sweep_id = runs[-1].sweep_id
    pairs = [(r.settings.model, use_of(r)) for r in runs if r.sweep_id == sweep_id]
    finished = set(pairs)
    if tuned:
        finished.discard(pairs[-1])
    return sweep_id, finished


# "localtune sweep --tune": a whole tuning session (tune_model) for every model and use, one
# after another, which can take all night. It ends with one table of what was fastest for each,
# and how much that beat llama.cpp's usual settings.


def tune_sweep(
    models, lengths, llama_bench, repetitions, electricity, gpu, sweep_id, finished=frozenset()
) -> None:
    planned = max(1, len(models) * len(lengths) - len(finished))
    console.print(
        f"Tuning {len(models)} model{'s' if len(models) > 1 else ''} for {', '.join(lengths)}: "
        f"{planned} tuning session{'s' if planned > 1 else ''} of about 13 to 16 runs each, one "
        "at a time. This can take hours. Ctrl+C stops it and keeps the finished runs."
    )
    sessions: list[tuple[str, str, TuneSession]] = []
    started, done = time.monotonic(), 0
    try:
        for model in models:
            model_file, model_name = find_model_file(model)
            if model_file is None:
                console.print(f"Skipping {model}: it isn't a model file or an Ollama model.")
                done += len(lengths)
                continue
            for use_name, (prompt_tokens, gen_tokens) in lengths.items():
                if (model_name or model_file.stem, use_name) in finished:
                    continue
                left = time_left(time.monotonic() - started, done, planned)
                console.print(f"\nSession {done + 1} of {planned}: {model}, {use_name}{left}")
                session = TuneSession(given={"gpu_layers": None, "cpu_moe_layers": None})
                sessions.append((model, use_name, session))
                try:
                    tune_model(
                        session,
                        llama_bench,
                        model_file,
                        model_name,
                        prompt_tokens,
                        gen_tokens,
                        repetitions,
                        electricity,
                        gpu,
                        sweep_id=sweep_id,
                        use=use_name,
                    )
                except typer.Exit:
                    console.print(f"Nothing fit for {model}, {use_name}; moving on.")
                done += 1
    except KeyboardInterrupt:
        console.print("Stopped. The runs that finished are saved.")

    show_tune_sweep(sessions)
    console.print(f"All runs saved to {results_file()}")


def show_tune_sweep(sessions: list[tuple[str, str, TuneSession]]) -> None:
    if not sessions:
        return
    table = Table(title="Fastest settings for each model and use")
    for column in ("Model", "Use", "Fastest settings", "Split"):
        table.add_column(column)
    for column in ("Answer time", "Vs usual settings"):
        table.add_column(column, justify="right")
    for model, use_name, session in sessions:
        best = fastest(session.runs)
        if best is None:
            table.add_row(model, use_name, "nothing worked", "", "", "")
            continue
        s, usual = best.settings, default_run(session.runs)
        winner = {"flash_attention": s.flash_attention, "kv_cache": s.kv_cache_k}
        gain = ""
        if usual is not None and answer_seconds(usual):
            gain = f"{answer_seconds(usual) / answer_seconds(best) - 1:+.0%}"
        table.add_row(
            model,
            use_name,
            describe(winner | {"ubatch_size": s.ubatch_size}),
            describe_placement(session.placement_of(best), session.layers),
            f"{answer_seconds(best):,.1f} s",
            gain,
        )
    console.print(table)
    console.print(
        "'Vs usual settings' is how much faster the fastest settings gave a whole answer than "
        "llama.cpp's usual settings. Run 'localtune tune' on one model and use for the "
        "commands that use its settings."
    )


# A model too big for the GPU gets tune's first guess at how much of it fits beside the
# conversation (tune.py); one that fits, a file localtune can't read or a computer without a GPU
# gets llama.cpp's usual placement, every layer on the GPU.


def guessed_split(model_file: Path, gpu: int, tokens: int) -> dict:
    gpus = detect_hardware().gpus
    if gpu >= len(gpus):
        return {}
    try:
        layout = read_layout(model_file)
    except (GgufError, OSError):
        return {}
    guess = first_guess(model_file.stat().st_size, layout, gpus[gpu].vram_gb, tokens)
    return {} if guess is None else split_settings(guess, layout)


def show_sweep(runs: list[tuple[str, RunResult]]) -> None:
    if not runs:
        return
    table = Table(title="Sweep results")
    for column in ("Model", "Use"):
        table.add_column(column)
    for column in ("Answer time", "Decode tok/s", "Estimate", "Prefill tok/s", "Peak VRAM"):
        table.add_column(column, justify="right")
    for use_name, run in runs:
        seconds, speed, gpu = answer_seconds(run), run.speed, run.gpu
        if seconds is None:
            failed = "out of GPU memory" if run.status == "out_of_memory" else "failed"
            table.add_row(run.settings.model, use_name, failed, "", "", "", "")
            continue
        estimate = estimate_for(run)
        table.add_row(
            run.settings.model,
            use_name,
            f"{seconds:,.1f} s",
            f"{speed.decode_tok_s:,.1f}",
            f"{estimate:,.0f}" if estimate is not None else "",
            f"{speed.prefill_tok_s:,.0f}",
            f"{gpu.peak_vram_gb:.1f} GB" if gpu and gpu.peak_vram_gb is not None else "",
        )
    console.print(table)


# "localtune quality": how much worse a q8_0 or q4_0 KV cache makes a model's answers than
# f16 (quality.py), read from the same text with each. The text is read in chunks as long as a
# use's conversation (--use, uses.py, rounded up like llama-server's room for it), or a length
# given with --context. A model too big for the GPU starts from tune's first guess at a GPU/RAM
# split, like sweep, unless one is given. Each result is saved as soon as it finishes.


@app.command()
def quality(
    model: Annotated[str, typer.Argument(help="Model file (.gguf), or an Ollama model name.")],
    gpu_layers: int | None = typer.Option(None, help="How many layers go on the GPU."),
    cpu_moe: int | None = typer.Option(
        None, help="Keep the experts of this many layers in RAM (mixture-of-experts models)."
    ),
    use: str | None = typer.Option(
        None,
        help="Read the text in chunks as long as a use's conversation: chat, code or documents.",
    ),
    context: int | None = typer.Option(
        None, help=f"How many tokens each chunk of text has (default {CONTEXT:,})."
    ),
    chunks: int = typer.Option(CHUNKS, help="How many chunks of text to read."),
    gpu: int = typer.Option(0, help="Which GPU to use."),
) -> None:
    """Measure how much each KV cache type changes the model's answers."""
    if context is None:
        context = server_context(sum(lengths_for(use, None, None))) if use else CONTEXT
    llama_perplexity = find_tool("llama-perplexity")
    if llama_perplexity is None:
        console.print("llama.cpp isn't installed. Run 'localtune doctor' to install it.")
        raise typer.Exit(1)
    model_file, model_name = find_model_file(model)
    if model_file is None:
        explain_missing_model(model)
        raise typer.Exit(1)
    try:
        with working(console, "run", f"Downloading the test text ({TEXT_NAME})..."):
            text = get_text()
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        console.print(f"Couldn't download the test text: {error}")
        raise typer.Exit(1) from error

    warn_if_busy(gpu)
    given = {"gpu_layers": gpu_layers, "cpu_moe_layers": cpu_moe}
    placement = given if gpu_layers or cpu_moe else guessed_split(model_file, gpu, context)
    base = {"rig_id": rig_id(), "model": model_name or model_file.stem}
    base["hardware"] = detect_hardware()
    found = []
    for number, cache in enumerate(KV_CACHES, 1):
        label = f"Reading {chunks} chunks of {context:,} tokens with a {cache} KV cache"
        progress = (number - 1) / len(KV_CACHES)
        with working(console, "run", f"{label} ({number} of {len(KV_CACHES)})...", progress):
            result = measure_quality(
                llama_perplexity, model_file, text, cache, placement, base, gpu, chunks, context
            )
        save_quality(result)
        found.append(result)
    show_quality(found)
    console.print(f"Saved to {quality_file()}")


def show_quality(found: list[QualityResult]) -> None:
    f16 = next((r for r in found if r.kv_cache == "f16" and r.perplexity), None)
    first = found[0]
    table = Table(
        title=f"{first.model}: answer quality by KV cache, {first.context_size:,}-token chunks"
    )
    table.add_column("KV cache")
    for column in ("Perplexity", "Vs f16"):
        table.add_column(column, justify="right")
    for r in found:
        if r.perplexity is None:
            failed = "out of GPU memory" if r.status == "out_of_memory" else "failed"
            table.add_row(r.kv_cache, failed, "")
            continue
        loss = loss_pct(r, f16) if f16 and r is not f16 else None
        table.add_row(
            r.kv_cache,
            f"{r.perplexity:.3f} ± {r.perplexity_error:.3f}",
            "" if loss is None else f"{loss:+.1f}%",
        )
    console.print(table)
    console.print(
        "Perplexity is how surprised the model was by real text: lower is better. 'Vs f16' is "
        "how much worse each smaller cache made it."
    )
    for r in found:
        loss = loss_pct(r, f16) if f16 and r is not f16 else None
        if loss is not None:
            console.print(f"A {r.kv_cache} KV cache: {max(loss, 0):.1f}% worse, {verdict(loss)}.")
    errors = [r for r in found if r.error]
    for r in errors:
        console.print(f"[red]{r.kv_cache} failed:[/red] {r.error}", highlight=False)


# "localtune results": shows the runs saved by "localtune run" (results.py) as one table, so
# runs with different models, engines and settings can be compared side by side. Failed runs
# are listed too, because knowing which settings don't work is part of tuning.


@app.command()
def results(
    last: int = typer.Option(20, help="How many of the most recent runs to show."),
    model: str | None = typer.Option(None, help="Only show runs whose model name contains this."),
    csv_file: Annotated[
        Path | None,
        typer.Option("--csv", help="Save every run to this CSV file instead, for Excel or pandas."),
    ] = None,
) -> None:
    """Show your saved benchmark runs."""
    runs = load_results()
    if model:
        runs = [r for r in runs if model.lower() in r.settings.model.lower()]
    if not runs:
        console.print("No saved runs yet. Benchmark a model with 'localtune run'.")
        return
    if csv_file:
        save_csv(runs, csv_file)
        console.print(f"Saved {len(runs)} run{'s' if len(runs) > 1 else ''} to {csv_file}")
        return
    shown = runs[-last:]

    table = Table(title=f"Saved runs ({len(shown)} of {len(runs)})")
    table.add_column("When", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Engine")
    for column in ("Quant", "Settings", "Result"):
        table.add_column(column)
    for column in ("Prefill tok/s", "Decode tok/s", "Estimate", "Peak VRAM", "Cost/1M tok"):
        table.add_column(column, justify="right")
    for run in shown:
        table.add_row(*result_row(run))
    console.print(table)
    console.print(f"From {results_file()}")


# --csv: every run as one row (results.py), with predict's estimate beside the measured speeds.
# Runs saved by older versions lack some columns, which stay empty. The file starts with a mark
# that tells Excel it is UTF-8, which pandas reads as well.


def save_csv(runs: list[RunResult], path: Path) -> None:
    rows = [flat_record(run) | {"estimate.decode_tok_s": estimate_for(run)} for run in runs]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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
        f"{estimate:,.0f}" if (estimate := estimate_for(run)) is not None else "",
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


# "localtune share": offers the saved runs not shared yet to localtune's public dataset
# (share.py). It writes exactly what would be sent to a file, shows a summary and one run in
# full, and sends only after a yes. Nothing is sent without an address to send to.


@app.command()
def share(
    to: str | None = typer.Option(
        None, help="The address to send the runs to (or set LOCALTUNE_SHARE_URL)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Send without asking first."),
) -> None:
    """Share your saved runs with localtune's public dataset, after showing what is sent."""
    runs = not_yet_shared(load_results())
    if not runs:
        console.print("No new runs to share.")
        return
    body = payload(runs)
    path = write_preview(body)

    summary = Table(title="What would be shared", show_header=False)
    summary.add_row("Runs", str(len(runs)))
    summary.add_row("Models", ", ".join(sorted({r.settings.model for r in runs})))
    summary.add_row("GPUs", ", ".join(sorted({g.name for r in runs for g in r.hardware.gpus})))
    summary.add_row("This computer's ID", f"{runs[0].rig_id} (random, made by localtune)")
    console.print(summary)
    console.print("Each run looks like this one:")
    console.print_json(data=body["runs"][-1])
    console.print(
        "Left out: error messages (they can show folder names), your electricity price and "
        "zip code, and the time of day. Never saved at all: your name, your computer's name, "
        "your IP address and prompts."
    )
    console.print(f"Everything that would be sent is in {path}")

    url = share_url(to)
    if url is None:
        console.print("localtune's server for shared runs isn't running yet, so nothing was sent.")
        return
    if not yes and not typer.confirm(f"Send these {len(runs)} runs to {url}?", default=False):
        console.print("Nothing was sent.")
        return
    try:
        send(url, body)
    except ShareError as error:
        console.print(f"Nothing was sent: {error}", markup=False)
        raise typer.Exit(1) from error
    mark_shared(runs)
    console.print(f"Shared {len(runs)} runs. Thank you!")


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
