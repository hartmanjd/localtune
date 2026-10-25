"""The localtune terminal command."""

# Everything a person types in the terminal ("localtune info", "localtune doctor",
# "localtune run", "localtune monitor", and later "results") starts here. This file only handles
# the conversation with the person: it asks other modules for the actual work (hardware.py for
# the computer's details, gpu_monitor.py, cost.py and prices.py for GPU readings and their cost,
# doctor.py for whether the computer is ready, runners/ for benchmarks) and shows the answers.

import sys
import time
from pathlib import Path
from typing import Annotated

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
from localtune.gpu_monitor import GpuMonitor
from localtune.hardware import detect_hardware
from localtune.install import (
    DOWNLOAD_MB,
    InstallError,
    choose_build,
    find_tool,
    install_llama_cpp,
)
from localtune.prices import find_price
from localtune.results import RunResult, save_result
from localtune.runners.base import measure
from localtune.runners.llamacpp import (
    KV_CACHE_TYPES,
    BenchOptions,
    requested_settings,
    run_llama_bench,
)

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """Benchmark local LLMs on your own hardware."""


# "localtune info": shows what localtune can see about this computer, the same details that
# will be saved with every benchmark run.


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
    gpus.add_column("Driver")
    for gpu in hw.gpus:
        gpus.add_row(str(gpu.index), gpu.name, f"{gpu.vram_gb} GB", gpu.driver_version)
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
        console.print("Not ready to benchmark: needs llama.cpp.")
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


# "localtune run": benchmarks one model file with llama.cpp and saves the result. While
# llama-bench runs, the GPU is watched and the energy priced (runners/base.py); afterwards the
# speeds, energy and cost are shown, and the run is added to the results file (results.py).
# A failed run is saved too, and ends with an error code.


@app.command()
def run(
    model: Annotated[Path, typer.Argument(help="The model file (.gguf) to benchmark.")],
    prompt_tokens: int = typer.Option(512, help="Length of the prompt the model reads."),
    gen_tokens: int = typer.Option(128, help="How many tokens the model writes."),
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
    """Benchmark a model file with llama.cpp and save the result."""
    llama_bench = find_tool("llama-bench")
    if llama_bench is None:
        console.print("llama.cpp isn't installed. Run 'localtune doctor' to install it.")
        raise typer.Exit(1)
    if not model.is_file():
        console.print(f"Model file not found: {model}")
        raise typer.Exit(1)
    if kv_cache is not None and kv_cache not in KV_CACHE_TYPES:
        console.print(f"Unknown KV cache type {kv_cache}. Choose from: {', '.join(KV_CACHE_TYPES)}")
        raise typer.Exit(1)

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
    electricity = find_price(zip_code=zip_code, state=state, country=country, your_price=price)

    with console.status(f"Benchmarking {model.name} with llama.cpp..."):
        result = measure(
            lambda: run_llama_bench(llama_bench, model, options),
            requested_settings(model, options),
            electricity,
            repetitions,
            gpu_index=gpu,
        )
    path = save_result(result)

    if result.status != "ok":
        reason = "ran out of GPU memory" if result.status == "out_of_memory" else "failed"
        console.print(f"The benchmark {reason}: {result.error}")
        console.print(f"Saved the failed run to {path}")
        raise typer.Exit(1)
    show_run(result)
    console.print(f"Saved to {path}")


def show_run(result: RunResult) -> None:
    speed, gpu, energy = result.speed, result.gpu, result.energy
    unit = energy.price.currency
    table = Table(title=f"{result.settings.model} ({result.settings.quant})", show_header=False)
    table.add_row("Prefill", f"{speed.prefill_tok_s:,.0f} tok/s")
    table.add_row("Decode", f"{speed.decode_tok_s:,.1f} tok/s")
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
