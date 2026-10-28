"""Benchmark a model file with llama.cpp's llama-bench."""

# llama-bench is llama.cpp's own speed test: it loads a model file (.gguf), reads a prompt of
# a set length ("prefill") and writes a set number of tokens ("decode"), several times each,
# and prints the speeds. This runner turns localtune's settings into llama-bench's options,
# runs it, and reads its answer back into localtune's shapes (results.py). base.py does the
# GPU watching and saving around it. llama-bench is found by install.py, in the user's PATH or
# in the copy localtune installed.

import json
import re
import subprocess
from pathlib import Path

from pydantic import BaseModel

from localtune.results import RunSettings, Speed
from localtune.runners.base import BackendError, EngineRun

ENGINE = "llama.cpp"

# The KV cache types llama.cpp accepts, checked before a run so a typo doesn't end up in the
# saved results as a failed benchmark.
KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")

# What a llama-bench run is asked to do. Settings left empty use llama-bench's own defaults,
# and the saved record shows the values it actually used.


class BenchOptions(BaseModel):
    prompt_tokens: int = 512
    gen_tokens: int = 128
    repetitions: int = 3
    gpu_layers: int | None = None
    cpu_moe_layers: int | None = None
    kv_cache: str | None = None
    flash_attention: bool | None = None
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    gpu_index: int = 0
    depth: int = 0


def build_command(llama_bench: str, model: Path, options: BenchOptions) -> list[str]:
    command = [
        llama_bench,
        "-m", str(model),
        "-p", str(options.prompt_tokens),
        "-n", str(options.gen_tokens),
        "-r", str(options.repetitions),
        "-mg", str(options.gpu_index),
        "-o", "json",
    ]  # fmt: skip
    optional = {
        "-ngl": options.gpu_layers,
        "-ncmoe": options.cpu_moe_layers,
        "-ctk": options.kv_cache,
        "-ctv": options.kv_cache,
        "-t": options.threads,
        "-b": options.batch_size,
        "-ub": options.ubatch_size,
        "-d": options.depth or None,
    }
    for flag, value in optional.items():
        if value is not None:
            command += [flag, str(value)]
    if options.flash_attention is not None:
        command += ["-fa", "on" if options.flash_attention else "off"]
    return command


# Reading llama-bench's answer. It prints one entry per test: the prefill test ("n_prompt") and
# the decode test ("n_gen"), each with its average speed in tokens per second ("avg_ts") and the
# settings it used, plus how long each repetition took ("samples_ns"), which together are the
# time spent on the tokens. Time to first token is worked out from the prefill speed: the time
# it takes to read the whole prompt before the first word of the answer can appear. The model
# is saved under the name it was asked for by (like qwen3:14b from ollama_models.py), or else
# its file name, and its quantization comes from llama-bench's description of the model.


def quant_name(model_type: str) -> str:
    # "qwen3 14B Q4_K - Medium" is the Q4_K_M everyone calls it; "IQ4_XS - 4.25 bpw" is IQ4_XS.
    base, _, variant = model_type.partition(" - ")
    quant = base.split()[-1]
    sizes = {"Small": "S", "Medium": "M", "Large": "L"}
    return f"{quant}_{sizes[variant]}" if variant in sizes else quant


def read_tests(stdout: str) -> list[dict]:
    try:
        tests = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise BackendError(f"couldn't read llama-bench's output ({error})") from error
    if not isinstance(tests, list):
        raise BackendError("couldn't read llama-bench's output (not a list of tests)")
    return tests


def parse_output(stdout: str, model: Path, repetitions: int, name: str | None = None) -> EngineRun:
    return parse_tests(read_tests(stdout), model, repetitions, name)


def parse_tests(
    tests: list[dict], model: Path, repetitions: int, name: str | None = None
) -> EngineRun:
    try:
        prefill = next(t for t in tests if t["n_prompt"] > 0 and t["n_gen"] == 0)
        decode = next(t for t in tests if t["n_gen"] > 0 and t["n_prompt"] == 0)
    except (StopIteration, KeyError, TypeError) as error:
        raise BackendError(f"couldn't read llama-bench's output ({error})") from error

    flash = {1: True, 0: False}.get(prefill.get("flash_attn"))
    settings = RunSettings(
        engine=ENGINE,
        engine_version=f"b{prefill['build_number']} ({prefill['build_commit']})",
        model=name or model.stem,
        model_file=model.name,
        quant=quant_name(prefill["model_type"]),
        context_size=prefill["n_prompt"] + decode["n_gen"],
        gpu_layers=prefill["n_gpu_layers"],
        cpu_moe_layers=prefill.get("n_cpu_moe"),
        kv_cache_k=prefill["type_k"],
        kv_cache_v=prefill["type_v"],
        flash_attention=flash,
        threads=prefill["n_threads"],
        batch_size=prefill["n_batch"],
        ubatch_size=prefill["n_ubatch"],
        decode_depth=decode.get("n_depth") or None,
    )
    speed = Speed(
        prompt_tokens=prefill["n_prompt"] * repetitions,
        generated_tokens=decode["n_gen"] * repetitions,
        prefill_tok_s=prefill["avg_ts"],
        decode_tok_s=decode["avg_ts"],
        ttft_ms=prefill["n_prompt"] / prefill["avg_ts"] * 1000,
        busy_s=sum(prefill["samples_ns"] + decode["samples_ns"]) / 1e9,
    )
    return EngineRun(settings=settings, speed=speed)


# When llama-bench fails it says why on its error output. Running out of GPU memory is the
# failure tuning cares most about (it marks settings that don't fit), so it gets its own status.

OUT_OF_MEMORY = re.compile(r"out of memory|failed to allocate|cudaMalloc failed", re.IGNORECASE)


def describe_failure(stderr: str) -> BackendError:
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    errors = [line for line in lines if "error" in line.lower()]
    message = (errors or lines or ["llama-bench stopped without saying why"])[-1]
    if OUT_OF_MEMORY.search(stderr):
        return BackendError(message, status="out_of_memory")
    return BackendError(message)


# Writing speed depends on how much conversation the model is already holding: each new word
# looks back over all of it, so a model writes more slowly after reading a long document than
# with an empty conversation, which is what llama-bench's decode test starts from. For a prompt
# of DEPTH_FROM tokens or more, llama-bench is run twice: once to read the prompt, and once to
# write the answer with the prompt already in memory (-d), the way the model is really used.
# Below that, the difference is too small to be worth loading the model a second time.

DEPTH_FROM = 4096


def bench(llama_bench: str, model: Path, options: BenchOptions) -> list[dict]:
    try:
        result = subprocess.run(
            build_command(llama_bench, model, options), capture_output=True, text=True
        )
    except OSError as error:
        raise BackendError(f"couldn't start llama-bench ({error})") from error
    if result.returncode != 0:
        raise describe_failure(result.stderr)
    return read_tests(result.stdout)


def run_llama_bench(
    llama_bench: str, model: Path, options: BenchOptions, name: str | None = None
) -> EngineRun:
    if options.prompt_tokens < DEPTH_FROM:
        return parse_tests(bench(llama_bench, model, options), model, options.repetitions, name)
    read = options.model_copy(update={"gen_tokens": 0})
    write = options.model_copy(update={"prompt_tokens": 0, "depth": options.prompt_tokens})
    tests = bench(llama_bench, model, read) + bench(llama_bench, model, write)
    return parse_tests(tests, model, options.repetitions, name)


# The settings that were asked for, saved in place of the ones llama-bench used when it fails
# before reporting them.


def requested_settings(model: Path, options: BenchOptions, name: str | None = None) -> RunSettings:
    return RunSettings(
        engine=ENGINE,
        model=name or model.stem,
        model_file=model.name,
        context_size=options.prompt_tokens + options.gen_tokens,
        gpu_layers=options.gpu_layers,
        cpu_moe_layers=options.cpu_moe_layers,
        kv_cache_k=options.kv_cache,
        kv_cache_v=options.kv_cache,
        flash_attention=options.flash_attention,
        threads=options.threads,
        batch_size=options.batch_size,
        ubatch_size=options.ubatch_size,
    )
