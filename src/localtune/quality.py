"""How much a smaller KV cache costs in answer quality, measured with llama-perplexity."""

# Tuning (tune.py) often finds a q8_0 or q4_0 KV cache fastest: it stores the conversation's
# memory in a half or a quarter of the space f16 takes, at some cost in how exactly the model
# remembers it. Speed tests can't see that cost, so this measures it. llama.cpp's
# llama-perplexity reads a long, fixed piece of real text and, at every word, checks how
# surprised the model was by the next one. The average surprise is the perplexity: lower means
# the model followed the text better. Reading the same text with each KV cache type, on the
# same model and computer, shows how much worse a smaller cache makes the model, as a
# percentage of f16's perplexity.
#
# The text is WikiText-2 (Wikipedia articles), the standard text llama.cpp's own quality
# comparisons use, so these numbers can be set beside published ones. It is downloaded once,
# the first time it's needed, into localtune's folder (results.py). The text is read in long
# chunks, because a KV cache only holds a conversation's memory: short chunks would leave
# little in it to get wrong. Every result is saved to quality.jsonl, beside results.jsonl.

import io
import re
import subprocess
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from localtune import __version__
from localtune.hardware import HardwareInfo
from localtune.results import home_dir
from localtune.runners.llamacpp import OUT_OF_MEMORY

TEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
TEXT_MEMBER = "wikitext-2-raw/wiki.test.raw"
TEXT_NAME = "WikiText-2 test"
CONTEXT = 4096
CHUNKS = 8
KV_CACHES = ("f16", "q8_0", "q4_0")


class QualityResult(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    rig_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    localtune_version: str = __version__
    engine_version: str | None = None
    model: str
    model_file: str
    kv_cache: str
    flash_attention: bool = True
    gpu_layers: int | None = None
    cpu_moe_layers: int | None = None
    context_size: int = CONTEXT
    chunks: int = CHUNKS
    text: str = TEXT_NAME
    status: Literal["ok", "out_of_memory", "crashed"] = "ok"
    error: str | None = None
    perplexity: float | None = None
    perplexity_error: float | None = None
    hardware: HardwareInfo
    gpu_index: int = 0


# The text, downloaded once and kept. It is only kept when it unpacks to the file llama.cpp's
# own scripts use, so a failed or wrong download is tried again next time.


def text_file() -> Path:
    return home_dir() / "data" / "wiki.test.raw"


def get_text(url: str = TEXT_URL) -> Path:
    path = text_file()
    if path.is_file():
        return path
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    text = archive.read(TEXT_MEMBER)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text)
    return path


# Running llama-perplexity. Flash attention is on for every cache type, because llama.cpp needs
# it for quantized ones, so the only difference between the runs is the cache. The model is
# placed on the GPU (and in RAM, for models too big for it) the way the caller says.


def build_command(
    llama_perplexity: str,
    model: Path,
    text: Path,
    kv_cache: str,
    placement: dict,
    gpu_index: int = 0,
    chunks: int = CHUNKS,
) -> list[str]:
    command = [
        llama_perplexity,
        "-m", str(model),
        "-f", str(text),
        "-c", str(CONTEXT),
        "--chunks", str(chunks),
        "-fa", "on",
        "-ctk", kv_cache,
        "-ctv", kv_cache,
        "-mg", str(gpu_index),
    ]  # fmt: skip
    if placement.get("gpu_layers") is not None:
        command += ["-ngl", str(placement["gpu_layers"])]
    if placement.get("cpu_moe_layers"):
        command += ["--n-cpu-moe", str(placement["cpu_moe_layers"])]
    return command


# Reading its answer: the last line it prints is "Final estimate: PPL = 6.1234 +/- 0.04567",
# the perplexity and how far off it could be from the text's length alone. It also prints which
# llama.cpp build it is ("build: 11669 (6b1d2e9)").

FINAL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")
BUILD = re.compile(r"build: (\d+) \((\w+)\)")


def parse_output(output: str) -> tuple[float, float, str | None] | None:
    final = FINAL.search(output)
    if final is None:
        return None
    build = BUILD.search(output)
    version = f"b{build.group(1)} ({build.group(2)})" if build else None
    return float(final.group(1)), float(final.group(2)), version


def measure(
    llama_perplexity: str,
    model: Path,
    text: Path,
    kv_cache: str,
    placement: dict,
    base: dict,
    gpu_index: int = 0,
    chunks: int = CHUNKS,
) -> QualityResult:
    result = QualityResult(
        model_file=model.name,
        kv_cache=kv_cache,
        gpu_layers=placement.get("gpu_layers"),
        cpu_moe_layers=placement.get("cpu_moe_layers"),
        chunks=chunks,
        gpu_index=gpu_index,
        **base,
    )
    command = build_command(llama_perplexity, model, text, kv_cache, placement, gpu_index, chunks)
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace")
    except OSError as error:
        result.status, result.error = "crashed", f"couldn't start llama-perplexity ({error})"
        return result
    output = done.stdout + done.stderr
    parsed = parse_output(output)
    if done.returncode != 0 or parsed is None:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        result.status = "out_of_memory" if OUT_OF_MEMORY.search(output) else "crashed"
        result.error = (lines or ["llama-perplexity stopped without saying why"])[-1]
        return result
    result.perplexity, result.perplexity_error, result.engine_version = parsed
    return result


# How much worse than f16 a cache type made the model, in plain words. Differences under 1% are
# about what reading a different piece of text would change, so they can't be noticed in use.


def loss_pct(result: QualityResult, f16: QualityResult) -> float | None:
    if result.perplexity is None or not f16.perplexity:
        return None
    return (result.perplexity / f16.perplexity - 1) * 100


def verdict(loss: float) -> str:
    if loss < 1:
        return "too small to notice"
    if loss < 3:
        return "small, but there"
    return "noticeable"


# Saving: one result per line in quality.jsonl, like results.jsonl.


def quality_file() -> Path:
    return home_dir() / "quality.jsonl"


def save_quality(result: QualityResult) -> Path:
    path = quality_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")
    return path
