"""Prove the biggest context: load the model at that size and read the GPU's memory."""

# tune.biggest_context works out the longest conversation that fits beside a model on the GPU
# from the sizes written in the model's file. That is arithmetic, and it can't know what else is
# using the GPU on this computer. "localtune context" (cli.py) checks it for real: llama-server
# is started with that much room, the GPU's memory is read while it loads (gpu_monitor.py), and
# it is stopped again. No question is asked, since llama-server sets the whole conversation's
# memory aside when it starts.
#
# A size is proven when the model loaded and every GPU it used still had SPILL_MARGIN_GB free,
# the same room tune asks of a winner. When the GPU came out too full, the next try is smaller
# by what was missing; when a lot was left over (some models need less memory per token than
# their file suggests), one bigger size is tried. Each proof is saved to context.jsonl, beside
# results.jsonl, so predict can show it.

import math
import subprocess
import tempfile
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from localtune.gpu_monitor import GpuMonitor
from localtune.hardware import GpuInfo
from localtune.results import home_dir
from localtune.runners.base import BackendError
from localtune.side_by_side import START_TIMEOUT_S, free_port, stop, wait_until_ready
from localtune.tune import ALL_LAYERS, KV_CACHE_SIZE, SMALLEST_CONTEXT, SPILL_MARGIN_GB

GIB = 1024**3
MAX_TRIES = 4
SLACK_GB = 0.25  # a bigger size is only tried when it would still leave this much over the margin
SETTLE_S = 2.0  # how long the loaded model is watched, so the memory reading is the full one


class ContextTry(BaseModel):
    context: int
    loaded: bool
    peak_vram_gb: float | None = None
    free_gb: float | None = None  # on the fullest GPU the model used

    @property
    def proven(self) -> bool:
        return self.loaded and self.free_gb is not None and self.free_gb >= SPILL_MARGIN_GB


class ContextProof(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rig_id: str
    model: str
    kv_cache: str
    formula_context: int
    proven_context: int | None
    tries: list[ContextTry]


# One try: llama-server with the whole model on the GPU and room for this much conversation.
# A model that can't load at this size (out of memory) is a try that failed, not an error.


def server_command(
    llama_server: str, model_file: Path, context: int, kv_cache: str, port: int
) -> list[str]:
    command = [llama_server, "-m", str(model_file), "-c", str(context), "-ngl", str(ALL_LAYERS)]
    command += ["-fa", "on", "-np", "1", "--host", "127.0.0.1", "--port", str(port)]
    if kv_cache != "f16":
        command += ["-ctk", kv_cache, "-ctv", kv_cache]
    return command


def free_on_fullest(gpus: list[GpuInfo], cards: list[int], peaks: list[float]) -> float | None:
    sizes = {g.index: g.vram_gb for g in gpus}
    left = [sizes[card] - peak for card, peak in zip(cards, peaks, strict=True) if card in sizes]
    return round(min(left), 2) if left else None


def load_and_measure(
    llama_server: str,
    model_file: Path,
    context: int,
    kv_cache: str,
    gpus: list[GpuInfo],
    gpu_index: int = 0,
    start: Callable = subprocess.Popen,
    monitor: Callable[[int], GpuMonitor] = lambda gpu_index: GpuMonitor(gpu_index=gpu_index),
    ready: Callable = wait_until_ready,
    sleep: Callable[[float], None] = time.sleep,
) -> ContextTry:
    port = free_port()
    loaded = True
    with tempfile.TemporaryDirectory() as folder:
        log = Path(folder) / "llama-server.log"
        with log.open("w", encoding="utf-8") as output, monitor(gpu_index) as gpu_monitor:
            process = start(
                server_command(llama_server, model_file, context, kv_cache, port),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                ready(process, f"http://127.0.0.1:{port}", log, START_TIMEOUT_S)
                sleep(SETTLE_S)
            except BackendError:
                loaded = False
            finally:
                stop(process)
    summary = gpu_monitor.summary()
    if summary is None:
        return ContextTry(context=context, loaded=loaded)
    return ContextTry(
        context=context,
        loaded=loaded,
        peak_vram_gb=summary.peak_vram_gb,
        free_gb=free_on_fullest(gpus, summary.cards, summary.card_peak_vram_gb),
    )


# The search. It starts at the formula's size and moves by whole thousands of tokens: down by
# what the GPU was short (a tenth of the size when the model didn't load at all, which says
# nothing about how much was missing), or up once into room that was left over.


def whole_thousands(tokens: float) -> int:
    return int(tokens // 1024) * 1024


def smaller(tried: ContextTry, token_bytes: float) -> int:
    if tried.loaded and tried.free_gb is not None:
        short = (SPILL_MARGIN_GB - tried.free_gb) * GIB
        return whole_thousands(tried.context - math.ceil(short / token_bytes) - 1024)
    return whole_thousands(tried.context * 0.9)


def bigger(tried: ContextTry, token_bytes: float, model_limit: int | None) -> int | None:
    spare = (tried.free_gb - SPILL_MARGIN_GB - SLACK_GB) * GIB
    more = whole_thousands(spare / token_bytes)
    if more < 1024:
        return None
    target = tried.context + more
    target = min(target, model_limit) if model_limit else target
    return target if target > tried.context else None


def prove(
    try_context: Callable[[int], ContextTry],
    formula_context: int,
    kv_bytes_per_token: int,
    kv_cache: str = "f16",
    model_limit: int | None = None,
) -> tuple[int | None, list[ContextTry]]:
    token_bytes = kv_bytes_per_token * KV_CACHE_SIZE[kv_cache]
    tries: list[ContextTry] = []
    proven = None
    context, came_down = formula_context, False
    while len(tries) < MAX_TRIES and context >= SMALLEST_CONTEXT:
        tried = try_context(context)
        tries.append(tried)
        if tried.proven:
            proven = context
            up = None if came_down else bigger(tried, token_bytes, model_limit)
            if up is None:
                break
            context, came_down = up, True  # one step up, then stop wherever it lands
        elif proven is not None:
            break
        else:
            context, came_down = smaller(tried, token_bytes), True
    return proven, tries


# Saving and reading the proofs back, one per line like results.jsonl.


def context_file() -> Path:
    return home_dir() / "context.jsonl"


def save_proof(proof: ContextProof) -> Path:
    path = context_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(proof.model_dump_json() + "\n")
    return path


def latest_proof(rig_id: str, model: str, kv_cache: str = "f16") -> ContextProof | None:
    path = context_file()
    if not path.exists():
        return None
    found = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            proof = ContextProof.model_validate_json(line)
        except ValueError:
            continue
        if (proof.rig_id, proof.model, proof.kv_cache) == (rig_id, model, kv_cache):
            found = proof
    return found
