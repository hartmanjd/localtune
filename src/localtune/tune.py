"""The settings "localtune tune" tries, and how it picks the fastest."""

# How fast a model runs on a GPU depends not only on the model and the GPU but on a few
# llama.cpp settings, and the best ones differ from one computer to the next:
#   - flash attention: a faster way of doing the attention step, which also saves memory;
#   - the KV cache type: how the conversation's memory is stored. f16 is full quality; q8_0 and
#     q4_0 take a half and a quarter of the memory (leaving room for longer conversations), and
#     llama.cpp only allows them with flash attention on;
#   - the micro-batch size: how many prompt tokens the GPU works on at once, which mostly
#     changes how fast a prompt is read.
# "localtune tune" (cli.py) benchmarks every combination below, one after another, saving each
# run like "localtune run" does. Together the runs say which settings are fastest on this
# computer, and they are exactly the kind of data the prediction model learns from: the same
# model and GPU with only the settings changed.

import math
from collections.abc import Callable
from itertools import product

from localtune.gguf import ModelLayout
from localtune.results import RunResult

ATTENTION_AND_CACHE = [(False, "f16"), (True, "f16"), (True, "q8_0"), (True, "q4_0")]
UBATCH_SIZES = [256, 512, 1024, 2048]


# Each combination is a set of BenchOptions fields (runners/llamacpp.py).


def combinations() -> list[dict]:
    return [
        {"flash_attention": flash, "kv_cache": cache, "ubatch_size": ubatch}
        for (flash, cache), ubatch in product(ATTENTION_AND_CACHE, UBATCH_SIZES)
    ]


def describe(settings: dict) -> str:
    flash = "on" if settings["flash_attention"] else "off"
    return f"flash attn {flash}, KV {settings['kv_cache']}, micro-batch {settings['ubatch_size']}"


# The fastest run is the one that gives a whole answer soonest: the time to read the prompt
# plus the time to write the answer. With a short prompt that is mostly writing speed (decode
# tok/s), the speed people feel while an answer appears; with a long prompt, like a pasted
# document, reading it matters as much. Judging by both also catches a GPU so full that it has
# started borrowing slow RAM, which on Windows can leave writing fast but make reading crawl.
# Runs that failed, like running out of GPU memory, can't win.


def answer_seconds(run: RunResult) -> float | None:
    speed = run.speed
    if run.status != "ok" or not (speed and speed.prefill_tok_s and speed.decode_tok_s):
        return None
    total_s = (
        speed.prompt_tokens / speed.prefill_tok_s + speed.generated_tokens / speed.decode_tok_s
    )
    return total_s / run.repetitions


def fastest(runs: list[RunResult]) -> RunResult | None:
    worked = [run for run in runs if answer_seconds(run) is not None]
    return min(worked, key=answer_seconds, default=None)


# The "localtune run" options that repeat the winner's tuned settings, so it can be used again.


def run_flags(run: RunResult) -> str:
    s = run.settings
    flash = "--flash-attn" if s.flash_attention else "--no-flash-attn"
    return f"{flash} --kv-cache {s.kv_cache_k} --ubatch-size {s.ubatch_size}"


# Models bigger than the GPU. llama.cpp can keep part of a model in the computer's normal
# memory (RAM) and run that part on the CPU, which is much slower, so the more of the model
# fits on the GPU the faster it writes, until the GPU runs out of memory (or, on Windows, starts
# borrowing slow RAM by itself, which makes it slower again). There are two ways to split it:
#   - an ordinary model is split by layers: the first layers on the GPU, the rest on the CPU;
#   - a mixture-of-experts model keeps every layer on the GPU but moves the experts of some
#     layers to RAM. Each word only uses a few experts, so this is much faster than moving
#     whole layers, and it is how these models are usually run on home GPUs.
# Either way the choice is one number: how many layers' worth of weights are on the GPU.
# tune.py makes a first guess from the file size and the GPU's memory, then tries one more
# and one fewer, keeping the fastest, so only a handful of runs are needed. A long conversation
# needs its own memory on the GPU (the KV cache), so the longer it is, the less of the model
# fits beside it.

RESERVED_GB = 1.5  # left free for the desktop, the KV cache and llama.cpp's working memory


def split_settings(on_gpu: int, layout: ModelLayout) -> dict:
    if layout.experts:
        return {"gpu_layers": layout.layers + 1, "cpu_moe_layers": layout.layers - on_gpu}
    return {"gpu_layers": on_gpu}


def describe_split(on_gpu: int, layout: ModelLayout) -> str:
    if layout.experts:
        return f"experts of {layout.layers - on_gpu} of {layout.layers} layers in RAM"
    return f"{on_gpu} of {layout.layers} layers on the GPU"


def first_guess(
    file_bytes: int, layout: ModelLayout, vram_gb: float, context: int = 0
) -> int | None:
    """How many layers' weights should fit on the GPU, or None if the whole model fits."""
    usable = (vram_gb - RESERVED_GB) * 1024**3 - context * layout.kv_bytes_per_token
    if file_bytes <= usable:
        return None
    if layout.experts and layout.expert_share:
        experts_per_layer = file_bytes * layout.expert_share / layout.layers
        moved = math.ceil((file_bytes - usable) / experts_per_layer)
        return max(0, layout.layers - moved)
    per_layer = file_bytes / (layout.layers + 1)  # the output layer is about one more
    return max(0, min(layout.layers, int(usable // per_layer) - 1))


def find_best_split(
    try_split: Callable[[int], float | None], start: int, layers: int, max_tries: int = 12
) -> int | None:
    """Tries splits around the first guess; try_split gives a split's answer time, or None."""
    times: dict[int, float | None] = {}

    def time(on_gpu: int) -> float | None:
        if on_gpu not in times:
            times[on_gpu] = try_split(on_gpu)
        return times[on_gpu]

    on_gpu = start
    while time(on_gpu) is None:
        if on_gpu == 0 or len(times) >= max_tries:
            return None
        on_gpu = max(0, on_gpu - 2)
    for step in (1, -1):
        while 0 <= on_gpu + step <= layers and len(times) < max_tries:
            quicker = time(on_gpu + step)
            if quicker is None or quicker >= times[on_gpu]:
                break
            on_gpu += step
    return on_gpu
