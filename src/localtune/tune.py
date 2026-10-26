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

from itertools import product

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


# The fastest run is the one that writes fastest (decode tok/s), the speed people feel while
# an answer appears. Runs that failed, like running out of GPU memory, can't win.


def fastest(runs: list[RunResult]) -> RunResult | None:
    worked = [run for run in runs if run.status == "ok" and run.speed and run.speed.decode_tok_s]
    return max(worked, key=lambda run: run.speed.decode_tok_s, default=None)


# The "localtune run" options that repeat the winner's tuned settings, so it can be used again.


def run_flags(run: RunResult) -> str:
    s = run.settings
    flash = "--flash-attn" if s.flash_attention else "--no-flash-attn"
    return f"{flash} --kv-cache {s.kv_cache_k} --ubatch-size {s.ubatch_size}"
