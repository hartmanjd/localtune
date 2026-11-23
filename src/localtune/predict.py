"""A quick estimate of whether a model fits on a GPU and how fast it will write there."""

# "localtune predict" answers the question people have before downloading a model: will it
# run on my GPU, and how fast? It works it out from the two tables shipped with localtune,
# without running anything: the model's file size and how much of it each word uses (models.py),
# and the GPU's memory and memory bandwidth (gpus.py). It is an estimate, not a measurement:
# "localtune measure" measures the real speed, and the runs people save are what a trained
# prediction model will learn from later.
#
# Writing speed (decode) is set by memory: every new word reads the part of the model it uses
# once, so the GPU's memory bandwidth divided by those bytes is the most words per second it
# can write. A real engine only reaches part of that, EFFICIENCY, the same default llmfit uses
# in its estimate (github.com/AlexsJones/llmfit, docs/how-it-works.md). It is a stand-in until
# the trained prediction model exists, and it isn't fitted to localtune's own runs, so the
# trained model can be compared against it fairly. Reading the prompt (prefill) is limited by
# compute instead, which the tables can't estimate well yet, so it isn't predicted.

from pydantic import BaseModel

from localtune.cpus import ram_bandwidth_gbs
from localtune.gpus import GpuSpec
from localtune.hardware import HardwareInfo
from localtune.models import ModelFile, ModelSpec
from localtune.results import ModelFacts, RunResult
from localtune.tune import KV_CACHE_SIZE, RESERVED_GB

GIB = 1024**3
EFFICIENCY = 0.55


# A model bigger than the GPU's memory still runs: llama.cpp keeps the rest in RAM, where the
# CPU reads it far more slowly. That part is read at RAM_BANDWIDTH_GBS, the speed of two sticks
# of DDR4-3200 memory on paper (2 x 3,200 million transfers x 8 bytes), the slowest common
# memory in home PCs today, so the estimate assumes the model is split well between the two (which
# "localtune tune" finds). A RAM speed can be given instead (ram_bandwidth), which the trained
# model's starting point uses (below); the estimate localtune shows keeps the fixed one, so it
# stays the plain formula the trained model is measured against. In a
# mixture-of-experts model, the part moved to RAM is experts, and each word only uses a few of
# them, so the RAM part slows it down much less. RAM_RESERVED_GB is left for the operating
# system and other programs; a model that doesn't fit in what's left of the RAM either can't
# run on the computer at all.

RAM_BANDWIDTH_GBS = 51.2
RAM_RESERVED_GB = 4


class Prediction(BaseModel):
    model: str
    quant: str
    file_gb: float
    conversation_gb: float
    gpu: str
    vram_gb: float
    memory_bandwidth_gbs: float
    in_ram_gb: float
    fits_on_computer: bool | None
    decode_tok_s: float | None


# The conversation's memory (KV cache): a key and a value per KV head in every layer, for every
# token, at 2 bytes per number. Hybrid models only keep it in some layers, so for them this
# is more than they really need.


def kv_bytes_per_token(model: ModelSpec | ModelFacts) -> int:
    return model.layers * model.kv_heads * model.head_dim * 2 * 2


def predict(
    model: ModelSpec | ModelFacts,
    file: ModelFile,
    gpu: GpuSpec,
    tokens: int,
    ram_gb: float | None = None,
    ram_bandwidth: float | None = None,
    forced_in_ram: float = 0.0,
) -> Prediction:
    conversation = tokens * kv_bytes_per_token(model)
    usable = (gpu.vram_gb - RESERVED_GB) * GIB - conversation
    in_ram = max(0.0, file.size_bytes - max(0.0, usable), forced_in_ram)
    fits = None if ram_gb is None else in_ram <= (ram_gb - RAM_RESERVED_GB) * GIB

    decode = None
    if fits is not False:
        used_per_word = file.size_bytes * model.active_params / model.params
        ram_share = model.experts_used / model.experts if model.experts else 1.0
        from_ram = min(in_ram * ram_share, used_per_word)
        seconds = (used_per_word - from_ram) / (gpu.memory_bandwidth_gbs * 1e9) + from_ram / (
            (ram_bandwidth or RAM_BANDWIDTH_GBS) * 1e9
        )
        decode = EFFICIENCY / seconds

    return Prediction(
        model=file.ollama,
        quant=file.quant,
        file_gb=file.size_bytes / GIB,
        conversation_gb=conversation / GIB,
        gpu=gpu.name,
        vram_gb=gpu.vram_gb,
        memory_bandwidth_gbs=gpu.memory_bandwidth_gbs,
        in_ram_gb=in_ram / GIB,
        fits_on_computer=fits,
        decode_tok_s=decode,
    )


# The same estimate for a run that was measured, from the model and GPU facts saved with it
# (results.py), so "localtune measure", "tune", "sweep" and "results" can show it beside the real
# speed: how far off the simple formula was, for this model on this computer. It is worked out
# for the run's own conversation length. Runs saved before the facts were kept, or with a model
# or GPU not in the tables, have no estimate. Neither has a run spread over several GPUs
# (gpu_monitor.py): the estimates describe one GPU, and no such run was ever learned from.
#
# The trained model (scripts/train.py) starts from a better version of it instead: the part of
# the model in RAM is read at the RAM speed of the run's own CPU (cpus.py), with the threads
# llama.cpp was given, where the CPU is in the table. Computers' RAM speeds differ several times
# over, and the fixed RAM speed above was where the formula missed most. It also counts the
# part of the model a run kept in RAM on purpose (--gpu-layers, --cpu-moe): tune's split search
# and the rental medium sweep do that even when the whole model would fit on the GPU.


def estimate_for(run: RunResult, own_ram: bool = False) -> float | None:
    p = prediction_for(run)
    if p is None or not own_ram:
        return None if p is None else p.decode_tok_s
    facts, gpu, speed, s = run.model_facts, run.gpu_facts, run.speed, run.settings
    tokens = (speed.prompt_tokens + speed.generated_tokens) // max(1, run.repetitions)
    return own_ram_estimate(
        facts,
        gpu,
        run.hardware,
        tokens,
        s.threads,
        s.gpu_layers,
        s.cpu_moe_layers,
        s.decode_depth,
        s.kv_cache_k,
    )


def several_gpus(run: RunResult) -> bool:
    return run.gpu is not None and len(run.gpu.cards) > 1


def prediction_for(run: RunResult) -> Prediction | None:
    facts, gpu, speed = run.model_facts, run.gpu_facts, run.speed
    if facts is None or gpu is None or speed is None or several_gpus(run):
        return None
    file = ModelFile(quant=facts.quant, size_bytes=facts.file_bytes, ollama=facts.name, source="")
    tokens = (speed.prompt_tokens + speed.generated_tokens) // max(1, run.repetitions)
    return predict(facts, file, gpu, tokens, run.hardware.ram_gb)


# Both estimates assume a model too big for the GPU is split well between the GPU and RAM. A
# run told to keep all of it on the GPU anyway (no --gpu-layers or --cpu-moe) doesn't fail on
# Windows: the NVIDIA driver quietly moves the overflow into the computer's RAM, where the GPU
# reaches it slowly over the PCIe slot, often several times slower than a split. Such a run
# measures something the estimates don't describe.


def overflowed_gpu(run: RunResult) -> bool:
    p, facts = prediction_for(run), run.model_facts
    if p is None or facts is None or p.in_ram_gb <= 0:
        return False
    return kept_in_ram(facts, run.settings.gpu_layers, run.settings.cpu_moe_layers) == 0


def own_ram_estimate(
    facts: ModelFacts,
    gpu: GpuSpec,
    hardware: HardwareInfo,
    tokens: int,
    threads: int | None = None,
    gpu_layers: int | None = None,
    cpu_moe_layers: int | None = None,
    depth: int | None = None,
    kv_cache: str | None = None,
) -> float | None:
    file = ModelFile(quant=facts.quant, size_bytes=facts.file_bytes, ollama=facts.name, source="")
    ram_bandwidth = ram_bandwidth_gbs(hardware.cpu_name, threads or hardware.cpu_threads)
    forced = kept_in_ram(facts, gpu_layers, cpu_moe_layers)
    decode = predict(facts, file, gpu, tokens, hardware.ram_gb, ram_bandwidth, forced).decode_tok_s
    return with_conversation(decode, facts, gpu, ram_bandwidth, depth, kv_cache, gpu_layers)


# Every new word also reads the conversation so far: the key and value each layer kept for every
# earlier token (the KV cache). Early in a chat that is tiny, but deep into a long document it is
# gigabytes per word, as much as half the model itself, and the formula above leaves it out. So
# the trained model's starting point adds it, read where it lives: llama.cpp keeps a layer's
# part of the conversation next to the layer, so the layers left in RAM with --gpu-layers have
# theirs read at RAM speed and the rest at GPU speed (--cpu-moe moves only experts, so the
# conversation stays on the GPU). A smaller cache type (tune.KV_CACHE_SIZE) is less to read.
#
# Not every model keeps the whole conversation in every layer. FULL_CONVERSATION is the share
# of a model's layers that do, by the architecture name the model file gives: gpt-oss alternates
# layers that look back over the whole conversation with layers that only see the last 128
# tokens; Gemma 3 and 4 keep 1 full layer in every 6, the others seeing the last 1,024 tokens (512
# on the smallest);
# Qwen 3.5 keeps 1 in every 4, the others carrying a small fixed-size memory instead (from each
# model's published config). Every other architecture reads it all. What the windowed layers
# read is small next to a long document, so it's left out.

FULL_CONVERSATION = {"gpt-oss": 1 / 2, "gemma3": 1 / 6, "gemma4": 1 / 6, "qwen35": 1 / 4}


def with_conversation(
    decode: float | None,
    facts: ModelFacts,
    gpu: GpuSpec,
    ram_bandwidth: float | None,
    depth: int | None,
    kv_cache: str | None = None,
    gpu_layers: int | None = None,
) -> float | None:
    if not decode or not depth:
        return decode
    extra = conversation_seconds(
        facts, gpu, ram_bandwidth or RAM_BANDWIDTH_GBS, depth, kv_cache, gpu_layers
    )
    return EFFICIENCY / (EFFICIENCY / decode + extra)


def conversation_seconds(
    facts: ModelFacts,
    gpu: GpuSpec,
    ram_bandwidth: float,
    depth: int,
    kv_cache: str | None = None,
    gpu_layers: int | None = None,
) -> float:
    size = KV_CACHE_SIZE.get(kv_cache or "f16", 1.0)
    full = FULL_CONVERSATION.get(facts.architecture, 1.0)
    read = depth * kv_bytes_per_token(facts) * size * full
    layers = facts.layers
    in_ram = 0.0
    if gpu_layers is not None and 0 <= gpu_layers < layers:
        in_ram = (layers - gpu_layers) / layers
    return read * (1 - in_ram) / (gpu.memory_bandwidth_gbs * 1e9) + read * in_ram / (
        ram_bandwidth * 1e9
    )


# How many bytes of the model a run's settings keep in RAM. --gpu-layers N leaves the other
# layers in RAM (a negative N means all on the GPU). --cpu-moe N leaves the experts of N layers in
# RAM; the experts' share of the file comes from the model's total and active sizes, since only
# the experts differ between them.


def kept_in_ram(facts: ModelFacts, gpu_layers: int | None, cpu_moe_layers: int | None) -> float:
    layers = facts.layers
    in_ram = 0.0
    if gpu_layers is not None and 0 <= gpu_layers < layers:
        in_ram += facts.file_bytes * (layers - gpu_layers) / layers
    experts, used = facts.experts, facts.experts_used
    if cpu_moe_layers and experts and used and experts > used:
        expert_params = (facts.params - facts.active_params) * experts / (experts - used)
        expert_bytes = facts.file_bytes * min(1.0, expert_params / facts.params)
        in_ram += expert_bytes * min(cpu_moe_layers, layers) / layers
    return min(in_ram, float(facts.file_bytes))
