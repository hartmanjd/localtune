"""Look up a GPU's specs: memory, bandwidth, compute power."""

# How fast a model runs depends mostly on a few facts about the GPU: how much memory it has
# (what fits), how fast that memory is (memory bandwidth, which sets the writing speed, since
# every new word reads the whole model once), and how much it can compute (which sets the
# prompt reading speed). nvidia-smi (hardware.py) only gives the name and memory size, so
# this module looks the rest up in a small table shipped inside the package. The prediction
# model will use these specs to estimate speeds on GPUs nobody has tested yet, and
# "localtune info" shows them.
#
# The table is src/localtune/data/gpus.csv: NVIDIA GeForce RTX 20-50 cards plus the data
# center GPUs common on rental sites. Each row says where its numbers come from, which is one
# of NVIDIA's own spec pages, datasheets or architecture whitepapers. Where NVIDIA lists cores
# and clock speed but not FP32 TFLOPS, the TFLOPS are worked out as cores x boost clock x 2.
# Where NVIDIA doesn't list the memory bandwidth, the row says whether it matches llmfit's
# table (github.com/AlexsJones/llmfit, MIT license) or was worked out from the memory bus
# width NVIDIA lists and the card's memory speed. Laptop GPUs aren't included: they share names
# with desktop cards but have slower memory. A few cards were also sold with slower memory
# under the same name and size (an RTX 4070 with GDDR6, an RTX 3060 Ti with GDDR6X); the
# table has the common version.

import csv
from importlib.resources import files

from pydantic import BaseModel


class GpuSpec(BaseModel):
    name: str
    vram_gb: float
    architecture: str
    compute_capability: str
    memory_bandwidth_gbs: float
    fp32_tflops: float
    board_power_w: float
    source: str


def load_gpu_table() -> list[GpuSpec]:
    with (files("localtune") / "data" / "gpus.csv").open(newline="") as f:
        return [GpuSpec(**row) for row in csv.DictReader(f)]


# Matching nvidia-smi's name to the table. nvidia-smi says "NVIDIA GeForce RTX 4070 Ti SUPER"
# or "Tesla T4"; the table leaves off the brand words. Some cards come with two memory sizes
# under one name (the RTX 3060 has 8 and 12 GB versions, with different memory speeds), so the
# memory size nvidia-smi reports picks between them. With no memory size, like a GPU named in
# "localtune predict --gpu-name", the version with the most memory is picked.

BRAND_WORDS = ("NVIDIA ", "GeForce ", "Tesla ")


def short_name(name: str) -> str:
    for word in BRAND_WORDS:
        name = name.replace(word, "")
    return name.strip().lower()


def find_gpu_spec(name: str, vram_gb: float | None = None) -> GpuSpec | None:
    matches = [spec for spec in load_gpu_table() if short_name(spec.name) == short_name(name)]
    if not matches:
        return None
    if vram_gb is None:
        return max(matches, key=lambda spec: spec.vram_gb)
    return min(matches, key=lambda spec: abs(spec.vram_gb - vram_gb))
