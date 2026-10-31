"""Look up a CPU's memory speed, for the part of a model that runs from RAM."""

# When a model doesn't fit on the GPU, llama.cpp keeps the rest in RAM and the CPU works
# through it, so that part is read at the speed of the computer's memory, not the GPU's. How
# fast that is depends on the CPU: how many memory channels it has and the fastest memory it
# takes, which its maker lists as its maximum memory bandwidth. A desktop CPU has two channels,
# a server CPU four to twelve. hardware.py only gets the CPU's name, so the rest is looked up in
# a small table shipped inside the package, src/localtune/data/cpus.csv, the same way gpus.py
# looks up GPUs. Each row says where its numbers come from: Intel's or AMD's own spec pages.
# The table starts with the CPUs localtune's runs were measured on; a CPU missing from it gets
# predict.py's assumed RAM speed instead.
#
# A computer only reaches that bandwidth when enough cores are reading from memory at once. A
# rented machine often gets a few of a big server CPU's cores, so the bandwidth is scaled by the
# share of the CPU's cores llama.cpp was given threads for.

import csv
from importlib.resources import files

from pydantic import BaseModel


class CpuSpec(BaseModel):
    name: str
    cores: int
    threads: int
    memory_channels: int
    memory_bandwidth_gbs: float
    source: str


def load_cpu_table() -> list[CpuSpec]:
    with (files("localtune") / "data" / "cpus.csv").open(newline="") as f:
        return [CpuSpec(**row) for row in csv.DictReader(f)]


# Matching the name the operating system reports to the table. Windows and Linux say
# "12th Gen Intel(R) Core(TM) i9-12900K" or "AMD EPYC 7402 24-Core Processor"; the table keeps
# just the model, "Core i9-12900K" or "EPYC 7402".

NOISE_WORDS = ("(r)", "(tm)", "intel", "amd", "cpu", "processor")


def short_name(name: str) -> str:
    name = name.lower().split("@")[0]
    for word in NOISE_WORDS:
        name = name.replace(word, " ")
    words = [w for w in name.split() if not w.endswith("-core") and w not in ("gen",)]
    words = [w for w in words if not (w[:-2].isdigit() and w[-2:] in ("th", "st", "nd", "rd"))]
    return " ".join(words)


def find_cpu_spec(name: str | None) -> CpuSpec | None:
    if not name:
        return None
    wanted = short_name(name)
    return next((spec for spec in load_cpu_table() if short_name(spec.name) == wanted), None)


def ram_bandwidth_gbs(name: str | None, threads: int | None) -> float | None:
    spec = find_cpu_spec(name)
    if spec is None:
        return None
    share = 1.0 if not threads else min(1.0, threads / spec.cores)
    return spec.memory_bandwidth_gbs * share
