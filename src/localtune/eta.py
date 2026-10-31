"""How long a benchmark should take, so the terminal can say how much time is left."""

# A run can take seconds or most of an hour: a model that fits on the GPU writes dozens of
# words a second, but one that is partly kept in RAM (too big for the GPU, or --gpu-layers)
# may write two. People want to know whether they can walk away, so "localtune run", "tune"
# and "sweep" (cli.py) show how long is left, and the loaders count it down (animations.py).
#
# The guess starts from predict's formula (predict.py): how fast this model should write on
# this GPU, with the part in RAM read at this computer's own RAM speed. Reading the prompt is
# guessed at PREFILL_PER_DECODE times the writing speed (about what runs so far have measured),
# llama-bench reads and writes once more as a warm-up before the timed repetitions, and the
# model file has to be loaded first (at LOAD_GBS). The formula is only rough, so the guess is
# scaled by how far off it was for this computer's own saved runs (pace): the median of the
# last PACE_RUNS, from runs of the same model when there are any. A computer with no saved runs
# yet gets the formula alone, and the terminal says it's a rough guess. A model or GPU missing
# from the tables (models.py, gpus.py) gets no guess at all.

import statistics
from dataclasses import dataclass, field

from localtune.gpus import GpuSpec
from localtune.hardware import HardwareInfo
from localtune.predict import own_ram_estimate
from localtune.results import ModelFacts, RunResult, gpu_facts, model_facts, rig_id

PREFILL_PER_DECODE = 50
LOAD_GBS = 2.0
PACE_RUNS = 20


@dataclass
class Guess:
    seconds: float
    rough: bool


def raw_seconds(
    facts: ModelFacts | None,
    gpu: GpuSpec | None,
    hardware: HardwareInfo,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    gpu_layers: int | None = None,
    cpu_moe_layers: int | None = None,
    threads: int | None = None,
) -> float | None:
    if facts is None or gpu is None:
        return None
    tokens = prompt_tokens + gen_tokens
    decode = own_ram_estimate(facts, gpu, hardware, tokens, threads, gpu_layers, cpu_moe_layers)
    if not decode:
        return None
    answer_s = gen_tokens / decode + prompt_tokens / (decode * PREFILL_PER_DECODE)
    return (repetitions + 1) * answer_s + facts.file_bytes / (LOAD_GBS * 1e9)


def raw_seconds_for(run: RunResult) -> float | None:
    if run.speed is None:
        return None
    reps, s = max(1, run.repetitions), run.settings
    return raw_seconds(
        run.model_facts,
        run.gpu_facts,
        run.hardware,
        run.speed.prompt_tokens // reps,
        run.speed.generated_tokens // reps,
        reps,
        s.gpu_layers,
        s.cpu_moe_layers,
        s.threads,
    )


def pace(runs: list[RunResult], model: str | None = None) -> float | None:
    me = rig_id()
    ratios: dict[str, list[float]] = {}
    for run in runs:
        if run.rig_id != me or run.status != "ok" or run.gpu is None or not run.gpu.duration_s:
            continue
        guess = raw_seconds_for(run)
        if guess:
            ratios.setdefault(run.settings.model, []).append(run.gpu.duration_s / guess)
    picked = ratios.get(model) or [r for model_ratios in ratios.values() for r in model_ratios]
    return statistics.median(picked[-PACE_RUNS:]) if picked else None


def guess_run(
    runs: list[RunResult],
    model: str,
    hardware: HardwareInfo,
    gpu: int,
    prompt_tokens: int,
    gen_tokens: int,
    repetitions: int,
    gpu_layers: int | None = None,
    cpu_moe_layers: int | None = None,
    threads: int | None = None,
) -> Guess | None:
    raw = raw_seconds(
        model_facts(model),
        gpu_facts(hardware, gpu),
        hardware,
        prompt_tokens,
        gen_tokens,
        repetitions,
        gpu_layers,
        cpu_moe_layers,
        threads,
    )
    if raw is None:
        return None
    scale = pace(runs, model)
    return Guess(raw * (scale or 1.0), rough=scale is None)


# How long a sweep has left. Its runs or tuning sessions differ a lot, so each gets a guess of
# its own (guesses, from raw_seconds, not yet scaled), and they are scaled by how this
# computer's earlier runs compared with their guesses (prior, from pace). Once some finish,
# their real times in this sweep say it better, since they were measured just now, and every
# one still to come is scaled by those instead. Something the tables can't guess is counted at
# the average of the finished ones, and until one has finished it can't be guessed at all.
# A run that failed (out of memory, a crash) ends early, so it isn't used to scale the rest.


@dataclass
class SweepClock:
    guesses: list[float | None]
    prior: float | None = None
    took: list[float | None] = field(default_factory=list)

    def scale(self) -> float | None:
        finished = [(g, t) for g, t in zip(self.guesses, self.took, strict=False) if g and t]
        if finished:
            return sum(t for _, t in finished) / sum(g for g, _ in finished)
        return self.prior

    @property
    def rough(self) -> bool:
        return self.prior is None and not any(self.took)

    def session(self, number: int) -> float | None:
        guess, scale = self.guesses[number], self.scale()
        if guess:
            return guess * (scale or 1.0)
        took = [t for t in self.took if t]
        return sum(took) / len(took) if took else None

    def after(self, number: int) -> float | None:
        later = [self.session(n) for n in range(number + 1, len(self.guesses))]
        return None if None in later else sum(later)

    def left(self, number: int) -> float | None:
        this, after = self.session(number), self.after(number)
        return None if this is None or after is None else this + after
