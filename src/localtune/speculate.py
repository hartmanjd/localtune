"""Measure speculative decoding: a small model guesses ahead and the big one checks."""

# llama.cpp can let something cheap guess the next few tokens (a small "draft" model from the
# same family, or a lookup of text already in the conversation), and have the big model check
# all the guesses in one pass. Every guess it agrees with is a token written for free; the first
# one it disagrees with is replaced by its own. The answer stays the big model's answer. Whether
# that is faster depends on how often the guesses are right, which depends on how predictable
# the text is, and on whether the draft model still fits on the GPU beside the big one.
#
# "localtune speculate" (cli.py) measures it instead of guessing. A setup is one way of starting
# llama-server (no speculation, a draft model with so many tokens guessed ahead, a lookup type);
# a case is one kind of request (a task, a prompt length, a temperature). Every setup answers
# every case a few times. llama-server reports its own writing speed and how many guessed tokens
# it made and kept, so each row has the speed, the share of guesses kept (the acceptance rate)
# and the speed-up over the setup without speculation on the same case.
#
# The built-in cases test what changes predictability directly: an answer that copies from the
# prompt against one written fresh, a short prompt against a long one, and a temperature of 0
# against a livelier one. Rows are saved to speculate.jsonl, apart from results.jsonl: they are
# measurements of a setting, not runs the speed model may learn from.

import itertools
import math
import random
import subprocess
import tempfile
import urllib.error
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from localtune.gguf import ModelLayout
from localtune.gpu_monitor import GpuMonitor
from localtune.hardware import GpuInfo
from localtune.results import home_dir
from localtune.runners.base import BackendError
from localtune.runners.web_api import get_json
from localtune.side_by_side import START_TIMEOUT_S, free_port, post_json, stop, wait_until_ready
from localtune.tune import ALL_LAYERS, SPILL_MARGIN_GB

NO_SPECULATION = "none"
DRAFT_TYPE = "draft-simple"
# The types llama.cpp's pinned build knows (common/speculative.cpp). Several can be used at
# once, written with a plus: draft-simple+ngram-mod.
DRAFT_TYPES = ("draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark")
LOOKUP_TYPES = ("ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache")
TASKS = ("copy", "fresh")
# The quick test answers "does this help me?" in a few minutes; the thorough one adds a long
# prompt (a pasted document or codebase), a livelier temperature and a third answer per case.
QUICK = {"context": "2048", "temperature": "0", "repeats": 2}
THOROUGH = {"context": "2048,16384", "temperature": "0,0.8", "repeats": 3}
SMALL_ENOUGH = 0.34  # a draft over a third of the big model's size costs more than it saves
WORTH_IT = 1.05  # a speed-up under 5% is within what two runs of the same setup differ by


class Setup(BaseModel):
    spec_type: str = NO_SPECULATION
    draft: str | None = None  # the draft model's name, as the person gave it
    draft_file: str | None = None
    n_max: int | None = None  # tokens guessed ahead; None leaves llama.cpp's own default
    n_min: int | None = None
    p_min: float | None = None

    @property
    def speculates(self) -> bool:
        return self.spec_type != NO_SPECULATION

    @property
    def label(self) -> str:
        if not self.speculates:
            return "no speculation"
        parts = [self.draft] if self.draft else []
        if self.spec_type != DRAFT_TYPE or not self.draft:
            parts.append(self.spec_type)
        for name, value in (("n-max", self.n_max), ("n-min", self.n_min), ("p-min", self.p_min)):
            if value is not None:
                parts.append(f"{name} {value:g}")
        return ", ".join(parts)


class Case(BaseModel):
    task: str
    context: int | None = None  # prompt length asked for; None for a prompt from a file
    temperature: float = 0.0
    prompt: str


class SpeculateRun(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str
    rig_id: str
    gpu: str | None = None
    model: str
    llama_cpp: str | None = None
    kv_cache: str = "f16"
    draft_gpu_layers: int | None = None
    draft_kv_cache: str | None = None
    llama_args: list[str] = []
    setup: Setup
    task: str
    context: int | None = None
    temperature: float = 0.0
    status: str = "ok"
    error: str | None = None
    decode_tok_s: float | None = None  # the middle one of decode_runs
    decode_runs: list[float] = []
    prefill_tok_s: float | None = None
    seconds: float | None = None  # per answer, reading the prompt included
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    draft_tokens: int | None = None
    draft_accepted: int | None = None
    peak_vram_gb: float | None = None
    free_gb: float | None = None  # on the fullest GPU the setup used
    baseline_decode_tok_s: float | None = None

    @property
    def acceptance(self) -> float | None:
        return self.draft_accepted / self.draft_tokens if self.draft_tokens else None

    @property
    def speedup(self) -> float | None:
        if not (self.decode_tok_s and self.baseline_decode_tok_s):
            return None
        return self.decode_tok_s / self.baseline_decode_tok_s

    @property
    def spilled(self) -> bool:
        return self.free_gb is not None and self.free_gb < SPILL_MARGIN_GB


# The setups. One without speculation always comes first, since every speed-up is measured
# against it. A type with a draft part is tried once per draft model; a lookup type needs no
# model. Every value given for n-max, n-min and p-min is combined with every other.


def needs_draft(spec_type: str) -> bool:
    return any(part in DRAFT_TYPES for part in spec_type.split("+"))


def known_type(spec_type: str) -> bool:
    return all(part in DRAFT_TYPES + LOOKUP_TYPES for part in spec_type.split("+"))


def setups(
    drafts: list[tuple[str, Path]],
    types: list[str],
    n_max: list[int | None],
    n_min: list[int | None],
    p_min: list[float | None],
) -> list[Setup]:
    found = [Setup()]
    for spec_type in types:
        if needs_draft(spec_type):
            for (name, file), most, least, sure in itertools.product(drafts, n_max, n_min, p_min):
                found.append(
                    Setup(
                        spec_type=spec_type,
                        draft=name,
                        draft_file=str(file),
                        n_max=most,
                        n_min=least,
                        p_min=sure,
                    )
                )
        else:
            found.append(Setup(spec_type=spec_type))
    return found


# A draft has to cut text into the same tokens as the big model, or its guesses mean nothing to
# it. llama.cpp compares the two vocabularies token by token when it loads the pair, and quietly
# runs without the draft when they differ (Qwen2.5 and Qwen3 share a size but not every token).
# So a draft is only picked when its file lists exactly the same tokens, with the same start
# and end tokens. A pair that slips through anyway shows up as a setup that made no guesses.


def can_draft_for(draft: ModelLayout, target: ModelLayout) -> bool:
    if not (draft.tokenizer and draft.vocab_digest):
        return False
    same_tokens = (draft.tokenizer, draft.vocab_digest) == (target.tokenizer, target.vocab_digest)
    return same_tokens and draft.special_tokens == target.special_tokens


def pick_drafts(
    target: Path,
    target_layout: ModelLayout,
    candidates: list[tuple[str, Path]],
    layout_of: Callable[[Path], ModelLayout],
    most: int = 2,
) -> list[tuple[str, Path]]:
    size = target.stat().st_size
    fitting = []
    for name, file in candidates:
        try:
            if file == target or file.stat().st_size > size * SMALL_ENOUGH:
                continue
            if can_draft_for(layout_of(file), target_layout):
                fitting.append((file.stat().st_size, name, file))
        except (OSError, ValueError):
            continue
    return [(name, file) for _, name, file in sorted(fitting)[:most]]


# The cases. Both tasks get the same made-up notes as their prompt, so the prompt's length is
# the same and only the answer differs: "copy" asks for the notes again word for word (as a
# summary, an edit of pasted code or a quote does in part), "fresh" asks for a story the notes
# have nothing to do with. The notes are built from a fixed seed, so every setup reads the same
# ones, and they come out at about TOKENS_PER_NOTE tokens each.

WHO = "the keeper,a surveyor,the baker,an engineer,the pilot,a gardener,the clerk,a diver"
DID = "measured,repaired,painted,counted,moved,recorded,cleaned,tested"
WHAT = "the north gate,a copper valve,the old ledger,a signal lamp,the tide gauge,a spare rope"
WHEN = "before dawn,at noon,after the storm,on the second day,late in the evening,in winter"
TOKENS_PER_NOTE = 23  # measured with Qwen3's tokenizer: 128 notes were 2,970 tokens
COPY = "Write out the notes above again, word for word, starting from Note 1. Write nothing else."
FRESH = (
    "Ignore the notes above. Write a new story about a lighthouse keeper who finds a letter "
    "that was never sent."
)
REQUESTS = {"copy": COPY, "fresh": FRESH}


def notes(tokens: int) -> str:
    pick = random.Random(tokens)
    lists = [words.split(",") for words in (WHO, DID, WHAT, WHEN)]
    lines = []
    for number in range(1, max(tokens // TOKENS_PER_NOTE, 4) + 1):
        who, did, what, when = (pick.choice(words) for words in lists)
        wrote = pick.randint(100, 999)
        lines.append(f"Note {number}: {who} {did} {what} {when}, and wrote down {wrote}.")
    return "\n".join(lines)


def built_in_cases(tasks: list[str], contexts: list[int], temperatures: list[float]) -> list[Case]:
    return [
        Case(
            task=task,
            context=context,
            temperature=temperature,
            prompt=f"{notes(context)}\n\n{REQUESTS[task]}",
        )
        for task, context, temperature in itertools.product(tasks, contexts, temperatures)
    ]


def file_cases(files: list[Path], temperatures: list[float]) -> list[Case]:
    return [
        Case(task=file.stem, temperature=temperature, prompt=file.read_text(encoding="utf-8"))
        for file, temperature in itertools.product(files, temperatures)
    ]


# How much conversation llama-server is given room for: the longest prompt, its answer, and
# some slack for the chat template and for the notes coming out longer than asked.


def room_for(cases: list[Case], gen_tokens: int) -> int:
    longest = max(case.context or math.ceil(len(case.prompt) / 3) for case in cases)
    return math.ceil((longest * 1.25 + gen_tokens + 256) / 1024) * 1024


# Each prompt length gets its own start of llama-server, sized for it: the conversation's memory
# is set aside when llama-server starts, for the draft as well as the model, so room for a long
# prompt would crowd the GPU during the short ones too.


def by_length(cases: list[Case]) -> list[list[Case]]:
    groups: dict[int | None, list[Case]] = {}
    for case in cases:
        groups.setdefault(case.context, []).append(case)
    return list(groups.values())


# A rough guess of how long it all takes, said before it starts. Writing is guessed at predict's
# formula speed (the GPU's memory speed over the model's size), a little slower on average since
# a draft whose guesses are mostly wrong slows writing down, and reading the prompt at about 40
# times that. Each start of llama-server reads the model from disk.

EFFICIENCY = 0.55  # predict.py's share of the memory speed a model reaches when writing
SLOWER_ON_AVERAGE = 1.3


def rough_seconds(
    model_gb: float,
    bandwidth_gbs: float | None,
    setup_count: int,
    cases: list[Case],
    repeats: int,
    gen_tokens: int,
    loads: int,
) -> float:
    decode = EFFICIENCY * bandwidth_gbs / model_gb if bandwidth_gbs else 30.0
    answers = 0.0
    for case in cases:
        prompt = case.context or len(case.prompt) / 3
        answers += gen_tokens / decode * SLOWER_ON_AVERAGE + prompt / (decode * 40)
    return loads * (5 + model_gb / 1.5) + setup_count * answers * repeats


# Starting llama-server for a setup. The big model is placed as tune would place it; the draft
# goes all on the GPU unless told otherwise. Anything in llama_args is passed on as it is, last,
# so it can override what is set here.


class ServerOptions(BaseModel):
    context: int
    kv_cache: str = "f16"
    placement: dict = {}
    gpu: int = 0
    draft_gpu_layers: int | None = None
    draft_kv_cache: str | None = None
    llama_args: list[str] = []


def server_command(
    llama_server: str, model_file: Path, setup: Setup, options: ServerOptions, port: int
) -> list[str]:
    gpu_layers = options.placement.get("gpu_layers")
    flags: dict[str, object] = {
        "-m": model_file,
        "-c": options.context,
        "-ngl": ALL_LAYERS if gpu_layers is None else gpu_layers,
        "--n-cpu-moe": options.placement.get("cpu_moe_layers"),
        "-fa": "on",
        "-np": 1,
        "-mg": options.gpu or None,
        "--host": "127.0.0.1",
        "--port": port,
    }
    if options.kv_cache != "f16":
        flags["-ctk"] = flags["-ctv"] = options.kv_cache
    if setup.speculates:
        if setup.draft_file:
            flags["--spec-draft-model"] = setup.draft_file
            flags["--spec-draft-ngl"] = (
                ALL_LAYERS if options.draft_gpu_layers is None else options.draft_gpu_layers
            )
            if options.draft_kv_cache:
                flags["--spec-draft-type-k"] = options.draft_kv_cache
                flags["--spec-draft-type-v"] = options.draft_kv_cache
        flags["--spec-type"] = setup.spec_type.replace("+", ",")
        flags["--spec-draft-n-max"] = setup.n_max
        flags["--spec-draft-n-min"] = setup.n_min
        flags["--spec-draft-p-min"] = setup.p_min
    command = [llama_server]
    command += [str(p) for flag, value in flags.items() if value is not None for p in (flag, value)]
    return command + options.llama_args


# One answer. The request isn't streamed, so llama-server's own timings come back whole:
# its reading and writing speeds, and the guessed tokens it made and kept. Thinking is turned
# off where the model allows it, so a thinking model writes the answer that was asked for.
# Each repeat has its own seed, the same one in every setup.


class Answer(BaseModel):
    decode_tok_s: float | None = None
    prefill_tok_s: float | None = None
    seconds: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    draft_tokens: int = 0
    draft_accepted: int = 0


def ask(url: str, case: Case, gen_tokens: int, seed: int, post: Callable = post_json) -> Answer:
    body = {
        "messages": [{"role": "user", "content": case.prompt}],
        "max_tokens": gen_tokens,
        "temperature": case.temperature,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        reply = post(url + "/v1/chat/completions", body, timeout_s=START_TIMEOUT_S)
    except urllib.error.HTTPError as error:
        said = error.read().decode("utf-8", errors="replace").strip()[:300]
        raise BackendError(f"llama-server refused the request: {said}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise BackendError(f"llama-server didn't answer ({error})") from error
    timings = reply.get("timings") or {}
    took_ms = (timings.get("prompt_ms") or 0) + (timings.get("predicted_ms") or 0)
    return Answer(
        decode_tok_s=timings.get("predicted_per_second"),
        prefill_tok_s=timings.get("prompt_per_second"),
        seconds=took_ms / 1000 if took_ms else None,
        prompt_tokens=timings.get("prompt_n"),
        generated_tokens=timings.get("predicted_n"),
        draft_tokens=timings.get("draft_n") or 0,
        draft_accepted=timings.get("draft_n_accepted") or 0,
    )


def middle(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    half = len(ordered) // 2
    return ordered[half] if len(ordered) % 2 else (ordered[half - 1] + ordered[half]) / 2


def average(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


def case_numbers(answers: list[Answer]) -> dict:
    speeds = [a.decode_tok_s for a in answers if a.decode_tok_s]
    guessed = sum(a.draft_tokens for a in answers)
    return {
        "decode_tok_s": middle(speeds),
        "decode_runs": speeds,
        "prefill_tok_s": average([a.prefill_tok_s for a in answers]),
        "seconds": average([a.seconds for a in answers]),
        "prompt_tokens": answers[0].prompt_tokens,
        "generated_tokens": answers[0].generated_tokens,
        "draft_tokens": guessed or None,
        "draft_accepted": sum(a.draft_accepted for a in answers) if guessed else None,
    }


# One setup: llama-server is started, answers every case, and is stopped, with the GPU's memory
# watched the whole time. A setup that can't start (out of memory with the draft beside the
# model, a flag llama.cpp refuses) gives every case a failed row that says why, and a case that
# fails on its own doesn't stop the others.


def free_on_fullest(gpus: list[GpuInfo], cards: list[int], peaks: list[float]) -> float | None:
    sizes = {g.index: g.vram_gb for g in gpus}
    left = [sizes[card] - peak for card, peak in zip(cards, peaks, strict=True) if card in sizes]
    return round(min(left), 2) if left else None


def build_of(url: str, props: Callable) -> str | None:
    try:
        return str(props(url + "/props", 5).get("build_info") or "") or None
    except (urllib.error.URLError, OSError, ValueError, AttributeError):
        return None


def measure_setup(
    llama_server: str,
    model_file: Path,
    setup: Setup,
    cases: list[Case],
    options: ServerOptions,
    gpus: list[GpuInfo],
    gen_tokens: int = 256,
    repeats: int = 3,
    start: Callable = subprocess.Popen,
    monitor: Callable[[int], GpuMonitor] = lambda gpu_index: GpuMonitor(gpu_index=gpu_index),
    ready: Callable = wait_until_ready,
    post: Callable = post_json,
    props: Callable = get_json,
) -> list[dict]:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    rows: list[dict] = []
    build = None
    with tempfile.TemporaryDirectory() as folder:
        log = Path(folder) / "llama-server.log"
        with log.open("w", encoding="utf-8") as output, monitor(options.gpu) as gpu_monitor:
            process = start(
                server_command(llama_server, model_file, setup, options, port),
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                ready(process, url, log, START_TIMEOUT_S)
                build = build_of(url, props)
                ask(url, cases[0].model_copy(update={"prompt": "Say hello."}), 8, 0, post)
                for case in cases:
                    try:
                        answers = [
                            ask(url, case, gen_tokens, seed, post) for seed in range(repeats)
                        ]
                        rows.append(case_numbers(answers))
                    except BackendError as error:
                        rows.append({"status": error.status, "error": str(error)})
            except BackendError as error:
                rows = [{"status": error.status, "error": str(error)} for _ in cases]
            finally:
                stop(process)
    summary = gpu_monitor.summary()
    memory: dict = {"llama_cpp": build}
    if summary is not None:
        memory |= {
            "peak_vram_gb": summary.peak_vram_gb,
            "free_gb": free_on_fullest(gpus, summary.cards, summary.card_peak_vram_gb),
        }
    return [{**row, **memory} for row in rows]


# Reading a finished session. The speed-up of a row is against the row without speculation on
# the same case. A setup's overall figure is the geometric mean of its speed-ups (the fair
# average of ratios), and the best setup is the highest of those among setups that kept room on
# the GPU.


def same_case(a: SpeculateRun, b: SpeculateRun) -> bool:
    return (a.task, a.context, a.temperature) == (b.task, b.context, b.temperature)


def with_baselines(runs: list[SpeculateRun]) -> list[SpeculateRun]:
    plain = [run for run in runs if not run.setup.speculates and run.decode_tok_s]
    for run in runs:
        base = next((p for p in plain if same_case(p, run)), None)
        run.baseline_decode_tok_s = base.decode_tok_s if base else None
    return runs


def overall(runs: list[SpeculateRun]) -> float | None:
    gains = [run.speedup for run in runs if run.speedup]
    if not gains:
        return None
    return math.exp(sum(math.log(gain) for gain in gains) / len(gains))


def by_setup(runs: list[SpeculateRun]) -> list[tuple[Setup, list[SpeculateRun]]]:
    groups: list[tuple[Setup, list[SpeculateRun]]] = []
    for run in runs:
        group = next((rows for setup, rows in groups if setup == run.setup), None)
        if group is None:
            groups.append((run.setup, [run]))
        else:
            group.append(run)
    return groups


def refused(setup: Setup, rows: list[SpeculateRun]) -> bool:
    """A draft setup that answered but never guessed: llama.cpp ran it without the draft."""
    answered = [row for row in rows if row.status == "ok"]
    return bool(setup.draft and answered) and not any(row.draft_tokens for row in answered)


def best_setup(runs: list[SpeculateRun]) -> tuple[Setup, float] | None:
    scored = [
        (gain, setup)
        for setup, rows in by_setup(runs)
        if setup.speculates
        and not refused(setup, rows)
        and not any(row.spilled for row in rows)
        and (gain := overall(rows)) is not None
    ]
    if not scored:
        return None
    gain, setup = max(scored, key=lambda pair: pair[0])
    return setup, gain


# Saving, one row per line like results.jsonl, and reading a session back.


def speculate_file() -> Path:
    return home_dir() / "speculate.jsonl"


def save_runs(runs: list[SpeculateRun]) -> Path:
    path = speculate_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for run in runs:
            f.write(run.model_dump_json() + "\n")
    return path


def load_runs(path: Path | None = None) -> list[SpeculateRun]:
    path = path or speculate_file()
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            runs.append(SpeculateRun.model_validate_json(line))
        except ValueError:
            continue
    return runs
