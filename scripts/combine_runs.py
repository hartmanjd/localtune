"""Combine every machine's benchmark runs into one table for training the prediction model.

Run from the project folder:
    uv run python scripts/combine_runs.py
"""

# The dataset lives in data/runs/, one folder per machine that measured it (your own PC, each
# rented cloud GPU), each holding the results.jsonl that localtune saved on that machine. This
# script reads them all and writes data/all_runs.csv: one row per run, with the same columns as
# 'localtune results --csv' (results.py), plus the folder it came from. Running it again after
# adding a machine's file rebuilds the table from scratch, so it never holds a run twice.

import csv
import sys
from collections import defaultdict
from pathlib import Path

from localtune.background import BUSY_CPU_PCT, OTHER_VRAM_GB
from localtune.predict import estimate_for, several_gpus
from localtune.results import RunResult, flat_record, gpu_facts, load_results, model_facts

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"
OUT_FILE = ROOT / "data" / "all_runs.csv"


# Which runs can be trusted for training. Every run is kept in the table, but a run that failed,
# or that shared the computer with another program, gets a reason in the skip_reason column, so
# training can leave it out while the table still shows what was measured. Failed runs still
# matter later: they show where a model stops fitting. Another program on the GPU is judged by
# the memory it holds, not by how busy the GPU looked: runs saved before localtune took the
# quietest of its readings often caught the end of the run before them, and any program using
# the GPU holds some of its memory. A run told to use more CPU threads than the computer allows
# makes the threads wait on each other, which can make it many times slower; that's a mistake in
# how it was started, not something about the computer, so it's left out too. So is a run that
# was spread over several GPUs, which the model has no way to describe yet.


def skip_reason(run: RunResult) -> str:
    if run.status != "ok":
        return run.status
    if several_gpus(run):
        return f"spread over {len(run.gpu.cards)} GPUs"
    b = run.background
    if b and b.gpu_vram_used_gb is not None and b.gpu_vram_used_gb >= OTHER_VRAM_GB:
        return f"busy: {b.gpu_vram_used_gb:.1f} GB of GPU memory in use by other programs"
    if b and b.cpu_load_pct is not None and b.cpu_load_pct >= BUSY_CPU_PCT:
        return f"busy: the CPU was {b.cpu_load_pct:.0f}% busy"
    if run.gpu and run.gpu.baseline_vram_gb >= OTHER_VRAM_GB:
        return f"busy: {run.gpu.baseline_vram_gb:.1f} GB of GPU memory in use at the start"
    threads, allowed = run.settings.threads, run.hardware.cpu_threads
    if threads and allowed and threads > allowed:
        return f"too many threads: {threads} on a computer that allows {allowed}"
    return ""


# Runs saved before localtune kept the model and GPU facts with each run (results.py) are missing
# them, so they would have no estimate and couldn't be used for training. Those facts come from
# localtune's own tables and don't change, so they are filled in from the tables now, the same
# way a new run gets them. A model is only filled in when the table's file for its name is the
# quantization the run used, since another file would have a different size; models the
# tables don't know (like a run of a downloaded file by its file name) stay empty. The
# facts_filled column says which runs got them this way.


def fill_facts(run: RunResult) -> tuple[RunResult, bool]:
    changes = {}
    if run.model_facts is None:
        facts = model_facts(run.settings.model)
        quant = run.settings.quant
        if facts and (not quant or facts.quant.lower() == quant.lower()):
            changes["model_facts"] = facts
    if run.gpu_facts is None:
        spec = gpu_facts(run.hardware, run.gpu_index)
        if spec:
            changes["gpu_facts"] = spec
    return run.model_copy(update=changes), bool(changes)


# Reading every machine's file. The same file copied into two folders, or copied again after
# more runs were added, would repeat runs, so each run_id is only kept the first time.


def combine(runs_dir: Path = RUNS_DIR) -> list[dict]:
    rows, seen = [], set()
    for path in sorted(runs_dir.glob("*/*.jsonl")):
        for run in load_results(path):
            if run.run_id in seen:
                continue
            seen.add(run.run_id)
            run, filled = fill_facts(run)
            rows.append(
                {"machine": path.parent.name}
                | flat_record(run)
                | {
                    "estimate.decode_tok_s": estimate_for(run),
                    "estimate.own_ram_decode_tok_s": estimate_for(run, own_ram=True),
                    "facts_filled": filled,
                    "skip_reason": skip_reason(run),
                }
            )
    return rows


def save(rows: list[dict], path: Path = OUT_FILE) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


# A summary per machine, so a missing or half-copied file shows up straight away.


def summary(rows: list[dict]) -> list[str]:
    machines = defaultdict(list)
    for row in rows:
        machines[row["machine"]].append(row)
    lines = []
    for machine, runs in machines.items():
        gpu = runs[0].get("hardware.gpu.name") or "unknown GPU"
        usable = sum(1 for run in runs if not run["skip_reason"])
        lines.append(f"{machine}: {gpu}, {len(runs)} runs, {usable} usable for training")
    return lines


def main() -> None:
    rows = combine()
    if not rows:
        print(f"No runs found. Put each machine's results.jsonl in its own folder in {RUNS_DIR}")
        sys.exit(1)
    try:
        save(rows)
    except PermissionError:
        print(f"Can't write {OUT_FILE}. If it's open in Excel, close it and run this again.")
        sys.exit(1)
    for line in summary(rows):
        print(line)
    print(f"Saved {len(rows)} runs to {OUT_FILE}")


if __name__ == "__main__":
    main()
