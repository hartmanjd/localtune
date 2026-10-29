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
from localtune.predict import estimate_for
from localtune.results import RunResult, flat_record, load_results

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"
OUT_FILE = ROOT / "data" / "all_runs.csv"


# Which runs can be trusted for training. Every run is kept in the table, but a run that failed,
# or that shared the computer with another program, gets a reason in the skip_reason column, so
# training can leave it out while the table still shows what was measured. Failed runs still
# matter later: they show where a model stops fitting. Another program on the GPU is judged by
# the memory it holds, not by how busy the GPU looked: runs saved before localtune took the
# quietest of its readings often caught the end of the run before them, and any program using
# the GPU holds some of its memory.


def skip_reason(run: RunResult) -> str:
    if run.status != "ok":
        return run.status
    b = run.background
    if b and b.gpu_vram_used_gb is not None and b.gpu_vram_used_gb >= OTHER_VRAM_GB:
        return f"busy: {b.gpu_vram_used_gb:.1f} GB of GPU memory in use by other programs"
    if b and b.cpu_load_pct is not None and b.cpu_load_pct >= BUSY_CPU_PCT:
        return f"busy: the CPU was {b.cpu_load_pct:.0f}% busy"
    if run.gpu and run.gpu.baseline_vram_gb >= OTHER_VRAM_GB:
        return f"busy: {run.gpu.baseline_vram_gb:.1f} GB of GPU memory in use at the start"
    return ""


# Reading every machine's file. The same file copied into two folders, or copied again after
# more runs were added, would repeat runs, so each run_id is only kept the first time.


def combine(runs_dir: Path = RUNS_DIR) -> list[dict]:
    rows, seen = [], set()
    for path in sorted(runs_dir.glob("*/*.jsonl")):
        for run in load_results(path):
            if run.run_id in seen:
                continue
            seen.add(run.run_id)
            rows.append(
                {"machine": path.parent.name}
                | flat_record(run)
                | {"estimate.decode_tok_s": estimate_for(run), "skip_reason": skip_reason(run)}
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
    save(rows)
    for line in summary(rows):
        print(line)
    print(f"Saved {len(rows)} runs to {OUT_FILE}")


if __name__ == "__main__":
    main()
