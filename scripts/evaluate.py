"""Score ways of predicting writing speed, each GPU predicted without its own runs.

Run from the project folder, after scripts/combine_runs.py:
    uv run python scripts/evaluate.py
"""

# The prediction model has one job: guess how fast a model will write (decode tok/s) on a GPU
# nobody has measured yet. The fair test of that is "leave one GPU out": every GPU in the
# combined table (data/all_runs.csv) takes a turn being the unseen one. A method learns from
# the runs on every other GPU, then predicts that GPU's runs, and its error is how far those
# guesses were from what was measured. Testing on runs from a GPU the method already learned
# from would only show how well it remembers that card.
#
# This scores the baselines a trained model has to beat:
#   formula      'localtune predict' as it ships (predict.py): memory bandwidth divided by the
#                bytes read for each word, times a fixed EFFICIENCY. It learns nothing, so it
#                is the honest starting point.
#   calibrated   the same formula with its efficiency learned from the other GPUs' runs, the
#                simplest possible use of the data.

import csv
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from localtune.predict import EFFICIENCY

ROOT = Path(__file__).resolve().parent.parent
RUNS_FILE = ROOT / "data" / "all_runs.csv"
console = Console()

TARGET = "speed.decode_tok_s"
ESTIMATE = "estimate.decode_tok_s"
GPU = "hardware.gpu.name"

# A method gets the training runs and the runs to predict, and returns one guess per run to
# predict, or None when it has nothing to learn from. The trained model will be one more
# method of this shape.
Method = Callable[[list[dict], list[dict]], list[float] | None]


# Which runs are scored: ones combine_runs.py found usable (they worked, and nothing else was
# using the computer), with a measured speed and an estimate. Runs saved before localtune kept
# the model and GPU facts have no estimate, so they can't be compared and are counted apart.


def load_usable(path: Path = RUNS_FILE) -> tuple[list[dict], int]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if not row["skip_reason"]]
    usable = [row for row in rows if row.get(TARGET) and row.get(ESTIMATE)]
    return usable, len(rows) - len(usable)


def formula(train: list[dict], test: list[dict]) -> list[float]:
    return [float(row[ESTIMATE]) for row in test]


def calibrated(train: list[dict], test: list[dict]) -> list[float] | None:
    if not train:
        return None
    return [float(row[ESTIMATE]) * learned_factor(train) for row in test]


# How much faster or slower than the formula the runs really were, as one number: the median of
# measured divided by estimated, so a few odd runs can't pull it around. The formula already
# includes EFFICIENCY, so this factor times EFFICIENCY is the efficiency the data points to.


def learned_factor(train: list[dict]) -> float:
    return statistics.median(float(row[TARGET]) / float(row[ESTIMATE]) for row in train)


# The error of a set of guesses: on average, how many percent each one was off the measured
# speed, whether too high or too low (mean absolute percentage error). 10% means a guess of
# 90 or 110 tok/s for a run that measured 100.


def percent_error(guesses: list[float], rows: list[dict]) -> float:
    measured = [float(row[TARGET]) for row in rows]
    return 100 * statistics.mean(abs(g - m) / m for g, m in zip(guesses, measured, strict=True))


# Leaving one GPU out at a time, and the table of errors: one row per GPU, for the runs it had
# to be predicted for, and one for every run together.


def leave_one_gpu_out(rows: list[dict], method: Method) -> dict[str, tuple[list[float], list]]:
    by_gpu = defaultdict(list)
    for row in rows:
        by_gpu[row[GPU]].append(row)
    scores = {}
    for gpu, test in by_gpu.items():
        guesses = method([row for row in rows if row[GPU] != gpu], test)
        if guesses is not None:
            scores[gpu] = (guesses, test)
    return scores


def score(rows: list[dict], methods: dict[str, Method]) -> list[list[str]]:
    gpus = sorted({row[GPU] for row in rows})
    results = {name: leave_one_gpu_out(rows, method) for name, method in methods.items()}
    table = []
    for gpu in gpus + ["All GPUs"]:
        everyone = gpu == "All GPUs"
        line = [gpu, str(len(rows) if everyone else sum(1 for row in rows if row[GPU] == gpu))]
        for name in methods:
            scored = results[name]
            if everyone:
                guesses = [g for gs, _ in scored.values() for g in gs]
                tested = [row for _, test in scored.values() for row in test]
            else:
                guesses, tested = scored.get(gpu, ([], []))
            line.append(f"{percent_error(guesses, tested):.0f}%" if tested else "")
        table.append(line)
    return table


def main() -> None:
    if not RUNS_FILE.exists():
        print("No combined table yet. Run this first: uv run python scripts/combine_runs.py")
        sys.exit(1)
    rows, without_estimate = load_usable(RUNS_FILE)
    if not rows:
        console.print("No usable runs with an estimate to score yet.")
        sys.exit(1)
    methods = {"Formula": formula, "Calibrated": calibrated}
    table = Table(title="Writing speed (decode tok/s), each GPU predicted without its own runs")
    table.add_column("GPU")
    for column in ["Runs"] + [f"{name} error" for name in methods]:
        table.add_column(column, justify="right")
    for line in score(rows, methods):
        table.add_row(*line)
    console.print(table)
    console.print(
        f"The formula uses an efficiency of {EFFICIENCY}; all runs together point to "
        f"{EFFICIENCY * learned_factor(rows):.2f}."
    )
    if len({row[GPU] for row in rows}) < 2:
        console.print(
            "Calibrated needs runs from at least two GPUs: one to learn from, one to test."
        )
    if without_estimate:
        console.print(
            f"{without_estimate} usable runs have no estimate (saved before localtune kept "
            "the model and GPU facts), so they aren't scored."
        )


if __name__ == "__main__":
    main()
