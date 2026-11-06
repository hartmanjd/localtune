"""Save a checkpoint: how big the dataset is and how well the trained model scores, today.

Run from the project folder, after scripts/combine_runs.py:
    uv run python scripts/checkpoint.py "5 home PCs shared their runs"
"""

# Every time a new batch of runs is added and the model retrained, the numbers on the website's
# data page change. A checkpoint keeps the numbers of that day in a file of its own in
# data/checkpoints/, named by its date, so the page can show how the data and the model's
# accuracy looked at each point in time, like a list of commits. A checkpoint is never
# changed once saved: a later batch makes a new one.
#
# The scores are the same leave-one-GPU-out test as train.py: each GPU in turn is left out, the
# model is trained on the others, and it predicts the one left out. They are split into runs
# where the whole model fit on the graphics card and runs where part of it sat in system RAM,
# because the model is much better at the first.

import csv
import json
import statistics
import sys
from datetime import date
from pathlib import Path

from evaluate import GPU, ROOT, RUNS_FILE, TARGET, console, leave_one_gpu_out, load_usable
from train import xgboost

from localtune.features import worked_out

CHECKPOINTS = ROOT / "data" / "checkpoints"


# A run "spilled" when part of the model sat in system RAM: fewer layers on the GPU than the
# model has, or some of a mixture-of-experts model's experts kept on the CPU.


def spilled(row: dict) -> bool:
    cpu_moe = row.get("settings.cpu_moe_layers") or "0"
    return worked_out(row)["share of layers on GPU"] < 1 or float(cpu_moe) > 0


def misses(guesses: list[float], rows: list[dict]) -> list[float]:
    return [
        100 * abs(g - float(r[TARGET])) / float(r[TARGET])
        for g, r in zip(guesses, rows, strict=True)
    ]


def average(values: list[float]) -> float | None:
    return round(statistics.mean(values), 1) if values else None


# The counts come from every run collected, including ones set aside for training (failed, or
# the computer was busy); the scores only from the usable ones.


def make(collected: list[dict], usable: list[dict], note: str, day: str) -> dict:
    scored = leave_one_gpu_out(usable, xgboost)
    rows = [r for _, test in scored.values() for r in test]
    every = misses([g for guesses, _ in scored.values() for g in guesses], rows)
    return {
        "date": day,
        "note": note,
        "runs": len(collected),
        "usable_runs": len(usable),
        "machines": len({r["machine"] for r in collected}),
        "cards": len({r[GPU] for r in collected}),
        "models": len({r["settings.model"] for r in collected}),
        "miss": average(every),
        "miss_fits": average([m for m, r in zip(every, rows, strict=True) if not spilled(r)]),
        "miss_spills": average([m for m, r in zip(every, rows, strict=True) if spilled(r)]),
        "median_miss": round(statistics.median(every), 1),
        "within_15": round(100 * sum(m <= 15 for m in every) / len(every), 1),
        "gpus": {
            gpu: {"runs": len(test), "miss": average(misses(guesses, test))}
            for gpu, (guesses, test) in sorted(scored.items())
        },
    }


def save(checkpoint: dict, folder: Path = CHECKPOINTS) -> Path:
    path = folder / f"{checkpoint['date']}.json"
    if path.exists():
        raise FileExistsError(f"{path.name} already exists; a checkpoint is never changed.")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    note = " ".join(sys.argv[1:]).strip()
    if not note:
        console.print('Say what changed, for example: checkpoint.py "5 home PCs shared their runs"')
        sys.exit(1)
    if not RUNS_FILE.exists():
        console.print(
            "No combined table yet. Run this first: uv run python scripts/combine_runs.py"
        )
        sys.exit(1)
    with RUNS_FILE.open(encoding="utf-8-sig") as f:
        collected = list(csv.DictReader(f))
    usable, _ = load_usable(RUNS_FILE)
    checkpoint = make(collected, usable, note, date.today().isoformat())
    try:
        path = save(checkpoint)
    except FileExistsError as error:
        console.print(str(error))
        sys.exit(1)
    console.print(
        f"Saved {path.name}: {checkpoint['runs']} runs, {checkpoint['cards']} cards, "
        f"average miss {checkpoint['miss']}%",
        markup=False,
    )


if __name__ == "__main__":
    main()
