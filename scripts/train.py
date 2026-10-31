"""Train an XGBoost model to predict writing speed, and score it against the formula.

Run from the project folder, after scripts/combine_runs.py:
    uv run --group train python scripts/train.py
Add --save to also keep a model trained on every run, for 'localtune predict' on this computer.
"""

# The formula in 'localtune predict' already gets the rough size of a model's writing speed
# right from physics: memory bandwidth divided by the bytes read for each word. What it misses
# is everything that makes real runs faster or slower than that: the settings, how much of the
# model sits in RAM, how the model is built, how long the conversation already is. So the
# model here doesn't learn the speed from scratch. It learns the gap: how many times faster or
# slower than the formula each run really was. Its guess is then the formula's estimate times
# that learned factor. Starting from physics this way needs far less data than learning speed
# alone, and a GPU the model never saw still gets a sensible answer from its spec sheet.
#
# The formula it starts from (START) is the one 'localtune predict' shows, except that the part
# of a model in RAM is read at the run's own CPU's RAM speed (predict.estimate_for) instead of
# one fixed speed for every computer. That makes it part of this model, not of the yardstick:
# "Formula" in the scores is still the plain formula, unchanged.
#
# It is scored exactly like the formula (evaluate.py): each GPU in turn is left out, the model
# is trained on every other GPU's runs, and it predicts the one left out.

import math
import sys
from pathlib import Path

import numpy as np
from evaluate import (
    GPU,
    RUNS_FILE,
    TARGET,
    Method,
    console,
    formula,
    load_usable,
    score,
)
from rich.table import Table

from localtune.features import FEATURES, inputs
from localtune.speed_model import START, model_file

# The inputs the model sees for each run are worked out in features.py, so that training and
# 'localtune predict' build them the same way.


def features(rows: list[dict]) -> np.ndarray:
    return np.array([inputs(row) for row in rows])


# What the model learns for each run: how many times faster or slower than the formula it was,
# as a logarithm, so twice as fast and half as fast count as the same size of miss.


def gap(rows: list[dict]) -> np.ndarray:
    return np.array([math.log(float(r[TARGET]) / float(r[START])) for r in rows])


# The model's settings, kept small and fixed (no searching for the best ones, which would need
# a separate set of runs to judge them on): a few hundred shallow trees, each correcting the
# last a little. Shallow trees and the random sampling of runs and inputs keep it from simply
# memorising the few GPUs it has seen. The seed makes every training give the same result.

SETTINGS = {
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "seed": 0,
}
TREES = 300


def fit(train: list[dict]):
    import xgboost

    data = xgboost.DMatrix(features(train), label=gap(train), feature_names=FEATURES)
    return xgboost.train(SETTINGS, data, num_boost_round=TREES)


def xgboost(train: list[dict], test: list[dict]) -> list[float] | None:
    if not train:
        return None
    import xgboost

    data = xgboost.DMatrix(features(test), feature_names=FEATURES)
    factors = np.exp(fit(train).predict(data))
    return [float(row[START]) * float(f) for row, f in zip(test, factors, strict=True)]


# Which inputs the model leaned on most, from a model trained on every usable run: how much
# each one improved its guesses overall (XGBoost's "gain"), as a share of the total.


def importance(rows: list[dict], top: int = 10) -> list[tuple[str, float]]:
    gains = fit(rows).get_score(importance_type="total_gain")
    total = sum(gains.values()) or 1.0
    return sorted(((name, v / total) for name, v in gains.items()), key=lambda x: -x[1])[:top]


# The model 'localtune predict' uses on this computer (speed_model.py): trained on every usable
# run, saved in localtune's own folder rather than the project, so it stays off GitHub. Saving
# again replaces it with one trained on the runs there are now.


def save(rows: list[dict], path: Path | None = None) -> Path:
    path = path or model_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fit(rows).save_model(str(path))
    return path


def main() -> None:
    if not RUNS_FILE.exists():
        console.print(
            "No combined table yet. Run this first: uv run python scripts/combine_runs.py"
        )
        sys.exit(1)
    rows, _ = load_usable(RUNS_FILE)
    if not rows:
        console.print("No usable runs with an estimate to train on yet.")
        sys.exit(1)
    methods: dict[str, Method] = {"Formula": formula, "XGBoost": xgboost}
    table = Table(title="Writing speed (decode tok/s), each GPU predicted without its own runs")
    table.add_column("GPU")
    for column in ["Runs"] + [f"{name} error" for name in methods]:
        table.add_column(column, justify="right")
    for line in score(rows, methods):
        table.add_row(*line)
    console.print(table)

    leaned = Table(title="What the model leaned on most, trained on every run")
    leaned.add_column("Input")
    leaned.add_column("Share", justify="right")
    for name, share in importance(rows):
        leaned.add_row(name, f"{share:.0%}")
    console.print(leaned)
    gpus = len({row[GPU] for row in rows})
    console.print(
        f"Trained on {len(rows)} runs from {gpus} GPUs. With few GPUs, the inputs that describe "
        "the GPU barely vary, so the model can't learn much about how GPUs differ yet."
    )
    if "--save" in sys.argv[1:]:
        console.print(f"Saved the model trained on every run to {save(rows)}", markup=False)


if __name__ == "__main__":
    main()
