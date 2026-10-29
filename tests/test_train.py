# Tests for scripts/train.py. They use made-up rows of the combined table: a small made-up
# dataset where speed follows one setting, so a model that learned anything has to beat the
# formula on it.

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import train  # noqa: E402


def row(gpu="RTX 3090", measured=100, estimate=80, **columns):
    return {
        "hardware.gpu.name": gpu,
        "speed.decode_tok_s": str(measured),
        "estimate.decode_tok_s": str(estimate),
        "skip_reason": "",
    } | {key: str(value) for key, value in columns.items()}


def test_every_run_gets_one_value_per_input():
    assert train.features([row(), row()]).shape == (2, len(train.FEATURES))


def test_the_gap_counts_twice_and_half_as_fast_the_same():
    gaps = train.gap([row(measured=160, estimate=80), row(measured=40, estimate=80)])
    assert gaps[0] == pytest.approx(-gaps[1])


# Three GPUs whose speed relative to the formula depends only on flash attention: on, they write
# at 1.5x the formula, off, at 0.75x. The formula misses by a lot, one learned factor can't fix
# both, and a model that learns from the setting can.


def made_up_runs():
    return [
        row(gpu, 80 * (1.5 if flash else 0.75), 80, **{"settings.flash_attention": flash})
        for gpu in ["RTX 3090", "RTX 4090", "RTX 5090"]
        for flash in [True, False] * 10
    ]


def test_the_model_learns_what_the_formula_misses():
    pytest.importorskip("xgboost")
    rows = made_up_runs()
    errors = dict(
        (line[0], line[2:]) for line in train.score(rows, {"F": train.formula, "X": train.xgboost})
    )
    formula_error, model_error = (float(e.rstrip("%")) for e in errors["All GPUs"])
    assert model_error < 5 < formula_error


def test_the_input_it_leaned_on_is_named():
    pytest.importorskip("xgboost")
    assert train.importance(made_up_runs())[0][0] == "flash attention"
