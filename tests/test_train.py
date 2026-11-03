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
        "estimate.own_ram_decode_tok_s": str(estimate),
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
    rows = made_up_runs()
    errors = dict(
        (line[0], line[2:]) for line in train.score(rows, {"F": train.formula, "X": train.xgboost})
    )
    formula_error, model_error = (float(e.rstrip("%")) for e in errors["All GPUs"])
    assert model_error < 5 < formula_error


def test_the_input_it_leaned_on_is_named():
    assert train.importance(made_up_runs())[0][0] == "flash attention"


# A saved model is the one 'localtune predict' reads (speed_model.py): its guess for a run is
# the formula's estimate times what it learned, so a run with flash attention on comes out at
# about 1.5x the estimate.


def test_a_saved_model_guesses_like_the_trained_one(tmp_path, monkeypatch):
    from localtune import speed_model

    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    assert train.save(made_up_runs()) == tmp_path / "speed-model.json"
    guess = speed_model.trained_estimate(row(**{"settings.flash_attention": True}))
    assert guess == pytest.approx(120, rel=0.05)


def test_a_model_saved_with_other_inputs_is_ignored(tmp_path, monkeypatch):
    import numpy as np
    import xgboost

    from localtune import speed_model

    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    data = xgboost.DMatrix(np.zeros((2, 1)), label=[0, 0], feature_names=["something else"])
    xgboost.train({}, data, num_boost_round=1).save_model(str(tmp_path / "speed-model.json"))
    assert speed_model.trained_estimate(row()) is None
