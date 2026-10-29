# Tests for scripts/evaluate.py. They use a few made-up rows of the combined table, with speeds
# picked so each error works out by hand.

import csv
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate", SCRIPT)
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def row(gpu, measured, estimate, skip_reason=""):
    return {
        "hardware.gpu.name": gpu,
        "speed.decode_tok_s": str(measured),
        "estimate.decode_tok_s": str(estimate),
        "skip_reason": skip_reason,
    }


# Two GPUs: the formula guesses 80 for every run, but the 3090 really writes 100 and the 4090
# 160, so each one's runs are 1.25x and 2x the formula.
ROWS = [row("RTX 3090", 100, 80), row("RTX 3090", 100, 80), row("RTX 4090", 160, 80)]


def test_percent_error_is_how_far_off_on_average():
    rows = [row("x", 100, 0), row("x", 200, 0)]
    assert evaluate.percent_error([90, 220], rows) == 10


def test_formula_guesses_the_saved_estimate():
    assert evaluate.formula([], ROWS) == [80, 80, 80]


def test_each_gpu_is_predicted_only_from_the_others():
    seen = []

    def method(train, test):
        seen.append({r["hardware.gpu.name"] for r in train})
        return [0.0] * len(test)

    evaluate.leave_one_gpu_out(ROWS, method)
    assert seen == [{"RTX 4090"}, {"RTX 3090"}]


def test_calibrated_learns_the_other_gpus_factor():
    scores = evaluate.leave_one_gpu_out(ROWS, evaluate.calibrated)
    # The 3090 gets the 4090's factor of 2, and the 4090 the 3090's 1.25.
    assert scores["RTX 3090"][0] == [160, 160]
    assert scores["RTX 4090"][0] == [100]


def test_calibrated_needs_another_gpu_to_learn_from():
    assert evaluate.leave_one_gpu_out(ROWS[:2], evaluate.calibrated) == {}


def test_score_has_a_row_per_gpu_and_one_for_all():
    table = evaluate.score(ROWS, {"Formula": evaluate.formula})
    assert table == [["RTX 3090", "2", "20%"], ["RTX 4090", "1", "50%"], ["All GPUs", "3", "30%"]]


def test_only_usable_runs_with_an_estimate_are_scored(tmp_path):
    path = tmp_path / "all_runs.csv"
    rows = [row("x", 100, 80), row("x", 100, 80, "out_of_memory"), row("x", 100, "")]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    usable, without_estimate = evaluate.load_usable(path)
    assert len(usable) == 1
    assert without_estimate == 1
