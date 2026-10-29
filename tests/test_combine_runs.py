# Tests for scripts/combine_runs.py. They save made-up runs into machine folders inside a
# temporary data/runs folder, the way real results files are copied in, and check what the
# combined table keeps, marks and counts.

import csv
import importlib.util
from pathlib import Path

from test_results import gpu_summary, make_run

from localtune.background import Background
from localtune.results import save_result

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "combine_runs.py"
spec = importlib.util.spec_from_file_location("combine_runs", SCRIPT)
combine_runs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combine_runs)


def test_every_machine_folder_is_read_with_its_name(tmp_path):
    save_result(make_run(), tmp_path / "rtx3090" / "results.jsonl")
    save_result(make_run(), tmp_path / "my-pc" / "results.jsonl")
    rows = combine_runs.combine(tmp_path)
    assert sorted(row["machine"] for row in rows) == ["my-pc", "rtx3090"]
    assert rows[0]["speed.decode_tok_s"] == 90


def test_a_run_copied_twice_is_kept_once(tmp_path):
    run = make_run()
    save_result(run, tmp_path / "rtx3090" / "results.jsonl")
    save_result(run, tmp_path / "rtx3090-again" / "results.jsonl")
    assert len(combine_runs.combine(tmp_path)) == 1


def test_failed_and_busy_runs_are_kept_but_marked(tmp_path):
    folder = tmp_path / "my-pc" / "results.jsonl"
    save_result(make_run(), folder)
    save_result(make_run(status="out_of_memory", speed=None), folder)
    save_result(make_run(background=Background(gpu_utilization_pct=60)), folder)
    save_result(make_run(gpu=gpu_summary(50).model_copy(update={"baseline_vram_gb": 5.0})), folder)
    reasons = [row["skip_reason"] for row in combine_runs.combine(tmp_path)]
    assert reasons[0] == ""
    assert reasons[1] == "out_of_memory"
    assert reasons[2].startswith("busy: the GPU is already 60% busy")
    assert reasons[3].startswith("busy: 5.0 GB")


def test_summary_counts_usable_runs_per_machine(tmp_path):
    folder = tmp_path / "my-pc" / "results.jsonl"
    save_result(make_run(), folder)
    save_result(make_run(status="out_of_memory", speed=None), folder)
    lines = combine_runs.summary(combine_runs.combine(tmp_path))
    assert lines == ["my-pc: Test GPU, 2 runs, 1 usable for training"]


def test_saved_table_has_every_column(tmp_path):
    save_result(make_run(), tmp_path / "runs" / "my-pc" / "results.jsonl")
    out = tmp_path / "all_runs.csv"
    combine_runs.save(combine_runs.combine(tmp_path / "runs"), out)
    with out.open(encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["machine"] == "my-pc"
    assert row["skip_reason"] == ""
    assert "settings.kv_cache_k" in row
