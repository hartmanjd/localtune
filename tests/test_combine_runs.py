# Tests for scripts/combine_runs.py. They save made-up runs into machine folders inside a
# temporary data/runs folder, the way real results files are copied in, and check what the
# combined table keeps, marks and counts.

import csv
import importlib.util
from pathlib import Path

from test_results import HARDWARE, SETTINGS, gpu_summary, make_run

from localtune.background import Background
from localtune.hardware import GpuInfo
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
    save_result(make_run(background=Background(gpu_vram_used_gb=6.0)), folder)
    save_result(make_run(background=Background(cpu_load_pct=80)), folder)
    save_result(make_run(gpu=gpu_summary(50).model_copy(update={"baseline_vram_gb": 5.0})), folder)
    reasons = [row["skip_reason"] for row in combine_runs.combine(tmp_path)]
    assert reasons[0] == ""
    assert reasons[1] == "out_of_memory"
    assert reasons[2] == "busy: 6.0 GB of GPU memory in use by other programs"
    assert reasons[3] == "busy: the CPU was 80% busy"
    assert reasons[4] == "busy: 5.0 GB of GPU memory in use at the start"


def test_the_end_of_the_run_before_doesnt_mark_a_run_busy(tmp_path):
    # Saved by older versions: the GPU looked 33% busy with nothing else holding its memory.
    background = Background(gpu_utilization_pct=33, gpu_vram_used_gb=0.0, cpu_load_pct=2)
    save_result(make_run(background=background), tmp_path / "rtx3090" / "results.jsonl")
    assert combine_runs.combine(tmp_path)[0]["skip_reason"] == ""


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


# Runs saved before localtune kept the model and GPU facts: a qwen3:14b run on an RTX 3090,
# both in localtune's tables.
OLD_HARDWARE = HARDWARE.model_copy(
    update={
        "gpus": [
            GpuInfo(index=0, name="NVIDIA GeForce RTX 3090", vram_gb=24.0, driver_version="580.00")
        ]
    }
)


def old_run(model="qwen3:14b", quant="Q4_K_M"):
    settings = SETTINGS.model_copy(update={"model": model, "quant": quant})
    return make_run(settings=settings, hardware=OLD_HARDWARE)


def test_missing_facts_are_filled_in_from_the_tables(tmp_path):
    save_result(old_run(), tmp_path / "my-pc" / "results.jsonl")
    row = combine_runs.combine(tmp_path)[0]
    assert row["facts_filled"] is True
    assert row["model_facts.file_bytes"] == 9276184896
    assert row["gpu_facts.memory_bandwidth_gbs"] == 936.0
    assert row["estimate.decode_tok_s"] is not None


def test_a_model_is_only_filled_in_for_the_quantization_it_used(tmp_path):
    save_result(old_run(quant="Q8_0"), tmp_path / "my-pc" / "results.jsonl")
    row = combine_runs.combine(tmp_path)[0]
    assert row["model_facts"] is None
    assert row["gpu_facts.name"] == "RTX 3090"


def test_a_model_the_tables_dont_know_stays_empty(tmp_path):
    save_result(old_run(model="my-own-model.gguf"), tmp_path / "my-pc" / "results.jsonl")
    row = combine_runs.combine(tmp_path)[0]
    assert row["model_facts"] is None
    assert row["estimate.decode_tok_s"] is None
