# Tests for scripts/checkpoint.py, on made-up rows of the combined table: three GPUs, one of
# them with a run that spilled into system RAM and one run set aside as unusable.

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import checkpoint  # noqa: E402


def row(gpu="RTX 3090", measured=100, estimate=80, machine="a", **columns):
    return {
        "machine": machine,
        "hardware.gpu.name": gpu,
        "settings.model": "qwen3:14b",
        "speed.decode_tok_s": str(measured),
        "estimate.decode_tok_s": str(estimate),
        "estimate.own_ram_decode_tok_s": str(estimate),
        "skip_reason": "",
    } | {key: str(value) for key, value in columns.items()}


def made_up_runs():
    return [
        row(
            gpu,
            80 * (1.5 if flash else 0.75),
            80,
            machine=gpu,
            **{"settings.flash_attention": flash},
        )
        for gpu in ["RTX 3090", "RTX 4090", "RTX 5090"]
        for flash in [True, False] * 10
    ]


def test_a_run_with_layers_or_experts_off_the_gpu_spilled():
    assert not checkpoint.spilled(row())
    assert checkpoint.spilled(row(**{"settings.gpu_layers": 20, "model_facts.layers": 40}))
    assert checkpoint.spilled(row(**{"settings.cpu_moe_layers": 6}))


def test_counts_include_set_aside_runs_and_scores_only_usable_ones():
    usable = made_up_runs()
    collected = usable + [row("RTX 3070", machine="busy", skip_reason="busy")]
    made = checkpoint.make(collected, usable, "First batch", "2026-10-31")
    assert (made["runs"], made["usable_runs"], made["cards"], made["machines"]) == (61, 60, 4, 4)
    assert made["models"] == 1
    assert set(made["gpus"]) == {"RTX 3090", "RTX 4090", "RTX 5090"}
    assert made["miss"] < 5 and made["within_15"] == 100
    assert made["miss_spills"] is None


def test_a_checkpoint_is_never_overwritten(tmp_path):
    made = {"date": "2026-10-31", "note": "First batch"}
    path = checkpoint.save(made, tmp_path)
    assert json.loads(path.read_text()) == made
    with pytest.raises(FileExistsError):
        checkpoint.save(made | {"note": "changed"}, tmp_path)
