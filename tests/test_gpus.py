# Tests for the GPU table and finding a GPU in it by the name nvidia-smi reports.

import pytest

from localtune.gpus import find_gpu_spec, load_gpu_table


def test_every_row_is_complete_and_listed_once():
    table = load_gpu_table()
    keys = [(spec.name, spec.vram_gb) for spec in table]
    assert len(keys) == len(set(keys))
    for spec in table:
        assert 0 < spec.vram_gb <= 96
        assert 100 <= spec.memory_bandwidth_gbs <= 4000
        assert 1 <= spec.fp32_tflops <= 200
        assert 50 <= spec.board_power_w <= 1000
        assert spec.source


@pytest.mark.parametrize(
    "name, vram_gb, bandwidth",
    [
        ("NVIDIA GeForce RTX 4070 Ti SUPER", 16.0, 672),
        ("NVIDIA GeForce RTX 4070 Ti", 12.0, 504),
        ("NVIDIA GeForce RTX 3060", 12.0, 360),
        ("NVIDIA GeForce RTX 3060", 8.0, 240),
        ("Tesla T4", 15.0, 320),
        ("NVIDIA A100-SXM4-80GB", 80.0, 2039),
    ],
)
def test_finds_a_gpu_by_its_nvidia_smi_name(name, vram_gb, bandwidth):
    assert find_gpu_spec(name, vram_gb).memory_bandwidth_gbs == bandwidth


def test_laptop_and_unknown_gpus_are_not_guessed():
    assert find_gpu_spec("NVIDIA GeForce RTX 4070 Laptop GPU", 8.0) is None
    assert find_gpu_spec("Some Future GPU", 16.0) is None
