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


def test_a_laptop_gpu_is_its_own_row_never_the_desktop_card():
    laptop = find_gpu_spec("NVIDIA GeForce RTX 4070 Laptop GPU", 8.0)
    assert laptop.name == "RTX 4070 Laptop GPU" and laptop.memory_bandwidth_gbs == 256
    assert find_gpu_spec("NVIDIA GeForce RTX 4070", 12.0).memory_bandwidth_gbs == 504
    assert find_gpu_spec("NVIDIA GeForce RTX 3080 Laptop GPU", 8.0).vram_gb == 8
    assert find_gpu_spec("NVIDIA GeForce RTX 5070 Laptop GPU", 12.0).vram_gb == 12
    assert find_gpu_spec("NVIDIA GeForce RTX 3050 6GB Laptop GPU", 6.0).vram_gb == 6


def test_small_laptop_and_unknown_gpus_are_not_guessed():
    assert find_gpu_spec("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0) is None  # the 4 GB one
    assert find_gpu_spec("NVIDIA GeForce RTX 3050 Ti Laptop GPU", 4.0) is None
    assert find_gpu_spec("Some Future GPU", 16.0) is None


def test_a_name_alone_picks_the_version_with_the_most_memory():
    assert find_gpu_spec("RTX 3060").vram_gb == 12
