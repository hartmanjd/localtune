# Tests for the settings "localtune tune" tries and how it picks the fastest.

from localtune.cost import ElectricityPrice
from localtune.hardware import HardwareInfo
from localtune.results import EnergyCost, RunResult, RunSettings, Speed
from localtune.tune import combinations, describe, fastest, run_flags

PC = HardwareInfo(os="Linux", is_wsl=False, cpu_name="CPU", cpu_threads=8, ram_gb=32.0, gpus=[])


def test_quantized_kv_caches_are_only_tried_with_flash_attention():
    combos = combinations()
    assert len(combos) == 16
    assert len({tuple(c.items()) for c in combos}) == 16
    for combo in combos:
        if combo["kv_cache"] != "f16":
            assert combo["flash_attention"]


def a_run(decode, status="ok", **settings):
    price = ElectricityPrice(per_kwh=0.2, currency="USD", kind="yours", source="you")
    return RunResult(
        rig_id="rig",
        status=status,
        settings=RunSettings(engine="llama.cpp", model="qwen3:14b", **settings),
        hardware=PC,
        speed=Speed(prompt_tokens=512, generated_tokens=128, decode_tok_s=decode),
        energy=EnergyCost(
            kwh=None,
            kwh_per_million_tokens=None,
            cost=None,
            cost_per_million_tokens=None,
            price=price,
        ),
    )


def test_the_fastest_writer_wins_and_failed_runs_dont_count():
    slow, quick = a_run(50.0), a_run(60.0)
    crashed = a_run(99.0, status="out_of_memory")
    assert fastest([slow, crashed, quick]) is quick
    assert fastest([crashed]) is None


def test_the_winner_can_be_run_again():
    run = a_run(60.0, flash_attention=True, kv_cache_k="q8_0", ubatch_size=1024, gpu_layers=-1)
    assert run_flags(run) == "--flash-attn --kv-cache q8_0 --ubatch-size 1024"


def test_settings_are_described_in_plain_words():
    settings = {"flash_attention": False, "kv_cache": "f16", "ubatch_size": 512}
    assert describe(settings) == "flash attn off, KV f16, micro-batch 512"
