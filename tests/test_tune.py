# Tests for the settings "localtune tune" tries and how it picks the fastest.

import pytest

from localtune.cost import ElectricityPrice
from localtune.gguf import ModelLayout
from localtune.hardware import HardwareInfo
from localtune.results import EnergyCost, RunResult, RunSettings, Speed
from localtune.tune import (
    answer_seconds,
    as_good_as,
    combinations,
    default_run,
    describe,
    describe_split,
    fastest,
    find_best_split,
    first_guess,
    run_flags,
    server_context,
    server_flags,
    split_settings,
)

PC = HardwareInfo(os="Linux", is_wsl=False, cpu_name="CPU", cpu_threads=8, ram_gb=32.0, gpus=[])


def test_quantized_kv_caches_are_only_tried_with_flash_attention():
    combos = combinations()
    assert len(combos) == 16
    assert len({tuple(c.items()) for c in combos}) == 16
    for combo in combos:
        if combo["kv_cache"] != "f16":
            assert combo["flash_attention"]


def test_long_prompts_try_flash_attention_off_only_once_and_last():
    combos = combinations(prompt_tokens=16000)
    assert len(combos) == 13
    flash_off = [c for c in combos if not c["flash_attention"]]
    assert flash_off == [{"flash_attention": False, "kv_cache": "f16", "ubatch_size": 512}]
    assert combos[-1] == flash_off[0]
    assert len(combinations(prompt_tokens=4095)) == 16


def a_run(decode, status="ok", prefill=5000.0, **settings):
    price = ElectricityPrice(per_kwh=0.2, currency="USD", kind="yours", source="you")
    return RunResult(
        rig_id="rig",
        status=status,
        settings=RunSettings(engine="llama.cpp", model="qwen3:14b", **settings),
        hardware=PC,
        speed=Speed(
            prompt_tokens=512, generated_tokens=128, prefill_tok_s=prefill, decode_tok_s=decode
        ),
        energy=EnergyCost(
            kwh=None,
            kwh_per_million_tokens=None,
            cost=None,
            cost_per_million_tokens=None,
            price=price,
        ),
    )


def test_the_quickest_whole_answer_wins_and_failed_runs_dont_count():
    slow, quick = a_run(50.0), a_run(60.0)
    crashed = a_run(99.0, status="out_of_memory")
    assert fastest([slow, crashed, quick]) is quick
    assert fastest([crashed]) is None


def test_slow_reading_can_lose_to_slower_writing():
    # Reading 512 tokens at 50 tok/s takes 10 s, more than writing 128 tokens is ever saved.
    crawling_reader = a_run(130.0, prefill=50.0)
    steady = a_run(100.0, prefill=1500.0)
    assert fastest([crawling_reader, steady]) is steady
    assert answer_seconds(steady) == pytest.approx(512 / 1500 + 128 / 100)


def test_runs_within_noise_are_as_good_as_the_fastest():
    best, close, far = a_run(100.0), a_run(98.0), a_run(90.0)
    crashed = a_run(99.0, status="out_of_memory")
    assert as_good_as(best, [best, close, far, crashed]) == [close]


def test_finds_the_run_with_llama_cpps_usual_settings():
    tuned = a_run(110.0, flash_attention=True, kv_cache_k="q8_0", ubatch_size=1024)
    usual = a_run(100.0, flash_attention=True, kv_cache_k="f16", ubatch_size=512)
    assert default_run([tuned, usual]) is usual
    assert default_run([tuned]) is None


def test_the_winner_can_be_run_again():
    run = a_run(60.0, flash_attention=True, kv_cache_k="q8_0", ubatch_size=1024, gpu_layers=-1)
    assert run_flags(run) == "--flash-attn --kv-cache q8_0 --ubatch-size 1024"


def test_the_winner_becomes_a_llama_server_command():
    run = a_run(60.0, flash_attention=True, kv_cache_k="q8_0", ubatch_size=1024)
    fits = {"gpu_layers": None, "cpu_moe_layers": None}
    assert server_flags(run, fits, 640) == "-c 4096 -ngl 999 -fa on -ctk q8_0 -ctv q8_0 -ub 1024"
    split = {"gpu_layers": 49, "cpu_moe_layers": 6}
    flags = server_flags(run, split, 640, threads=8, gpu=1)
    assert flags.startswith("-c 4096 -ngl 49 --n-cpu-moe 6 -fa on")
    assert flags.endswith("-t 8 -mg 1")


def test_the_server_gets_room_for_the_tested_conversation():
    assert server_context(768) == 4096
    assert server_context(16256) == 16384
    assert server_context(8192) == 8192


def test_settings_are_described_in_plain_words():
    settings = {"flash_attention": False, "kv_cache": "f16", "ubatch_size": 512}
    assert describe(settings) == "flash attn off, KV f16, micro-batch 512"


# Splitting a model bigger than the GPU between the GPU and RAM.

DENSE = ModelLayout(layers=40, experts=None, expert_share=0.0)
MOE = ModelLayout(layers=48, experts=128, expert_share=0.9)
GB = 1024**3


def test_a_model_that_fits_is_not_split():
    assert first_guess(9 * GB, DENSE, vram_gb=16) is None


def test_first_guess_for_an_ordinary_model_is_the_layers_that_fit():
    # 41 parts of 0.5 GB each, 14.5 GB usable: 29 parts, one of them the output layer.
    guess = first_guess(int(20.5 * GB), DENSE, vram_gb=16)
    assert guess == 28
    assert split_settings(guess, DENSE) == {"gpu_layers": 28}
    assert describe_split(guess, DENSE) == "28 of 40 layers on the GPU"


def test_first_guess_for_a_mixture_of_experts_model_moves_only_experts():
    # 18 GB, of which 90% experts: 0.3375 GB of experts per layer, 3.5 GB too much: 11 layers.
    guess = first_guess(18 * GB, MOE, vram_gb=16)
    assert guess == 37
    assert split_settings(guess, MOE) == {"gpu_layers": 49, "cpu_moe_layers": 11}
    assert describe_split(guess, MOE) == "experts of 11 of 48 layers in RAM"


def test_a_long_conversation_leaves_less_room_for_the_model():
    # 1 MB per token: a 4,096-token conversation takes 4 GB of the GPU's memory, leaving 10.5
    # GB for 41 parts of 0.29 GB each: 35 parts, one of them the output layer.
    layout = DENSE.model_copy(update={"kv_bytes_per_token": 1024**2})
    assert first_guess(12 * GB, layout, vram_gb=16) is None
    assert first_guess(12 * GB, layout, vram_gb=16, context=4096) == 34


def test_all_layers_on_the_gpu_includes_the_output_layer():
    assert split_settings(40, DENSE) == {"gpu_layers": 41}
    assert describe_split(40, DENSE) == "the whole model on the GPU"
    assert describe_split(48, MOE) == "the whole model on the GPU"


def test_a_smaller_kv_cache_leaves_room_for_more_of_the_model():
    layout = ModelLayout(layers=40, experts=None, expert_share=0.0, kv_bytes_per_token=98304)
    f16 = first_guess(16 * GB, layout, vram_gb=16, context=16000)
    q8 = first_guess(16 * GB, layout, vram_gb=16, context=16000, kv_cache="q8_0")
    q4 = first_guess(16 * GB, layout, vram_gb=16, context=16000, kv_cache="q4_0")
    assert f16 < q8 < q4


def test_first_guess_never_goes_below_nothing_on_the_gpu():
    assert first_guess(500 * GB, DENSE, vram_gb=8) == 0
    assert first_guess(500 * GB, MOE, vram_gb=8) == 0


def times(curve):
    tried = []

    def try_split(on_gpu):
        tried.append(on_gpu)
        return curve(on_gpu)

    return try_split, tried


def test_best_split_climbs_while_more_on_the_gpu_is_faster():
    # Quicker answers with every layer on the GPU up to 23, then out of memory.
    try_split, tried = times(lambda n: None if n > 23 else 50 - n)
    assert find_best_split(try_split, 20, layers=40) == 23
    assert tried == [20, 21, 22, 23, 24]


def test_best_split_steps_down_when_the_guess_does_not_fit():
    try_split, tried = times(lambda n: None if n > 15 else 50 - n)
    assert find_best_split(try_split, 20, layers=40) == 15
    assert tried[:4] == [20, 18, 16, 14]


def test_best_split_backs_off_when_the_gpu_slows_down():
    # On Windows a too-full GPU borrows RAM by itself and gets slower instead of failing.
    try_split, _ = times(lambda n: 10 + abs(n - 17))
    assert find_best_split(try_split, 20, layers=40) == 17


def test_no_split_when_nothing_fits():
    try_split, tried = times(lambda n: None)
    assert find_best_split(try_split, 4, layers=40) is None
    assert tried == [4, 2, 0]
