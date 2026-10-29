# Tests for features.py: the inputs the trained model sees, worked out by hand from made-up
# rows of the combined table, and from a row built in memory with plain values.

import math

from localtune.features import FEATURES, inputs, worked_out


def row(**columns):
    return {key: str(value) for key, value in columns.items()}


def test_inputs_are_worked_out_from_the_columns():
    found = worked_out(
        row(
            **{
                "model_facts.layers": 40,
                "settings.gpu_layers": 30,
                "settings.kv_cache_k": "q8_0",
                "settings.flash_attention": True,
                "model_facts.params": 1000,
                "model_facts.file_bytes": 500,
                "hardware.gpu.power_limit_w": 280,
                "hardware.gpu.default_power_limit_w": 350,
                "settings.engine": "llama.cpp",
                "hardware.os": "Windows 11",
            }
        )
    )
    assert found["share of layers on GPU"] == 0.75
    assert found["KV cache size"] == 34 / 64
    assert found["flash attention"] == 1.0
    assert found["bits per number"] == 4.0
    assert found["decode depth"] == 0.0
    assert found["power vs usual"] == 0.8
    assert found["Ollama"] == 0.0
    assert found["Windows"] == 1.0


def test_all_layers_on_gpu_and_missing_columns():
    found = worked_out(row(**{"settings.gpu_layers": -1}))
    assert found["share of layers on GPU"] == 1.0
    assert found["KV cache size"] == 1.0
    assert math.isnan(found["bits per number"])


def test_plain_values_count_the_same_as_text():
    plain = {"settings.flash_attention": True, "model_facts.layers": 40, "settings.gpu_layers": 20}
    as_text = row(**plain)
    assert all(
        a == b or (math.isnan(a) and math.isnan(b))
        for a, b in zip(inputs(plain), inputs(as_text), strict=True)
    )
    assert len(inputs(plain)) == len(FEATURES)
