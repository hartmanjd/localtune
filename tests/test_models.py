# Tests for the model tables and finding a model in them by its own or its Ollama name.

from localtune.models import find_model, load_model_table


def test_every_model_is_complete_and_listed_once():
    table = load_model_table()
    names = [model.name for model in table]
    assert len(names) == len(set(names))
    for model in table:
        assert 0 < model.active_params <= model.params
        assert model.layers > 0 and model.kv_heads > 0 and model.head_dim > 0
        assert model.config_source
        assert model.files


def test_every_file_size_is_sensible_for_its_model():
    # A quantized file holds between about 2 and 9 bits per parameter.
    for model in load_model_table():
        for f in model.files:
            assert 2 / 8 < f.size_bytes / model.params < 9 / 8, (model.name, f.quant)
            assert f.source


def test_finds_a_model_by_its_ollama_name():
    model, ollama_file = find_model("qwen3:14b")
    assert model.name == "Qwen3-14B"
    assert ollama_file.ollama == "qwen3:14b"
    assert ollama_file.quant == "Q4_K_M"


def test_finds_a_model_by_its_own_name_with_every_quantization():
    model, ollama_file = find_model("qwen3-14b")
    assert ollama_file is None
    smaller, bigger = model.file("Q4_K_M"), model.file("Q8_0")
    assert smaller.size_bytes < bigger.size_bytes


def test_mixture_of_experts_models_use_part_of_their_parameters():
    model, _ = find_model("qwen3:30b")
    assert model.experts and model.experts_used
    assert model.active_params < model.params / 5


def test_unknown_models_are_not_guessed():
    assert find_model("some-future-model:7b") is None
