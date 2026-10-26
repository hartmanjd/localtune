# Tests for the model tables and finding a model in them by the name people type into Ollama.

from localtune.models import find_model, load_model_table


def test_every_model_is_complete_ranked_and_listed_once():
    table = load_model_table()
    names = [model.name for model in table]
    assert len(names) == len(set(names))
    assert len({model.family for model in table}) <= 100
    for model in table:
        assert 1 <= model.popularity_rank <= 100 and model.pulls > 0
        assert 0 < model.active_params <= model.params
        assert model.layers > 0 and model.kv_heads > 0 and model.head_dim > 0
        assert model.files and model.files[0].ollama == model.name


def test_every_file_size_is_sensible_for_its_model():
    # A file holds between about 2 bits per parameter (Q2_K) and 16 (the unquantized original).
    for model in load_model_table():
        for f in model.files:
            assert 1.5 / 8 < f.size_bytes / model.params < 17 / 8, (model.name, f.quant)
            assert f.source


def test_finds_a_model_and_the_file_its_name_downloads():
    model, default = find_model("qwen3:14b")
    assert model.family == "qwen3"
    assert default.quant == "Q4_K_M"
    _, q8 = find_model("qwen3:14b-q8_0")
    assert q8.quant == "Q8_0" and q8.size_bytes > default.size_bytes
    assert model.file("q8_0") == q8


def test_a_name_without_a_size_means_ollamas_default_size():
    model, _ = find_model("qwen3")
    assert model.latest and model.family == "qwen3"
    assert find_model("qwen3:latest")[0] == model


def test_mixture_of_experts_models_use_part_of_their_parameters():
    model, _ = find_model("qwen3:30b")
    assert model.experts and model.experts_used
    assert model.active_params < model.params / 5


def test_unknown_models_are_not_guessed():
    assert find_model("some-future-model:7b") is None
