# Tests for finding Ollama's model files. Each test builds a pretend Ollama folder with the
# same layout Ollama uses (manifests per model name, model files under blobs/), so Ollama
# itself isn't needed.

import json

import pytest

from localtune import ollama_models
from localtune.ollama_models import downloaded_models, find_ollama_model, manifest_path


@pytest.fixture
def ollama(tmp_path, monkeypatch):
    folder = tmp_path / "ollama"
    monkeypatch.setenv("OLLAMA_MODELS", str(folder))
    return folder


def add_model(folder, manifest_parts, digest="sha256:abc123", blob=True):
    manifest = folder.joinpath("manifests", *manifest_parts)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    layers = [
        {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:t"},
        {"mediaType": "application/vnd.ollama.image.model", "digest": digest},
    ]
    manifest.write_text(json.dumps({"layers": layers}))
    if blob:
        (folder / "blobs").mkdir(exist_ok=True)
        (folder / "blobs" / digest.replace(":", "-")).write_bytes(b"GGUF")
    return folder / "blobs" / digest.replace(":", "-")


def test_finds_a_models_file_by_its_name(ollama):
    model_file = add_model(ollama, ["registry.ollama.ai", "library", "qwen3", "14b"])
    assert find_ollama_model("qwen3:14b") == model_file


def test_a_name_without_a_tag_means_latest(ollama):
    model_file = add_model(ollama, ["registry.ollama.ai", "library", "llama3.2", "latest"])
    assert find_ollama_model("llama3.2") == model_file


def test_names_with_a_namespace_or_another_registry(ollama):
    assert manifest_path("someone/model:q4") == ollama.joinpath(
        "manifests", "registry.ollama.ai", "someone", "model", "q4"
    )
    assert manifest_path("hf.co/bartowski/Qwen3-8B-GGUF:Q4_K_M") == ollama.joinpath(
        "manifests", "hf.co", "bartowski", "Qwen3-8B-GGUF", "Q4_K_M"
    )


def test_unknown_or_missing_models_are_not_found(ollama):
    assert find_ollama_model("qwen3:14b") is None
    add_model(ollama, ["registry.ollama.ai", "library", "qwen3", "8b"], blob=False)
    assert find_ollama_model("qwen3:8b") is None


@pytest.mark.parametrize("name", ["", ":14b", "../../etc:x", "qwen3:../x"])
def test_odd_names_are_not_looked_up(ollama, name):
    assert manifest_path(name) is None


def test_lists_every_downloaded_model_once(ollama):
    add_model(ollama, ["registry.ollama.ai", "library", "qwen3", "8b"], digest="sha256:a")
    add_model(ollama, ["registry.ollama.ai", "library", "qwen3", "latest"], digest="sha256:a")
    add_model(ollama, ["registry.ollama.ai", "library", "qwen3", "14b"], digest="sha256:b")
    add_model(ollama, ["hf.co", "bartowski", "Qwen3-8B-GGUF", "Q4_K_M"], digest="sha256:c")
    add_model(ollama, ["registry.ollama.ai", "library", "cloudy", "cloud"], blob=False)
    assert downloaded_models() == [
        "hf.co/bartowski/Qwen3-8B-GGUF:Q4_K_M",
        "qwen3:14b",
        "qwen3:8b",
    ]


def test_no_ollama_folder_means_no_models(ollama):
    assert downloaded_models() == []


# Ollama's Linux installer keeps its models under its own "ollama" user, so those are found too,
# and a model in both places is listed once.
def test_finds_models_in_ollamas_linux_service_folder(tmp_path, monkeypatch):
    service = tmp_path / "service"
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(ollama_models.Path, "home", lambda: tmp_path / "home")
    monkeypatch.setattr(ollama_models, "LINUX_SERVICE_DIR", service)
    monkeypatch.setattr(ollama_models.sys, "platform", "linux")
    model_file = add_model(service, ["registry.ollama.ai", "library", "qwen3", "14b"])
    add_model(service, ["registry.ollama.ai", "library", "qwen3", "8b"], digest="sha256:b")
    add_model(
        tmp_path / "home" / ".ollama" / "models",
        ["registry.ollama.ai", "library", "qwen3", "8b"],
        digest="sha256:b",
    )
    assert find_ollama_model("qwen3:14b") == model_file
    assert downloaded_models() == ["qwen3:14b", "qwen3:8b"]

    monkeypatch.setattr(ollama_models.sys, "platform", "win32")
    assert find_ollama_model("qwen3:14b") is None
