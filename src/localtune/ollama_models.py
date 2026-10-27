"""Find the model file behind an Ollama model name, so llama.cpp can benchmark it."""

# Ollama stores every model it downloads as a .gguf file, the same kind of file llama.cpp
# reads, but under a fingerprint name like sha256-a8cc... in its own folder. A small "manifest"
# file per model name (qwen3:14b) says which of those files is the model. Reading it lets
# "localtune run qwen3:14b" benchmark Ollama's copy with llama.cpp (runners/llamacpp.py)
# without anyone looking up the file by hand, and saves the run under the name people know.
# Only Ollama's files are read; Ollama itself doesn't need to be running.

import json
import os
from pathlib import Path

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
MODEL_LAYER = "application/vnd.ollama.image.model"


# Ollama's folder: the one its OLLAMA_MODELS setting points to, or .ollama/models in the home
# folder (on Windows, Linux and macOS alike).


def models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


# A name like qwen3:14b is short for registry.ollama.ai/library/qwen3:14b, and a name with no
# tag means the "latest" tag; the manifest sits in a folder per part of that full name.


def manifest_path(name: str) -> Path | None:
    model, _, tag = name.partition(":")
    parts = model.split("/")
    if not model or "/" in tag or any(part in ("", ".", "..") for part in parts):
        return None
    if len(parts) == 1:
        parts = [DEFAULT_REGISTRY, DEFAULT_NAMESPACE, *parts]
    elif len(parts) == 2:
        parts = [DEFAULT_REGISTRY, *parts]
    return models_dir().joinpath("manifests", *parts, tag or "latest")


def find_ollama_model(name: str) -> Path | None:
    manifest = manifest_path(name)
    if manifest is None or not manifest.is_file():
        return None
    try:
        layers = json.loads(manifest.read_text())["layers"]
        digest = next(layer["digest"] for layer in layers if layer["mediaType"] == MODEL_LAYER)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, StopIteration):
        return None
    model_file = models_dir() / "blobs" / digest.replace(":", "-")
    return model_file if model_file.is_file() else None


# Every model downloaded with Ollama, by the name Ollama shows for it, for "localtune sweep".
# One file can have several names (qwen3:latest is often qwen3:8b), so each file is listed once.
# Models that only run in Ollama's cloud have no model file and are left out.


def downloaded_models() -> list[str]:
    root = models_dir() / "manifests"
    names, seen = [], set()
    for manifest in sorted(root.glob("*/*/*/*")):
        if not manifest.is_file():
            continue
        registry, namespace, model, tag = manifest.relative_to(root).parts
        if registry != DEFAULT_REGISTRY:
            name = f"{registry}/{namespace}/{model}:{tag}"
        elif namespace != DEFAULT_NAMESPACE:
            name = f"{namespace}/{model}:{tag}"
        else:
            name = f"{model}:{tag}"
        model_file = find_ollama_model(name)
        if model_file is not None and model_file not in seen:
            seen.add(model_file)
            names.append(name)
    return names
