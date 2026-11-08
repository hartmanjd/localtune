"""Find the model file behind an Ollama model name, so llama.cpp can benchmark it."""

# Ollama stores every model it downloads as a .gguf file, the same kind of file llama.cpp
# reads, but under a fingerprint name like sha256-a8cc... in its own folder. A small "manifest"
# file per model name (qwen3:14b) says which of those files is the model. Reading it lets
# "localtune run qwen3:14b" benchmark Ollama's copy with llama.cpp (runners/llamacpp.py)
# without anyone looking up the file by hand, and saves the run under the name people know.
# Only Ollama's files are read; Ollama itself doesn't need to be running.

import json
import os
import sys
from pathlib import Path

DEFAULT_REGISTRY = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
MODEL_LAYER = "application/vnd.ollama.image.model"


# Ollama's folder: the one its OLLAMA_MODELS setting points to, or .ollama/models in the home
# folder (on Windows, Linux and macOS alike). Models localtune downloads itself go there.
# On Linux, Ollama's own installer runs Ollama as a separate "ollama" user whose models live in
# /usr/share/ollama/.ollama/models, so that folder is searched too when nothing else is set.
LINUX_SERVICE_DIR = Path("/usr/share/ollama/.ollama/models")


def models_dir() -> Path:
    return Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")


def model_dirs() -> list[Path]:
    folders = [models_dir()]
    if not os.environ.get("OLLAMA_MODELS") and sys.platform.startswith("linux"):
        folders.append(LINUX_SERVICE_DIR)
    return folders


# A name like qwen3:14b is short for registry.ollama.ai/library/qwen3:14b, and a name with no
# tag means the "latest" tag; the manifest sits in a folder per part of that full name, and
# downloading (ollama_pull.py) asks the registry for the model by the same parts.


def full_name(name: str) -> list[str] | None:
    model, _, tag = name.partition(":")
    parts = model.split("/")
    if not model or "/" in tag or any(part in ("", ".", "..") for part in parts):
        return None
    if len(parts) == 1:
        parts = [DEFAULT_REGISTRY, DEFAULT_NAMESPACE, *parts]
    elif len(parts) == 2:
        parts = [DEFAULT_REGISTRY, *parts]
    return [*parts, tag or "latest"]


def manifest_path(name: str, folder: Path | None = None) -> Path | None:
    parts = full_name(name)
    return None if parts is None else (folder or models_dir()).joinpath("manifests", *parts)


def find_ollama_model(name: str) -> Path | None:
    for folder in model_dirs():
        model_file = _model_file(name, folder)
        if model_file is not None:
            return model_file
    return None


def _model_file(name: str, folder: Path) -> Path | None:
    manifest = manifest_path(name, folder)
    try:
        if manifest is None or not manifest.is_file():
            return None
        layers = json.loads(manifest.read_text())["layers"]
        digest = next(layer["digest"] for layer in layers if layer["mediaType"] == MODEL_LAYER)
        model_file = folder / "blobs" / digest.replace(":", "-")
        return model_file if model_file.is_file() else None
    except (OSError, json.JSONDecodeError, KeyError, TypeError, StopIteration):
        return None


# Every model downloaded with Ollama, by the name Ollama shows for it, for "localtune sweep".
# One file can have several names (qwen3:latest is often qwen3:8b), so each file is listed once.
# Models that only run in Ollama's cloud have no model file and are left out.


def downloaded_models() -> list[str]:
    found: dict[str, Path] = {}
    for folder in model_dirs():
        root = folder / "manifests"
        try:
            manifests = sorted(root.glob("*/*/*/*"))
        except OSError:
            continue
        for manifest in manifests:
            if not manifest.is_file():
                continue
            registry, namespace, model, tag = manifest.relative_to(root).parts
            if registry != DEFAULT_REGISTRY:
                name = f"{registry}/{namespace}/{model}:{tag}"
            elif namespace != DEFAULT_NAMESPACE:
                name = f"{namespace}/{model}:{tag}"
            else:
                name = f"{model}:{tag}"
            model_file = _model_file(name, folder)
            if name not in found and model_file is not None:
                found[name] = model_file
    names, seen = [], set()
    for name in sorted(found):
        if found[name] not in seen:
            seen.add(found[name])
            names.append(name)
    return names
