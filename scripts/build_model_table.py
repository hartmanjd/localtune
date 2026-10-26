"""Build localtune's model tables from Hugging Face and Ollama.

Run from the project folder:
    python scripts/build_model_table.py
"""

# localtune ships two model tables (src/localtune/data/) so it can tell, without downloading
# anything, how big a model is and how it is built: models.csv has one row per model, and
# model_files.csv one row per downloadable file of it (each quantization, from Hugging Face,
# plus the file Ollama gives for its name). The prediction model needs both: a model's file
# size sets what fits in a GPU's memory and how fast it can write, and its layers and
# attention heads set how much memory the conversation (KV cache) takes on top.
#
# Every number is read from the source, not estimated. Which models are in the table is
# decided by hand in scripts/model_list.csv; this script fills in the rest. A GitHub Action
# (.github/workflows/model-table.yml) runs it whenever that list changes and once a week, so
# new quantizations and re-uploads show up without anyone typing numbers.

import csv
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_LIST = ROOT / "scripts" / "model_list.csv"
OUT_DIR = ROOT / "src" / "localtune" / "data"

HF = "https://huggingface.co"
OLLAMA_REGISTRY = "https://registry.ollama.ai/v2"

# Who made the Hugging Face GGUF copies we list file sizes from, in order of preference.
# bartowski converts most popular models within days, always with the same wide set of
# quantizations and file names; unsloth covers most of the rest.
GGUF_AUTHORS = ["bartowski", "unsloth"]

# The quantizations kept: the common choices from tiny to near-original quality. MXFP4 is
# the format gpt-oss is released in.
QUANTS = ["Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "MXFP4"]

# Files skipped: image readers for vision models (mmproj), and unsloth's "dynamic" (UD)
# versions, which share quantization names with the standard files but are a different size.
SKIP = re.compile(r"mmproj|(?:^|[/_-])UD-")
QUANT_IN_NAME = re.compile(r"[-_.](IQ\d_[A-Z]+|Q\d_K(?:_[A-Z]+)?|Q\d_\d|MXFP4)(?=[-_.])")


# Fetching. Gated models (Llama, Gemma) only share their config.json after signing in, so
# model_list.csv points those at an exact public copy instead; an HF_TOKEN, if set, is used
# for everything else.


def get(url: str, accept: str = "application/json"):
    headers = {"Accept": accept, "User-Agent": "localtune-model-table"}
    if url.startswith(HF) and os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
        return json.load(r)


# Model facts. The parameter count is the exact total Hugging Face counts from the model's
# own weight files. Layers, attention heads and context length come from its config.json
# (inside "text_config" for models that also read images). Mixture-of-experts models only
# use part of their parameters per word; that number isn't in config.json, so it comes from
# the model card by hand (active_params in model_list.csv).


def model_facts(entry: dict) -> dict:
    info = get(f"{HF}/api/models/{entry['hf_repo']}")
    config_repo = entry["config_repo"] or entry["hf_repo"]
    config = get(f"{HF}/{config_repo}/resolve/main/config.json")
    config = config.get("text_config", config)

    heads = config.get("num_attention_heads")
    head_dim = config.get("head_dim") or (config["hidden_size"] // heads if heads else None)
    experts = (
        config.get("num_local_experts")
        or config.get("num_experts")
        or config.get("n_routed_experts")
        or ""
    )
    return {
        "name": entry["name"],
        "hf_repo": entry["hf_repo"],
        "params": info["safetensors"]["total"],
        "active_params": entry["active_params"] or info["safetensors"]["total"],
        "layers": config["num_hidden_layers"],
        "kv_heads": config.get("num_key_value_heads") or heads,
        "head_dim": head_dim,
        "context_length": config.get("max_position_embeddings", ""),
        "experts": experts,
        "experts_used": config.get("num_experts_per_tok", "") if experts else "",
        "config_source": f"huggingface.co/{config_repo}",
    }


# File sizes on Hugging Face: find the GGUF copy made from the model's own repo (Hugging Face
# tags every copy with the repo it came from), then read the size of each file in its file
# list. Big models are split into several files per quantization; those are added up.


def find_gguf_repo(hf_repo: str) -> str | None:
    for author in GGUF_AUTHORS:
        query = urllib.parse.urlencode(
            [("filter", f"base_model:quantized:{hf_repo}"), ("filter", "gguf"), ("author", author)]
        )
        repos = [m["id"] for m in get(f"{HF}/api/models?{query}")]
        if repos:
            return min(repos, key=len)
    return None


def hf_files(model: str, gguf_repo: str) -> list[dict]:
    tree = get(f"{HF}/api/models/{gguf_repo}/tree/main?recursive=true")
    sizes: dict[str, int] = {}
    for item in tree:
        name = item["path"].rsplit("/", 1)[-1]
        found = QUANT_IN_NAME.search(name)
        if item["type"] != "file" or not name.endswith(".gguf") or SKIP.search(item["path"]):
            continue
        if found and found.group(1) in QUANTS:
            sizes[found.group(1)] = sizes.get(found.group(1), 0) + item["size"]
    return [
        {
            "model": model,
            "quant": quant,
            "size_bytes": sizes[quant],
            "ollama": "",
            "source": f"huggingface.co/{gguf_repo}",
        }
        for quant in QUANTS
        if quant in sizes
    ]


# The file Ollama downloads for a name like qwen3:14b: its registry lists the model file's
# size and fingerprint (the sha256 name it is saved under, see src/localtune/ollama_models.py),
# and a small config file next to it says which quantization it is.


def ollama_file(model: str, ollama_name: str) -> dict:
    name, _, tag = ollama_name.partition(":")
    path = name if "/" in name else f"library/{name}"
    manifest = get(
        f"{OLLAMA_REGISTRY}/{path}/manifests/{tag or 'latest'}",
        accept="application/vnd.docker.distribution.manifest.v2+json",
    )
    layer = next(
        layer
        for layer in manifest["layers"]
        if layer["mediaType"] == "application/vnd.ollama.image.model"
    )
    config = get(f"{OLLAMA_REGISTRY}/{path}/blobs/{manifest['config']['digest']}")
    return {
        "model": model,
        "quant": config.get("file_type", "").upper(),
        "size_bytes": layer["size"],
        "ollama": ollama_name,
        "source": f"ollama.com/library/{ollama_name}",
    }


def write(name: str, rows: list[dict]) -> None:
    with (OUT_DIR / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {name}: {len(rows)} rows")


# Nothing is written if any model fails, so a bad week on Hugging Face can't shrink the
# table; the problems are printed for whoever looks at the Action's log.


def main() -> None:
    with MODEL_LIST.open(newline="") as f:
        entries = list(csv.DictReader(f))

    models, model_files, problems = [], [], []
    for entry in entries:
        try:
            models.append(model_facts(entry))
            gguf_repo = find_gguf_repo(entry["hf_repo"])
            if gguf_repo is None:
                problems.append(f"{entry['name']}: no GGUF copy by {', '.join(GGUF_AUTHORS)}")
            else:
                model_files.extend(hf_files(entry["name"], gguf_repo))
            for ollama_name in entry["ollama"].split():
                model_files.append(ollama_file(entry["name"], ollama_name))
        except (urllib.error.URLError, KeyError, StopIteration, TypeError, ValueError) as e:
            problems.append(f"{entry['name']}: {type(e).__name__}: {e}")
        print(f"{entry['name']}: done")

    if problems:
        print("\nNot written, these need fixing in scripts/model_list.csv:")
        print("\n".join(f"  {p}" for p in problems))
        sys.exit(1)
    write("models.csv", models)
    write("model_files.csv", model_files)


if __name__ == "__main__":
    main()
