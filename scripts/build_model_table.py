"""Build localtune's model tables from Ollama's most popular models.

Run from the project folder:
    python scripts/build_model_table.py
"""

# localtune ships two model tables (src/localtune/data/) so it can tell, without downloading
# anything, how big a model is and how it is built: models.csv has one row per model, and
# model_files.csv one row per downloadable file of it (each quantization Ollama offers). The
# prediction model needs both: a model's file size sets what fits in a GPU's memory and how
# fast it can write, and its layers and attention heads set how much memory the conversation
# (KV cache) takes on top.
#
# Nothing is picked or typed by hand. Which models are in the table is Ollama's own ranking:
# the top models on ollama.com/library sorted by downloads ("pulls"), with each one's rank and
# pull count kept as the reason it is there. Every other number comes from Ollama's files
# themselves. A GitHub Action (.github/workflows/model-table.yml) runs this every week, so
# the table follows what people actually download.

import csv
import json
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "src" / "localtune" / "data"

LIBRARY = "https://ollama.com/library"
REGISTRY = "https://registry.ollama.ai/v2/library"
MODEL_LAYER = "application/vnd.ollama.image.model"
TOP = 100

# The quantizations kept, from smallest to the unquantized original (fp16/bf16), as Ollama
# spells them at the end of its tag names (qwen3:14b-q8_0).
QUANTS = ["q2_K", "q3_K_M", "q4_0", "q4_K_M", "q5_K_M", "q6_K", "q8_0", "mxfp4", "fp16", "bf16"]


# Ollama's servers sometimes take too long to answer one of the thousands of requests; trying
# again a couple of times gets past that.


def get(url: str, accept: str = "application/json"):
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "localtune-model-table"}
    )
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == 2:
                raise
            time.sleep(5)


def get_json(url: str, accept: str = "application/json") -> dict:
    with get(url, accept) as r:
        return json.load(r)


def get_text(url: str) -> str:
    with get(url, "text/html") as r:
        return r.read().decode()


# The ranking: ollama.com/library?sort=popular lists every model with its pull count, the
# sizes it comes in (8b, 70b) and what it can do. Embedding models are left out, since they
# turn text into numbers for search rather than write, and so are models that only run in
# Ollama's cloud.


def pulls_number(text: str) -> int:
    scale = {"K": 1e3, "M": 1e6, "B": 1e9}.get(text[-1:].upper(), 1)
    return int(float(text.rstrip("KMBkmb")) * scale)


def ranking() -> list[dict]:
    html = get_text(f"{LIBRARY}?sort=popular")
    models = []
    for block in html.split("<li ")[1:]:
        name = re.search(r'href="/library/([^"/?#:]+)"', block)
        pulls = re.search(r"<span\s*>([\d.]+[KMB]?)</span>\s*<span[^>]*>&nbsp;Pulls", block)
        if not (name and pulls):
            continue
        labels = re.findall(r'rounded-md ([^"]*)"\s*>([^<]+)<', block)
        sizes = [text.strip() for style, text in labels if "#ddf4ff" in style]
        capabilities = [text.strip() for style, text in labels if "#ddf4ff" not in style]
        if "embedding" in capabilities:
            continue
        models.append(
            {
                "family": name.group(1),
                "pulls": pulls_number(pulls.group(1)),
                "sizes": sizes or ["latest"],
            }
        )
    if not models:
        first = max(html.find('href="/library/'), 0)
        sys.exit("Couldn't read Ollama's model list:\n" + html[first - 500 : first + 2500])
    return models


def tag_list(family: str) -> list[str]:
    html = get_text(f"{LIBRARY}/{family}/tags")
    return sorted(set(re.findall(rf'href="/library/{re.escape(family)}:([^"]+)"', html)))


def manifest(family: str, tag: str) -> dict:
    return get_json(
        f"{REGISTRY}/{family}/manifests/{tag}",
        accept="application/vnd.docker.distribution.manifest.v2+json",
    )


def model_layer(m: dict) -> dict | None:
    return next((layer for layer in m["layers"] if layer["mediaType"] == MODEL_LAYER), None)


# The model's facts, read from the start of Ollama's own model file. A GGUF file begins with a
# list of facts about the model (architecture, layers, attention heads, context length,
# experts), then the name and shape of every block of weights in it. Only that beginning is
# downloaded. Adding up the weight shapes gives the exact parameter count; for mixture-of-
# experts models, the expert weights (named "_exps") count only for the experts used per word,
# which gives the exact number of active parameters.


class Stream:
    def __init__(self, response):
        self.response = response

    def take(self, n: int) -> bytes:
        data = self.response.read(n)
        if len(data) != n:
            raise ValueError("model file ended early")
        return data

    def num(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.num("Q")).decode(errors="replace")


# How each kind of value is stored (8 is text and 9 a list, handled below).
GGUF_SCALARS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def gguf_value(s: Stream, kind: int):
    if kind == 8:
        return s.string()
    if kind == 9:
        item_kind, count = s.num("I"), s.num("Q")
        return [gguf_value(s, item_kind) for _ in range(count)]
    return s.num(GGUF_SCALARS[kind])


def gguf_facts(family: str, digest: str) -> dict:
    with get(f"{REGISTRY}/{family}/blobs/{digest}", accept="*/*") as response:
        s = Stream(response)
        if s.take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        s.num("I")
        tensor_count, kv_count = s.num("Q"), s.num("Q")
        meta = {}
        for _ in range(kv_count):
            key = s.string()
            meta[key] = gguf_value(s, s.num("I"))
        params = expert_params = 0
        for _ in range(tensor_count):
            name = s.string()
            dims = [s.num("Q") for _ in range(s.num("I"))]
            s.num("I"), s.num("Q")
            count = 1
            for d in dims:
                count *= d
            params += count
            if "_exps" in name:
                expert_params += count

    arch = meta["general.architecture"]

    def fact(key):
        value = meta.get(f"{arch}.{key}")
        return max(value) if isinstance(value, list) else value

    heads, experts, used = fact("attention.head_count"), fact("expert_count"), None
    if experts:
        used = fact("expert_used_count")
        active = round(params - expert_params + expert_params * used / experts)
    else:
        active = params
    return {
        "architecture": arch,
        "params": params,
        "active_params": active,
        "layers": fact("block_count"),
        "kv_heads": fact("attention.head_count_kv") or heads,
        "head_dim": fact("attention.key_length") or fact("embedding_length") // heads,
        "context_length": fact("context_length"),
        "experts": experts or "",
        "experts_used": used or "",
    }


# One size of a model (qwen3:14b): its facts and every quantization Ollama offers of it. The
# name people type is a shortcut to one exact file, which also has a longer name that ends in
# its quantization (qwen3:14b is the same file as qwen3:14b-q4_K_M). Finding that longer name
# shows how Ollama names this size's other quantizations (qwen3:14b-q8_0, qwen3:14b-fp16).


def quant_name(text: str) -> str:
    return text.upper().replace("FP16", "F16")


def one_size(
    family: str, size: str, rank: int, pulls: int, tags: list[str], latest_digest
) -> tuple:
    name = f"{family}:{size}"
    main = manifest(family, size)
    layer = model_layer(main)
    if layer is None:
        return None, [], None
    config = get_json(f"{REGISTRY}/{family}/blobs/{main['config']['digest']}")
    model = {"name": name, "family": family, "popularity_rank": rank, "pulls": pulls}
    model["latest"] = "yes" if layer["digest"] == latest_digest else "no"
    model |= gguf_facts(family, layer["digest"])
    quant = quant_name(config.get("file_type", ""))
    files = [{"model": name, "quant": quant, "size_bytes": layer["size"]}]
    files[0] |= {"ollama": name, "source": f"ollama.com/library/{name}"}

    same_file = [
        t for t in tags if t.startswith(size + "-") and quant_name(t.rsplit("-", 1)[-1]) == quant
    ]
    stem = next(
        (
            t.rsplit("-", 1)[0]
            for t in same_file
            if model_layer(manifest(family, t))["digest"] == layer["digest"]
        ),
        None,
    )
    for other_quant in QUANTS if stem else []:
        tag = f"{stem}-{other_quant}"
        if tag not in tags or quant_name(other_quant) == quant:
            continue
        other = model_layer(manifest(family, tag))
        files.append(
            {
                "model": name,
                "quant": quant_name(other_quant),
                "size_bytes": other["size"],
                "ollama": f"{family}:{tag}",
                "source": f"ollama.com/library/{family}:{tag}",
            }
        )
    return model, files, None


def one_family(rank: int, entry: dict) -> tuple[list, list, list]:
    models, files, notes = [], [], []
    try:
        tags = tag_list(entry["family"])
        latest = model_layer(manifest(entry["family"], "latest")) if "latest" in tags else None
        for size in entry["sizes"]:
            try:
                model, size_files, note = one_size(
                    entry["family"], size, rank, entry["pulls"], tags, latest and latest["digest"]
                )
            except (OSError, KeyError, TypeError, ValueError) as e:
                model, size_files, note = None, [], f"{entry['family']}:{size}: {e}"
            if model:
                models.append(model)
                files.extend(size_files)
            if note:
                notes.append(note)
    except (OSError, KeyError, TypeError) as e:
        notes.append(f"{entry['family']}: {e}")
    print(f"#{rank} {entry['family']}: {len(models)} sizes")
    return models, files, notes


def write(name: str, rows: list[dict]) -> None:
    with (OUT_DIR / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {name}: {len(rows)} rows")


# Sizes that only run in Ollama's cloud have no file and are left out quietly. A few sizes
# failing (a file that isn't GGUF, an odd tag) is normal and they are listed in the log. If
# many fail, something is wrong with Ollama or this script, and nothing is written, so a bad
# week can't wipe out the table.


def main() -> None:
    top = ranking()[:TOP]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one_family, range(1, len(top) + 1), top))
    models = [m for r in results for m in r[0]]
    files = [f for r in results for f in r[1]]
    notes = [n for r in results for n in r[2]]

    print(f"\n{len(notes)} sizes failed:" if notes else "")
    print("\n".join(f"  {n}" for n in notes))
    tried = len(models) + len(notes)
    if not models or len(notes) > tried / 10:
        sys.exit(f"Not written: {len(notes)} of {tried} sizes failed.")
    write("models.csv", models)
    write("model_files.csv", files)


if __name__ == "__main__":
    main()
