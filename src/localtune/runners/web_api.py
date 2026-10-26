"""Benchmark a model through a local AI server's OpenAI-style web API."""

# Ollama, llama-server, LM Studio and vLLM all keep a model loaded and answer requests in the
# same OpenAI-style way, so one runner covers them all. It finds a running server on this
# computer (the same ports doctor.py checks), asks it to write answers to a prompt of a set
# length, and times them as they stream back. base.py watches the GPU and saves the run around
# it, in the same shape as the llama.cpp runner's runs. A server keeps the settings it was
# started with (GPU layers, KV cache, ...), so here only the prompt and answer lengths are set.

import json
import re
import time
import urllib.error
import urllib.request
from uuid import uuid4

from pydantic import BaseModel

from localtune.doctor import KNOWN_SERVERS
from localtune.results import RunSettings, Speed
from localtune.runners.base import BackendError, EngineRun

# What a web API run is asked to do.


class ApiOptions(BaseModel):
    prompt_tokens: int = 512
    gen_tokens: int = 128
    repetitions: int = 3


# Finding the server. Anything that answers "which models do you have?" at /v1/models can be
# benchmarked. Which server it is gets saved with every run, since the same model can run at
# different speeds in different servers: each one has its own page that gives its name and
# version away, and a server that has none is named after the port it usually uses.


class Server(BaseModel):
    name: str
    version: str | None = None
    url: str
    models: list[str]


IDENTITIES = (
    ("Ollama", "/api/version", "version"),
    ("llama-server", "/props", "build_info"),
    ("vLLM", "/version", "version"),
)


def get_json(url: str, timeout_s: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read())


def identify(url: str, models: list[str], timeout_s: float) -> Server:
    for name, path, key in IDENTITIES:
        try:
            answer = get_json(url + path, timeout_s)
        except (urllib.error.URLError, OSError, ValueError):
            continue
        if isinstance(answer, dict) and key in answer:
            return Server(name=name, version=str(answer[key]), url=url, models=models)
    port = url.rsplit(":", 1)[-1]
    by_port = {str(p): n for n, p in KNOWN_SERVERS.items()}
    return Server(name=by_port.get(port, "OpenAI-style server"), url=url, models=models)


def find_server(
    url: str | None = None,
    host: str = "127.0.0.1",
    servers: dict[str, int] = KNOWN_SERVERS,
    timeout_s: float = 0.5,
) -> Server | None:
    if url:
        candidates = [url.rstrip("/").removesuffix("/v1")]
    else:
        candidates = [f"http://{host}:{port}" for port in servers.values()]
    for candidate in candidates:
        try:
            listed = get_json(candidate + "/v1/models", timeout_s)["data"]
            models = [entry["id"] for entry in listed]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            continue
        return identify(candidate, models, timeout_s)
    return None


# Which model to benchmark: the one asked for, or the only one the server has. Ollama names its
# models like "llama3.2:latest", so "llama3.2" finds it too.


def pick_model(models: list[str], wanted: str | None) -> str | None:
    if wanted is None:
        return models[0] if len(models) == 1 else None
    for name in (wanted, f"{wanted}:latest"):
        if name in models:
            return name
    return None


# The prompt. Filler words come out at about one token each, so the prompt is close to the
# length asked for, and the server's own token count is what gets saved. Each prompt starts
# differently, so a server can't reuse the work it did for the last one and skip the reading.

FILLER = "the quick brown fox jumps over the lazy dog".split()
REQUEST = "Ignore the words above and write a long story about a lighthouse keeper."


def build_prompt(prompt_tokens: int) -> str:
    words = [FILLER[i % len(FILLER)] for i in range(max(prompt_tokens - 30, 1))]
    return f"Note {uuid4().hex[:8]}: {' '.join(words)}\n\n{REQUEST}"


# One request, timed as the answer streams back: when it was sent, when the first piece of the
# answer arrived, and when the last one did. Servers that can are asked to keep writing until
# the set length (ignore_eos) and not to reuse earlier prompts (cache_prompt); the others ignore
# those. Most servers add their token counts at the end, and llama-server also its own timings.


class Reply(BaseModel):
    sent_s: float
    first_token_s: float | None = None
    finished_s: float
    pieces: int = 0
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    timings: dict | None = None


OUT_OF_MEMORY = re.compile(r"out of memory|requires more system memory|failed to allocate", re.I)


def send(server: Server, model: str, prompt: str, max_tokens: int, timeout_s: float = 600) -> Reply:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "cache_prompt": False,
    }
    request = urllib.request.Request(
        server.url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    sent = time.monotonic()
    reply = {"sent_s": sent, "finished_s": sent, "pieces": 0}
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                read_chunk(json.loads(data), reply)
    except urllib.error.HTTPError as error:
        raise describe_failure(server, error.read().decode("utf-8", errors="replace")) from error
    except (urllib.error.URLError, OSError) as error:
        raise BackendError(f"couldn't reach {server.name} at {server.url} ({error})") from error
    except ValueError as error:
        raise BackendError(f"couldn't read {server.name}'s answer ({error})") from error
    reply["finished_s"] = time.monotonic()
    return Reply(**reply)


def read_chunk(chunk: dict, reply: dict) -> None:
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        if delta.get("content") or delta.get("reasoning_content"):
            reply.setdefault("first_token_s", time.monotonic())
            reply["pieces"] += 1
    usage = chunk.get("usage") or {}
    if usage.get("prompt_tokens"):
        reply["prompt_tokens"] = usage["prompt_tokens"]
    if usage.get("completion_tokens"):
        reply["generated_tokens"] = usage["completion_tokens"]
    if chunk.get("timings"):
        reply["timings"] = chunk["timings"]


def describe_failure(server: Server, body: str) -> BackendError:
    try:
        error = json.loads(body).get("error")
        message = error.get("message") if isinstance(error, dict) else error
    except (ValueError, AttributeError):
        message = None
    message = f"{server.name}: {message or body.strip() or 'the request failed'}"
    if OUT_OF_MEMORY.search(body):
        return BackendError(message, status="out_of_memory")
    return BackendError(message)


# Turning one reply into speeds. Time to first token is the wait before the answer started,
# which is mostly the server reading the prompt, so prefill speed is the prompt over that wait.
# Decode speed is the rest of the answer over the time it took to stream. llama-server measures
# both itself, without the time spent sending, so its own numbers are used when given.


def reply_speed(reply: Reply, asked_prompt_tokens: int) -> Speed:
    if reply.first_token_s is None:
        raise BackendError("the server answered without writing anything")
    prompt_tokens = reply.prompt_tokens or asked_prompt_tokens
    generated = reply.generated_tokens or reply.pieces
    wait_s = reply.first_token_s - reply.sent_s
    stream_s = reply.finished_s - reply.first_token_s
    timings = reply.timings or {}
    prefill = timings.get("prompt_per_second") or (prompt_tokens / wait_s if wait_s else None)
    decode = timings.get("predicted_per_second") or (
        (generated - 1) / stream_s if generated > 1 and stream_s else None
    )
    return Speed(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated,
        prefill_tok_s=prefill,
        decode_tok_s=decode,
        ttft_ms=wait_s * 1000,
        busy_s=reply.finished_s - reply.sent_s,
    )


def average(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None


# The whole run: one short request first, so a server that loads models on demand (like
# Ollama) has the model ready before the timing starts, then the measured repetitions. Only
# the measured requests count as busy time, so loading the model isn't counted in the energy.


def run_web_api(server: Server, model: str, options: ApiOptions, request=send) -> EngineRun:
    request(server, model, "Hi", 1)
    speeds = [
        reply_speed(
            request(server, model, build_prompt(options.prompt_tokens), options.gen_tokens),
            options.prompt_tokens,
        )
        for _ in range(options.repetitions)
    ]
    speed = Speed(
        prompt_tokens=sum(s.prompt_tokens for s in speeds),
        generated_tokens=sum(s.generated_tokens for s in speeds),
        prefill_tok_s=average([s.prefill_tok_s for s in speeds]),
        decode_tok_s=average([s.decode_tok_s for s in speeds]),
        ttft_ms=average([s.ttft_ms for s in speeds]),
        busy_s=sum(s.busy_s for s in speeds),
    )
    return EngineRun(settings=requested_settings(server, model, options), speed=speed)


def requested_settings(server: Server, model: str, options: ApiOptions) -> RunSettings:
    return RunSettings(
        engine=server.name,
        engine_version=server.version,
        model=model,
        context_size=options.prompt_tokens + options.gen_tokens,
    )
