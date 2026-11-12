"""The tuned settings and Ollama, measured the same way: both as chat servers, in turn."""

# llama-bench (the tool tune measures with) times llama.cpp on its own, with no chat around it,
# while Ollama can only be timed through its chat API (runners/web_api.py), which also turns
# the prompt into a chat, streams every word back and waits on the network. Putting a
# llama-bench number beside an Ollama number would favour the tuned side. So for the
# comparison the tuned settings run in llama-server, llama.cpp's own chat server, and both are
# sent the same requests through the same OpenAI-style API by the same code, so the only
# difference left is the engine and its settings.
#
# Only one of them holds the model at a time: Ollama keeps a model loaded for a few minutes
# after its last answer, so it is asked to let go of it first (keep_alive 0), then llama-server
# is started on a free port with the tuned settings, measured and stopped, and only then is
# Ollama measured, with whatever settings it picks for itself.

import json
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from localtune.results import RunResult, RunSettings
from localtune.runners.base import BackendError, EngineRun
from localtune.runners.web_api import ApiOptions, Server, find_server, run_web_api
from localtune.tune import server_args

START_TIMEOUT_S = 600  # a big model read from a slow drive can take minutes to load
UNLOAD_TIMEOUT_S = 30


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def post_json(url: str, body: dict, timeout_s: float = 30) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read() or b"{}")


def loaded_in_ollama(server: Server) -> list[str]:
    try:
        with urllib.request.urlopen(server.url + "/api/ps", timeout=5) as response:
            return [entry.get("name", "") for entry in json.loads(response.read())["models"]]
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return []


def unload_from_ollama(server: Server, model: str, timeout_s: float = UNLOAD_TIMEOUT_S) -> None:
    try:
        post_json(server.url + "/api/generate", {"model": model, "keep_alive": 0})
    except (urllib.error.URLError, OSError, ValueError):
        return
    deadline = time.monotonic() + timeout_s
    while model in loaded_in_ollama(server) and time.monotonic() < deadline:
        time.sleep(0.5)


# llama-server answers /health with 503 while it loads the model and 200 once it's ready. If it
# stops instead (out of memory, a file it can't read), the end of what it printed says why.


def wait_until_ready(process: subprocess.Popen, url: str, log: Path, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            said = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
            message = " ".join(said) or f"it stopped with code {process.returncode}"
            status = "out_of_memory" if "out of memory" in message.lower() else "crashed"
            raise BackendError(f"llama-server didn't start: {message}", status=status)
        try:
            with urllib.request.urlopen(url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise BackendError(f"llama-server wasn't ready after {timeout_s:.0f} s")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# The tuned run, through llama-server: started with the same flags tune prints for chatting
# (tune.py), measured with the web API runner, and stopped again. The run is saved with the
# tuned settings it was started with, so it can be told apart from llama-server runs at its
# defaults.


def run_tuned_server(
    llama_server: str,
    model_file: Path,
    best: RunResult,
    placement: dict,
    options: ApiOptions,
    model_name: str,
    threads: int | None = None,
    gpu: int = 0,
    start=subprocess.Popen,
    timeout_s: float = START_TIMEOUT_S,
) -> EngineRun:
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    tokens = options.prompt_tokens + options.gen_tokens
    flags = server_args(best, placement, tokens, threads, gpu, port=port)
    # On Windows a stopped server can hold its log a moment longer; a log that can't be deleted
    # yet must not fail the run.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        log = Path(folder) / "llama-server.log"
        with log.open("w", encoding="utf-8") as output:
            process = start(
                [llama_server, "-m", str(model_file), *flags],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_until_ready(process, url, log, timeout_s)
                server = find_server(url, timeout_s=5)
                if server is None or not server.models:
                    raise BackendError("llama-server started but doesn't list the model")
                server.name = "llama-server"
                measured = run_web_api(server, server.models[0], options)
            finally:
                stop(process)
    tuned = best.settings
    measured.settings = RunSettings(
        engine="llama-server",
        engine_version=server.version,
        model=model_name,
        model_file=str(model_file),
        quant=tuned.quant,
        context_size=int(flags[flags.index("-c") + 1]),
        gpu_layers=placement.get("gpu_layers"),
        cpu_moe_layers=placement.get("cpu_moe_layers"),
        kv_cache_k=tuned.kv_cache_k,
        kv_cache_v=tuned.kv_cache_k,
        flash_attention=tuned.flash_attention,
        threads=threads,
        ubatch_size=tuned.ubatch_size,
    )
    return measured
