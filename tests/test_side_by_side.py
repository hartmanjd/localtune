import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from localtune import side_by_side
from localtune.results import RunResult, RunSettings
from localtune.runners.base import BackendError
from localtune.runners.web_api import ApiOptions, Server
from localtune.tune import server_args

# A pretend llama-server: it takes the same flags, says it's loading for a moment, then answers
# the same pages and chat requests the real one does, writing four pieces per answer.
FAKE_SERVER = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    started = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, body):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_GET(self):
            if self.path == "/health":
                ready = time.monotonic() - started > 0.3
                self.send_json(200 if ready else 503, {})
            elif self.path == "/v1/models":
                self.send_json(200, {"data": [{"id": "model.gguf"}]})
            elif self.path == "/props":
                self.send_json(200, {"build_info": "b9999"})
            else:
                self.send_json(404, {})

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for _ in range(4):
                chunk = {"choices": [{"delta": {"content": "word"}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\\n\\n".encode())
            end = {"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 4}}
            end["timings"] = {"prompt_per_second": 2000.0, "predicted_per_second": 90.0}
            self.wfile.write(f"data: {json.dumps(end)}\\n\\ndata: [DONE]\\n\\n".encode())

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """
)


def tuned_run() -> RunResult:
    settings = RunSettings(
        engine="llama.cpp",
        model="qwen3:30b",
        quant="Q4_K_M",
        gpu_layers=49,
        cpu_moe_layers=6,
        kv_cache_k="f16",
        kv_cache_v="f16",
        flash_attention=True,
        ubatch_size=1024,
    )
    return RunResult.model_construct(settings=settings)


def test_the_server_flags_are_the_ones_tune_prints_on_the_port_asked_for():
    placement = {"gpu_layers": 49, "cpu_moe_layers": 6}
    args = server_args(tuned_run(), placement, 768, port=5123)
    assert args[args.index("--n-cpu-moe") + 1] == "6"
    assert args[args.index("-fa") + 1] == "on"
    assert args[args.index("--port") + 1] == "5123"


def test_the_tuned_settings_are_measured_in_llama_server_and_it_is_stopped(tmp_path):
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_SERVER)
    started = []

    def start(command, **options):
        process = subprocess.Popen([sys.executable, str(script), *command[1:]], **options)
        started.append(process)
        return process

    placement = {"gpu_layers": 49, "cpu_moe_layers": 6}
    options = ApiOptions(prompt_tokens=500, gen_tokens=4, repetitions=2)
    run = side_by_side.run_tuned_server(
        "llama-server", Path("model.gguf"), tuned_run(), placement, options, "qwen3:30b",
        start=start, timeout_s=20,
    )  # fmt: skip
    assert run.speed.decode_tok_s == 90.0
    assert run.settings.engine == "llama-server" and run.settings.engine_version == "b9999"
    assert run.settings.cpu_moe_layers == 6 and run.settings.ubatch_size == 1024
    assert run.settings.model == "qwen3:30b"
    assert started[0].poll() is not None


def test_a_server_that_stops_while_loading_says_why(tmp_path):
    log = tmp_path / "llama-server.log"
    log.write_text("loading model\nCUDA error: out of memory\n")
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    with pytest.raises(BackendError) as failure:
        side_by_side.wait_until_ready(process, "http://127.0.0.1:9", log, timeout_s=5)
    assert "out of memory" in str(failure.value)
    assert failure.value.status == "out_of_memory"


def test_ollama_is_asked_to_let_go_of_the_model(monkeypatch):
    asked = []
    loaded = [["qwen3:30b"], []]
    monkeypatch.setattr(side_by_side, "post_json", lambda url, body: asked.append((url, body)))
    monkeypatch.setattr(side_by_side, "loaded_in_ollama", lambda server: loaded.pop(0))
    monkeypatch.setattr(side_by_side.time, "sleep", lambda s: None)
    ollama = Server(name="Ollama", url="http://127.0.0.1:11434", models=["qwen3:30b"])
    side_by_side.unload_from_ollama(ollama, "qwen3:30b")
    assert asked == [
        ("http://127.0.0.1:11434/api/generate", {"model": "qwen3:30b", "keep_alive": 0})
    ]
    assert loaded == []
