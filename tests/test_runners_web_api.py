# Tests for the web API runner. A tiny pretend server is started inside each test: it answers
# like Ollama or llama-server (its models, its version page, and an answer streamed back in
# pieces), so no real server, model or GPU is needed.

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localtune.runners import web_api
from localtune.runners.base import BackendError
from localtune.runners.web_api import ApiOptions, Reply, Server


def start_server(pages: dict, answer: list[dict] | None = None, status: int = 200, error=b""):
    requests = []

    class Pretend(BaseHTTPRequestHandler):
        def do_GET(self):
            found = self.path in pages
            self.send_response(200 if found else 404)
            self.end_headers()
            self.wfile.write(json.dumps(pages[self.path]).encode() if found else b"")

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(status)
            self.end_headers()
            if status != 200:
                self.wfile.write(error)
                return
            for chunk in answer or []:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Pretend)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, requests


@pytest.fixture
def servers():
    started = []

    def start(*args, **kwargs):
        server, requests = start_server(*args, **kwargs)
        started.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}", requests

    yield start
    for server in started:
        server.shutdown()


MODELS = {"/v1/models": {"data": [{"id": "llama3.2:latest"}]}}
WORDS = [{"choices": [{"delta": {"content": word}}]} for word in ("Once", " upon", " a", " time")]
USAGE = {"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 4}}


def test_finds_and_names_an_ollama_server(servers):
    url, _ = servers({**MODELS, "/api/version": {"version": "0.12.6"}})
    port = int(url.rsplit(":", 1)[1])
    server = web_api.find_server(servers={"vLLM": 9, "Ollama": port})
    assert server == Server(name="Ollama", version="0.12.6", url=url, models=["llama3.2:latest"])


def test_names_llama_server_by_its_props_page(servers):
    url, _ = servers({**MODELS, "/props": {"build_info": "b6900-abc1234"}})
    server = web_api.find_server(url + "/v1/")
    assert (server.name, server.version, server.url) == ("llama-server", "b6900-abc1234", url)


def test_a_server_without_a_version_page_is_named_by_its_port(servers):
    url, _ = servers(MODELS)
    port = int(url.rsplit(":", 1)[1])
    server = web_api.identify(url, [], timeout_s=0.5)
    assert server.name == "OpenAI-style server"
    renamed = web_api.identify(url.replace(str(port), "1234"), [], timeout_s=0.1)
    assert renamed.name == "LM Studio"


def test_nothing_running_finds_nothing():
    assert web_api.find_server(servers={"Ollama": 9}) is None


def test_picks_the_model():
    assert web_api.pick_model(["llama3.2:latest"], None) == "llama3.2:latest"
    assert web_api.pick_model(["a", "llama3.2:latest"], "llama3.2") == "llama3.2:latest"
    assert web_api.pick_model(["a", "b"], None) is None
    assert web_api.pick_model(["a"], "c") is None


def test_each_prompt_is_about_the_length_asked_and_starts_differently():
    first, second = web_api.build_prompt(512), web_api.build_prompt(512)
    assert 450 < len(first.split()) < 520
    assert first.split()[1] != second.split()[1]


def test_streams_and_reads_an_answer(servers):
    url, requests = servers(MODELS, answer=[*WORDS, USAGE])
    server = Server(name="Ollama", url=url, models=["llama3.2:latest"])
    reply = web_api.send(server, "llama3.2:latest", "Hi there", 128)
    assert reply.pieces == 4
    assert (reply.prompt_tokens, reply.generated_tokens) == (500, 4)
    assert reply.sent_s <= reply.first_token_s <= reply.finished_s
    assert requests[0]["max_tokens"] == 128
    assert requests[0]["stream"] is True


def test_thinking_counts_as_writing():
    reply = {"pieces": 0}
    web_api.read_chunk({"choices": [{"delta": {"reasoning": "Hmm"}}]}, reply)
    web_api.read_chunk({"choices": [{"delta": {"content": "Hi"}}]}, reply)
    assert reply["pieces"] == 2
    assert "first_token_s" in reply


def test_server_errors_are_backend_errors(servers):
    body = b'{"error": {"message": "CUDA error: out of memory"}}'
    url, _ = servers(MODELS, status=500, error=body)
    server = Server(name="Ollama", url=url, models=[])
    with pytest.raises(BackendError) as failure:
        web_api.send(server, "llama3.2:latest", "Hi", 1)
    assert failure.value.status == "out_of_memory"
    assert str(failure.value) == "Ollama: CUDA error: out of memory"


def test_an_unreachable_server_is_a_backend_error():
    server = Server(name="Ollama", url="http://127.0.0.1:9", models=[])
    with pytest.raises(BackendError, match="couldn't reach Ollama"):
        web_api.send(server, "m", "Hi", 1)


def test_speeds_from_the_timing_of_a_reply():
    # 500 tokens read in 0.25 s, then 101 tokens streamed, the last 100 in 2 s.
    reply = Reply(
        sent_s=10.0, first_token_s=10.25, finished_s=12.25, prompt_tokens=500, generated_tokens=101
    )
    speed = web_api.reply_speed(reply, 512)
    assert speed.prefill_tok_s == pytest.approx(2000)
    assert speed.decode_tok_s == pytest.approx(50)
    assert speed.ttft_ms == pytest.approx(250)
    assert speed.busy_s == pytest.approx(2.25)


def test_llama_servers_own_timings_are_used():
    timings = {"prompt_per_second": 7000.0, "predicted_per_second": 120.0}
    reply = Reply(sent_s=0, first_token_s=1, finished_s=2, pieces=10, timings=timings)
    speed = web_api.reply_speed(reply, 512)
    assert (speed.prefill_tok_s, speed.decode_tok_s) == (7000.0, 120.0)
    assert (speed.prompt_tokens, speed.generated_tokens) == (512, 10)


def test_a_prompt_the_server_cut_short_is_a_backend_error():
    reply = Reply(sent_s=0, first_token_s=1, finished_s=2, prompt_tokens=4096, pieces=10)
    with pytest.raises(BackendError, match="only read 4,096 of the 16,000-token prompt"):
        web_api.reply_speed(reply, 16000)


def test_an_empty_answer_is_a_backend_error():
    with pytest.raises(BackendError, match="without writing"):
        web_api.reply_speed(Reply(sent_s=0, finished_s=1), 512)


def test_a_run_warms_up_then_measures_each_repetition():
    server = Server(name="Ollama", version="0.12.6", url="http://x", models=["m"])
    asked = []

    def request(server, model, prompt, max_tokens):
        asked.append(max_tokens)
        return Reply(
            sent_s=0, first_token_s=0.5, finished_s=2.5, prompt_tokens=500, generated_tokens=101
        )

    run = web_api.run_web_api(server, "m", ApiOptions(repetitions=3), request=request)
    assert asked == [1, 128, 128, 128]
    assert run.speed.prompt_tokens == 1500
    assert run.speed.generated_tokens == 303
    assert run.speed.decode_tok_s == pytest.approx(50)
    assert run.speed.busy_s == pytest.approx(7.5)
    assert (run.settings.engine, run.settings.engine_version) == ("Ollama", "0.12.6")
    assert run.settings.context_size == 512 + 128
