# Tests for downloading a model with Ollama. Ollama and its registry are pretend: their answers
# are written out here in the shape the real ones send, so no network or Ollama is needed.

import io
import json
import urllib.error

import pytest

from localtune import ollama_pull
from localtune.ollama_pull import PullError, UnknownModel, download_size, ollama_url, pull


class Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def answer_with(monkeypatch, body: bytes, seen: list | None = None):
    def urlopen(request, timeout=None):
        if seen is not None:
            seen.append(request)
        return Answer(body)

    monkeypatch.setattr(ollama_pull.urllib.request, "urlopen", urlopen)


def fail_with(monkeypatch, code: int, body: bytes = b""):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError("url", code, "error", {}, io.BytesIO(body))

    monkeypatch.setattr(ollama_pull.urllib.request, "urlopen", urlopen)


@pytest.mark.parametrize(
    ("setting", "url"),
    [
        ("", "http://127.0.0.1:11434"),
        ("0.0.0.0", "http://127.0.0.1:11434"),
        ("0.0.0.0:11500", "http://127.0.0.1:11500"),
        ("https://ollama.example.com:443/", "https://ollama.example.com:443"),
    ],
)
def test_finds_where_ollama_listens(monkeypatch, setting, url):
    monkeypatch.setenv("OLLAMA_HOST", setting)
    assert ollama_url() == url


def test_download_size_adds_up_the_models_files(monkeypatch):
    manifest = {"config": {"size": 500}, "layers": [{"size": 9_000}, {"size": 1_500}]}
    seen = []
    answer_with(monkeypatch, json.dumps(manifest).encode(), seen)
    assert download_size("qwen3:14b") == 11_000
    assert seen[0].full_url == "https://registry.ollama.ai/v2/library/qwen3/manifests/14b"


def test_download_size_of_a_name_the_registry_doesnt_know(monkeypatch):
    fail_with(monkeypatch, 404)
    with pytest.raises(UnknownModel):
        download_size("qwen9:14b")


def test_download_size_is_unknown_without_the_registry(monkeypatch):
    fail_with(monkeypatch, 503)
    assert download_size("qwen3:14b") is None


def test_pull_reports_bytes_across_all_files(monkeypatch):
    lines = [
        {"status": "pulling manifest"},
        {"status": "pulling a1", "digest": "sha256:a1", "total": 1000, "completed": 400},
        {"status": "pulling b2", "digest": "sha256:b2", "total": 100, "completed": 100},
        {"status": "pulling a1", "digest": "sha256:a1", "total": 1000, "completed": 1000},
        {"status": "verifying sha256 digest"},
        {"status": "success"},
    ]
    answer_with(monkeypatch, "\n".join(json.dumps(line) for line in lines).encode())
    seen = []
    pull("qwen3:14b", "http://127.0.0.1:11434", lambda *update: seen.append(update))
    assert seen[1] == ("Downloading", 400, 1000)
    assert seen[3] == ("Downloading", 1100, 1100)
    assert seen[-1] == ("verifying sha256 digest", 1100, 1100)


def test_pull_passes_on_ollamas_error(monkeypatch):
    answer_with(
        monkeypatch, json.dumps({"error": "pull model manifest: file does not exist"}).encode()
    )
    with pytest.raises(PullError, match="file does not exist"):
        pull("qwen9:14b", "http://127.0.0.1:11434", lambda *update: None)


def test_pull_that_stops_early_is_an_error(monkeypatch):
    answer_with(monkeypatch, json.dumps({"status": "pulling manifest"}).encode())
    with pytest.raises(PullError, match="before the download finished"):
        pull("qwen3:14b", "http://127.0.0.1:11434", lambda *update: None)
