# Tests for downloading an Ollama model. Ollama and its registry are pretend: their answers
# are written out here in the shape the real ones send, so no network or Ollama is needed.

import hashlib
import io
import json
import urllib.error

import pytest

from localtune import ollama_pull
from localtune.ollama_models import downloaded_models, find_ollama_model
from localtune.ollama_pull import (
    PullError,
    UnknownModel,
    download_size,
    fetch,
    ollama_url,
    pull,
)


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


# The direct download from Ollama's library, into a pretend Ollama folder. The library is
# pretend too: it knows one model made of two small files, and can answer part of a file
# when asked to carry on (a "Range"), the way the real one does.

MODEL_BYTES = b"GGUF" + b"x" * 5000
CONFIG_BYTES = b'{"model_format": "gguf"}'


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def library(monkeypatch, tmp_path, files=None, broken_after=None, ranges=True, config_size=None):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    files = files or {digest_of(MODEL_BYTES): MODEL_BYTES, digest_of(CONFIG_BYTES): CONFIG_BYTES}
    manifest = json.dumps(
        {
            "config": {
                "digest": digest_of(CONFIG_BYTES),
                "size": config_size or len(CONFIG_BYTES),
            },
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": digest_of(MODEL_BYTES),
                    "size": len(MODEL_BYTES),
                }
            ],
        }
    ).encode()
    asked = []

    def urlopen(request, timeout=None):
        url = request.full_url
        asked.append((url, request.get_header("Range")))
        assert request.get_header("User-agent").startswith("localtune/")
        if url == "https://registry.ollama.ai/v2/library/qwen3/manifests/14b":
            return Answer(manifest)
        if "/blobs/" not in url:
            raise urllib.error.HTTPError(url, 404, "not found", {}, io.BytesIO(b""))
        data = files[url.rsplit("/", 1)[1]]
        start = 0
        wanted = request.get_header("Range")
        if wanted and ranges:
            start = int(wanted.removeprefix("bytes=").rstrip("-"))
        body = data[start:]
        if broken_after is not None and len(body) > broken_after:
            answer = BrokenAnswer(body[:broken_after])
        else:
            answer = Answer(body)
        answer.status = 206 if start else 200
        return answer

    monkeypatch.setattr(ollama_pull.urllib.request, "urlopen", urlopen)
    return asked


class BrokenAnswer(Answer):
    def read(self, size=-1):
        chunk = super().read(size)
        if not chunk:
            raise ConnectionResetError("connection reset")
        return chunk


def test_fetch_saves_the_model_the_way_ollama_does(monkeypatch, tmp_path):
    library(monkeypatch, tmp_path)
    seen = []
    fetch("qwen3:14b", lambda status, done, total: seen.append((done, total)))
    assert find_ollama_model("qwen3:14b").read_bytes() == MODEL_BYTES
    assert downloaded_models() == ["qwen3:14b"]
    total = len(MODEL_BYTES) + len(CONFIG_BYTES)
    assert seen[-1] == (total, total)
    assert not list(tmp_path.rglob("*partial*"))


def test_fetch_skips_files_already_there(monkeypatch, tmp_path):
    asked = library(monkeypatch, tmp_path)
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / digest_of(MODEL_BYTES).replace(":", "-")).write_bytes(MODEL_BYTES)
    fetch("qwen3:14b", lambda *a: None)
    assert [url for url, _ in asked if "/blobs/" in url] == [
        "https://registry.ollama.ai/v2/library/qwen3/blobs/" + digest_of(CONFIG_BYTES)
    ]


def test_fetch_carries_on_after_the_connection_breaks(monkeypatch, tmp_path):
    library(monkeypatch, tmp_path, broken_after=1000)
    with pytest.raises(PullError, match="carry on"):
        fetch("qwen3:14b", lambda *a: None)
    assert find_ollama_model("qwen3:14b") is None
    asked = library(monkeypatch, tmp_path)
    fetch("qwen3:14b", lambda *a: None)
    assert (
        "https://registry.ollama.ai/v2/library/qwen3/blobs/" + digest_of(MODEL_BYTES),
        "bytes=1000-",
    ) in asked
    assert find_ollama_model("qwen3:14b").read_bytes() == MODEL_BYTES


def test_fetch_carries_on_after_the_connection_closes_early(monkeypatch, tmp_path):
    short = {digest_of(MODEL_BYTES): MODEL_BYTES[:1000], digest_of(CONFIG_BYTES): CONFIG_BYTES}
    library(monkeypatch, tmp_path, files=short)
    with pytest.raises(PullError, match="carry on"):
        fetch("qwen3:14b", lambda *a: None)
    asked = library(monkeypatch, tmp_path)
    fetch("qwen3:14b", lambda *a: None)
    assert (
        "https://registry.ollama.ai/v2/library/qwen3/blobs/" + digest_of(MODEL_BYTES),
        "bytes=1000-",
    ) in asked
    assert find_ollama_model("qwen3:14b").read_bytes() == MODEL_BYTES


def test_fetch_keeps_a_file_whose_listed_size_is_off(monkeypatch, tmp_path):
    asked = library(monkeypatch, tmp_path, config_size=len(CONFIG_BYTES) - 3)
    fetch("qwen3:14b", lambda *a: None)
    assert find_ollama_model("qwen3:14b").read_bytes() == MODEL_BYTES
    assert not list(tmp_path.rglob("*partial*"))
    assert len(asked) == 3


def test_fetch_starts_over_when_the_library_sends_the_whole_file(monkeypatch, tmp_path):
    library(monkeypatch, tmp_path, broken_after=1000)
    with pytest.raises(PullError):
        fetch("qwen3:14b", lambda *a: None)
    library(monkeypatch, tmp_path, ranges=False)
    fetch("qwen3:14b", lambda *a: None)
    assert find_ollama_model("qwen3:14b").read_bytes() == MODEL_BYTES


def test_fetch_throws_away_a_damaged_file(monkeypatch, tmp_path):
    damaged = {digest_of(MODEL_BYTES): b"GGUF" + b"y" * 5000, digest_of(CONFIG_BYTES): CONFIG_BYTES}
    library(monkeypatch, tmp_path, files=damaged)
    with pytest.raises(PullError, match="damaged"):
        fetch("qwen3:14b", lambda *a: None)
    assert find_ollama_model("qwen3:14b") is None
    assert not list(tmp_path.rglob("*partial*"))


def test_fetch_a_name_the_library_doesnt_know(monkeypatch, tmp_path):
    library(monkeypatch, tmp_path)
    with pytest.raises(UnknownModel):
        fetch("qwen9:14b", lambda *a: None)


def test_fetch_refuses_odd_file_names_in_the_list(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_MODELS", str(tmp_path))
    manifest = {"config": {"digest": "sha256:../../evil", "size": 1}, "layers": []}
    answer_with(monkeypatch, json.dumps(manifest).encode())
    with pytest.raises(PullError, match="can't read"):
        fetch("qwen3:14b", lambda *a: None)
    assert not (tmp_path / "evil").exists()
