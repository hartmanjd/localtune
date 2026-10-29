"""Download a model with Ollama, so a missing model doesn't stop a benchmark."""

# When someone names an Ollama model they haven't downloaded yet ("localtune tune qwen3:14b"),
# localtune offers to download it instead of sending them off to run "ollama pull" themselves
# (cli.py asks). The download is Ollama's own: localtune asks the Ollama program running on this
# computer to fetch the model, so it lands in Ollama's folder where ollama_models.py finds it,
# exactly as if "ollama pull" had been typed. Before asking, the model's size is looked up on
# the registry Ollama downloads from, which also catches a misspelled name.

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from localtune.ollama_models import full_name

DEFAULT_URL = "http://127.0.0.1:11434"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"


class PullError(Exception):
    pass


class UnknownModel(PullError):
    pass


# Where Ollama listens: its OLLAMA_HOST setting, or its usual port. "0.0.0.0" there means
# "every address", which is reached as this computer's own address.


def ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
    if not host:
        return DEFAULT_URL
    if "://" not in host:
        host = "http://" + host
    scheme, _, address = host.partition("://")
    address = address.replace("0.0.0.0", "127.0.0.1")
    if ":" not in address.rsplit("]", 1)[-1]:
        address += ":11434"
    return f"{scheme}://{address}"


# The Ollama program: on the PATH, or where Ollama's Windows installer puts it.


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    if platform.system() == "Windows":
        installed = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if installed.is_file():
            return str(installed)
    return None


def is_running(url: str, timeout_s: float = 1) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/version", timeout=timeout_s):
            return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


# Ollama installed but not running (its app closed, or a Linux computer without its service):
# it is started in the background, the same as "ollama serve", and keeps running afterwards the
# way the Ollama app would. Returns whether it answered within a few seconds.


def start_ollama(program: str, url: str, wait_s: float = 15) -> bool:
    options: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    try:
        subprocess.Popen([program, "serve"], **options)
    except OSError:
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if is_running(url):
            return True
        time.sleep(0.5)
    return False


# How much a download is, in bytes, from the model's list of files on the registry. None when
# the registry can't be reached; a name the registry doesn't know is an UnknownModel.


def download_size(name: str, timeout_s: float = 10) -> int | None:
    parts = full_name(name)
    if parts is None:
        raise UnknownModel(f"{name} isn't a model name Ollama can download.")
    registry, *path, tag = parts
    url = f"https://{registry}/v2/{'/'.join(path)}/manifests/{tag}"
    request = urllib.request.Request(url, headers={"Accept": MANIFEST_TYPE})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            manifest = json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise UnknownModel(f"Ollama has no model called {name}.") from error
        return None
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        files = [manifest["config"], *manifest["layers"]]
        return sum(int(entry["size"]) for entry in files)
    except (KeyError, TypeError, ValueError):
        return None


# The download itself. Ollama reports progress per file as it goes; the bytes done across all of
# them are passed on, along with what it's doing once the bytes are in (checking the download).


def pull(name: str, url: str, progress: Callable[[str, int, int], None]) -> None:
    body = json.dumps({"model": name, "stream": True}).encode()
    request = urllib.request.Request(
        url + "/api/pull", data=body, headers={"Content-Type": "application/json"}
    )
    files: dict[str, tuple[int, int]] = {}
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for line in response:
                if not line.strip():
                    continue
                update = json.loads(line)
                if "error" in update:
                    raise PullError(str(update["error"]))
                status = str(update.get("status", ""))
                if status == "success":
                    return
                if "digest" in update and update.get("total"):
                    files[update["digest"]] = (
                        int(update.get("completed", 0)),
                        int(update["total"]),
                    )
                    status = "Downloading"
                done = sum(completed for completed, _ in files.values())
                total = sum(size for _, size in files.values())
                progress(status, done, total)
    except urllib.error.HTTPError as error:
        raise PullError(describe_http_error(error)) from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise PullError(f"the connection to Ollama broke off ({error})") from error
    raise PullError("Ollama stopped before the download finished")


def describe_http_error(error: urllib.error.HTTPError) -> str:
    try:
        return str(json.loads(error.read())["error"])
    except (OSError, ValueError, KeyError, TypeError):
        return f"Ollama answered with error {error.code}"
