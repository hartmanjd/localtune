"""Download an Ollama model, so a missing model doesn't stop a benchmark."""

# When someone names an Ollama model they haven't downloaded yet ("localtune tune qwen3:14b"),
# localtune offers to download it instead of sending them off to run "ollama pull" themselves
# (cli.py asks). fetch() downloads it straight from Ollama's online library (the registry), the
# same files "ollama pull" would get, so the Ollama program doesn't need to be installed. They
# are saved in Ollama's folder the way Ollama saves them, so ollama_models.py finds them, and an
# Ollama installed now or later lists the model as its own. Before asking, the model's size is
# looked up on the registry, which also catches a misspelled name.
#
# pull() is the other way, through the Ollama program running on this computer; "localtune
# samples" uses it, because it needs Ollama to answer the questions anyway.

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from localtune import __version__
from localtune.ollama_models import full_name, manifest_path, models_dir

DEFAULT_URL = "http://127.0.0.1:11434"
MANIFEST_TYPE = "application/vnd.docker.distribution.manifest.v2+json"
HEADERS = {"Accept": MANIFEST_TYPE, "User-Agent": f"localtune/{__version__}"}
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
CHUNK = 1 << 20


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
    request = urllib.request.Request(url, headers=HEADERS)
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


# The direct download. The model's list of files (its manifest) comes first; then each file,
# named by its fingerprint (sha256), is downloaded unless it's already there, which also skips
# files another model shares. A file is checked against its fingerprint before it's kept, and a
# download that broke off carries on from where it stopped the next time. The manifest is
# saved last, so a model only shows up as downloaded once all of its files are in.


def fetch(name: str, progress: Callable[[str, int, int], None]) -> None:
    parts = full_name(name)
    if parts is None:
        raise UnknownModel(f"{name} isn't a model name Ollama can download.")
    registry, *path, tag = parts
    base = f"https://{registry}/v2/{'/'.join(path)}"
    try:
        with _get(f"{base}/manifests/{tag}") as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise UnknownModel(f"Ollama has no model called {name}.") from error
        raise PullError(f"Ollama's library answered with error {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise PullError(f"Ollama's library couldn't be reached ({error})") from error
    try:
        manifest = json.loads(raw)
        files = [
            (entry["digest"], int(entry["size"]))
            for entry in [manifest["config"], *manifest["layers"]]
        ]
    except (ValueError, KeyError, TypeError) as error:
        raise PullError("Ollama's library sent a list of files localtune can't read") from error
    if not all(DIGEST.fullmatch(str(digest)) for digest, _ in files):
        raise PullError("Ollama's library sent a list of files localtune can't read")

    total = sum(size for _, size in files)
    blobs = models_dir() / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    done = 0
    progress("Downloading", done, total)
    for digest, size in files:
        target = blobs / digest.replace(":", "-")
        if not (target.is_file() and target.stat().st_size == size):
            start = done
            _fetch_file(
                f"{base}/blobs/{digest}",
                target,
                digest,
                size,
                lambda got, start=start: progress("Downloading", start + got, total),
            )
        done += size
        progress("Downloading", done, total)

    manifest_file = manifest_path(name)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    partial = manifest_file.with_name(manifest_file.name + ".localtune-partial")
    partial.write_bytes(raw)
    os.replace(partial, manifest_file)


def _get(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers={**HEADERS, **(headers or {})})
    return urllib.request.urlopen(request, timeout=60)


def _fetch_file(url: str, target: Path, digest: str, size: int, got: Callable[[int], None]):
    partial = target.with_name(target.name + "-localtune-partial")
    have = partial.stat().st_size if partial.is_file() else 0
    if have > size:
        partial.unlink()
        have = 0
    check = hashlib.sha256()
    if have:
        with partial.open("rb") as kept:
            while chunk := kept.read(CHUNK):
                check.update(chunk)
    try:
        with _get(url, {"Range": f"bytes={have}-"} if have else None) as response:
            if have and response.status != 206:
                have, check = 0, hashlib.sha256()
            with partial.open("ab" if have else "wb") as out:
                got(have)
                while chunk := response.read(CHUNK):
                    out.write(chunk)
                    check.update(chunk)
                    have += len(chunk)
                    got(have)
    except urllib.error.HTTPError as error:
        raise PullError(f"Ollama's library answered with error {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise PullError(
            f"the download broke off ({error}). Run it again to carry on where it stopped"
        ) from error
    # The fingerprint decides whether the file is right. The size in the library's list can be a
    # few bytes off (it is for some models' small settings file), so a file with the right
    # fingerprint is kept whatever its length. A connection that closes early ends the reading
    # without an error, so a file that's still short just broke off: what arrived is kept for
    # the next time.
    if f"sha256:{check.hexdigest()}" != digest:
        if have < size:
            raise PullError("the download broke off. Run it again to carry on where it stopped")
        partial.unlink(missing_ok=True)
        raise PullError("a downloaded file came out damaged. Run it again to download it anew")
    os.replace(partial, target)


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
