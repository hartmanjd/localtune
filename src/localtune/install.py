"""Download and install llama.cpp, so benchmarks can run without setting it up by hand."""

# llama.cpp is the engine localtune benchmarks with, through its llama-bench and llama-server
# programs. Instead of asking people to find, download and unpack the right version, localtune
# installs one itself into its own folder (next to the results from results.py), and
# doctor.py offers to do it when llama.cpp is missing. Nothing outside that folder is changed,
# and localtune finds the programs there without anyone editing their PATH.

import hashlib
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from localtune.hardware import find_nvidia_smi
from localtune.results import home_dir

# One fixed llama.cpp build, so every localtune run of the same version measures the same
# engine; moving to a newer build is a deliberate change here. llama.cpp publishes it ready-made
# for Linux (.tar.gz) and Windows (.zip), each for two CUDA versions. Each one needs two
# downloads: llama.cpp itself, and NVIDIA's CUDA libraries it runs on the GPU with. The
# fingerprints (SHA-256) make sure each download arrived complete and unchanged.

LLAMA_CPP_BUILD = "b11669"
RELEASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_BUILD}"

DOWNLOADS = {
    "linux-cuda-13.4": {
        f"llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-13.4-x64.tar.gz": (
            "80878473d27278dbe9c38fb7e704f292a51e889d3c28a1e863137cdee1e2ba4e"
        ),
        f"cudart-llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-13.4-x64.tar.gz": (
            "1108bd9300a281e8eabb18d3fa028d4240ecd983f9f90dbbbc3297671016d697"
        ),
    },
    "linux-cuda-12.8": {
        f"llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz": (
            "4703ed1a3cd6e8504affb6cb9866f501e96d634d68a002dc9a8251c9e2819b49"
        ),
        f"cudart-llama-{LLAMA_CPP_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz": (
            "ab3eda89a82096bcba00b6448218c3c11eec2016b0f38d64884dc29736549f26"
        ),
    },
    "windows-cuda-13.4": {
        f"llama-{LLAMA_CPP_BUILD}-bin-win-cuda-13.4-x64.zip": (
            "3b92d5c064f5032e603e531d3ec8961f3bd6613e5dfad047399a92dec8e95dbc"
        ),
        "cudart-llama-bin-win-cuda-13.4-x64.zip": (
            "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"
        ),
    },
    "windows-cuda-12.4": {
        f"llama-{LLAMA_CPP_BUILD}-bin-win-cuda-12.4-x64.zip": (
            "c2da1f73ed176ae093ac5aceb978a7b167a16113f6d74c6789eeaee2135cb8dc"
        ),
        "cudart-llama-bin-win-cuda-12.4-x64.zip": (
            "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"
        ),
    },
}

DOWNLOAD_MB = {
    "linux-cuda-13.4": 600,
    "linux-cuda-12.8": 770,
    "windows-cuda-13.4": 550,
    "windows-cuda-12.4": 630,
}


class InstallError(Exception):
    pass


# Picking the build. Ready-made builds exist for Windows and Linux (which includes WSL) on
# x86-64 PCs. NVIDIA's driver says which CUDA version it supports (the "CUDA Version" that
# nvidia-smi prints, called "CUDA UMD Version" by newer drivers), and the build must not need a
# newer one. Older drivers are asked to update instead.


def parse_cuda_version(nvidia_smi_output: str) -> tuple[int, int] | None:
    match = re.search(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)", nvidia_smi_output)
    return (int(match[1]), int(match[2])) if match else None


OLDEST_CUDA = {"linux": (12, 8), "windows": (12, 4)}


def pick_build(cuda_version: tuple[int, int] | None, system: str = "linux") -> str | None:
    if cuda_version is None:
        return None
    if cuda_version[0] >= 13:
        return f"{system}-cuda-13.4"
    oldest = OLDEST_CUDA[system]
    if cuda_version >= oldest:
        return f"{system}-cuda-{oldest[0]}.{oldest[1]}"
    return None


def choose_build() -> str:
    system = platform.system().lower()
    if system not in OLDEST_CUDA or platform.machine() not in ("x86_64", "AMD64"):
        raise InstallError(
            "Automatic install only works on Windows, Linux and WSL (x86-64 PCs) for now."
        )
    cuda_version = driver_cuda_version()
    if cuda_version is None:
        raise InstallError("Couldn't read the CUDA version from NVIDIA's driver.")
    build = pick_build(cuda_version, system)
    if build is None:
        oldest = OLDEST_CUDA[system]
        raise InstallError(
            f"Your NVIDIA driver supports CUDA {cuda_version[0]}.{cuda_version[1]}, and llama.cpp "
            f"needs {oldest[0]}.{oldest[1]} or newer. Update the driver, then try again."
        )
    return build


def driver_cuda_version() -> tuple[int, int] | None:
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [str(nvidia_smi)], capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_cuda_version(result.stdout)


# Where the installed programs live, and finding a program either there or anywhere on the
# PATH (for people who installed llama.cpp themselves, which always takes priority). On
# Windows the programs end in .exe.


def install_dir() -> Path:
    return home_dir() / "llama.cpp" / LLAMA_CPP_BUILD


def find_tool(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    if platform.system() == "Windows":
        name += ".exe"
    installed = install_dir() / name
    return str(installed) if installed.is_file() else None


# Downloading and unpacking. Each file is checked against its fingerprint before anything is
# unpacked, and unpacking refuses any file that would land outside the install folder.
# Everything is unpacked into a temporary folder first and only moved into place once it all
# worked, so a failed install never leaves a half-finished one behind.


def download(url: str, dest: Path, sha256: str, progress: Callable[[int, int], None]) -> None:
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as f:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1024 * 1024):
            f.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            progress(done, total)
    if digest.hexdigest() != sha256:
        raise InstallError(f"{dest.name} didn't download correctly (fingerprint mismatch)")


def unpack(archive: Path, dest: Path) -> None:
    if archive.suffix == ".zip":
        unpack_zip(archive, dest)
    else:
        unpack_tar(archive, dest)


def unpack_zip(archive: Path, dest: Path) -> None:
    # The Windows archives hold their files directly, with no top folder.
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/")).parts
            if not parts:
                continue
            if member.filename.startswith(("/", "\\")) or ".." in parts or ":" in member.filename:
                raise InstallError(f"{archive.name} contains an unsafe file: {member.filename}")
            target = dest.joinpath(*parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zip_file.open(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)


def unpack_tar(archive: Path, dest: Path) -> None:
    # The Linux archives hold one top folder; its contents go straight into dest.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts[1:]
            if not parts:
                continue
            if (
                member.name.startswith("/")
                or ".." in parts
                or not (member.isfile() or member.isdir() or member.issym())
            ):
                raise InstallError(f"{archive.name} contains an unsafe file: {member.name}")
            if member.issym() and (member.linkname.startswith("/") or ".." in member.linkname):
                raise InstallError(f"{archive.name} contains an unsafe link: {member.name}")
            target = dest.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.unlink(missing_ok=True)
                target.symlink_to(member.linkname)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def install_llama_cpp(
    build: str,
    progress: Callable[[str, int, int], None] = lambda name, done, total: None,
    base_url: str = RELEASE_URL,
) -> Path:
    dest = install_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        staging = Path(tmp) / "unpacked"
        for name, sha256 in DOWNLOADS[build].items():
            archive = Path(tmp) / name
            download(f"{base_url}/{name}", archive, sha256, lambda d, t, n=name: progress(n, d, t))
            unpack(archive, staging)
            archive.unlink()
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
    return dest
