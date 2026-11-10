"""Check that this computer is ready to benchmark, and say what to fix if it isn't."""

# A benchmark needs several things outside localtune to be in place: an NVIDIA GPU that
# nvidia-smi can see (hardware.py), power readings for energy (gpu_monitor.py), an
# engine to run the model (llama.cpp's tools, or a local AI server like Ollama or llama-server),
# disk space for model files, and a folder to save results in (results.py). Each check here
# looks at one of those and comes back as ok, a warning (benchmarks work, with something
# missing) or a failure (benchmarks can't run), with what to do about it.

import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from localtune.gpu_monitor import GpuSample, read_sample
from localtune.hardware import HardwareInfo, detect_hardware, find_nvidia_smi
from localtune.install import find_tool
from localtune.results import home_dir

# Model files are often 5 to 20 GB each, so less free space than this gets a warning.
MIN_FREE_DISK_GB = 20


class Check(BaseModel):
    name: str
    status: Literal["ok", "warn", "fail"]
    detail: str
    fix: str | None = None


# One check per requirement. Each takes what was found on this computer, so the tests can try
# every case without the real hardware.


def check_gpu(hw: HardwareInfo, nvidia_smi_found: bool) -> Check:
    if not nvidia_smi_found:
        return Check(
            name="NVIDIA GPU",
            status="fail",
            detail="nvidia-smi not found",
            fix="Install NVIDIA's latest driver (with WSL, on Windows), then open a new terminal.",
        )
    if not hw.gpus:
        return Check(
            name="NVIDIA GPU",
            status="fail",
            detail="nvidia-smi found, but it lists no GPUs",
            fix="Restart the computer; if that doesn't help, reinstall NVIDIA's driver.",
        )
    names = ", ".join(f"{gpu.name} ({gpu.vram_gb} GB)" for gpu in hw.gpus)
    return Check(name="NVIDIA GPU", status="ok", detail=names)


def check_power(sample: GpuSample | None) -> Check:
    if sample is None:
        return Check(name="Power readings", status="warn", detail="no GPU readings")
    if sample.power_w is not None:
        return Check(name="Power readings", status="ok", detail=f"{sample.power_w:.0f} W now")
    return Check(
        name="Power readings",
        status="warn",
        detail="the GPU doesn't report power, so energy will be empty",
    )


def check_tool(name: str, path: str | None) -> Check:
    if path:
        return Check(name=name, status="ok", detail=path)
    return Check(
        name=name,
        status="warn",
        detail="not found",
        fix=(
            "localtune can install llama.cpp for you, or use one you already have: answer the "
            "questions below, use --install, or set LOCALTUNE_LLAMA_DIR to its folder."
        ),
    )


def check_servers(servers: list[str]) -> Check:
    if servers:
        return Check(name="Local AI server", status="ok", detail=", ".join(servers))
    return Check(
        name="Local AI server",
        status="warn",
        detail="none on the usual ports",
        fix=(
            "Only needed for 'localtune run -b api': start Ollama, llama-server or LM Studio. "
            "A server on another port or computer works too: give its address with --url."
        ),
    )


def check_disk(free_gb: float) -> Check:
    detail = f"{free_gb:.0f} GB free"
    if free_gb >= MIN_FREE_DISK_GB:
        return Check(name="Disk space", status="ok", detail=detail)
    return Check(
        name="Disk space",
        status="warn",
        detail=detail,
        fix=f"Model files take 5 to 20 GB each; free up at least {MIN_FREE_DISK_GB} GB.",
    )


def check_results_folder(folder: Path) -> Check:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
    except OSError as error:
        return Check(
            name="Results folder",
            status="fail",
            detail=f"{folder} can't be written to ({error.strerror})",
            fix="Set LOCALTUNE_HOME to a folder you can write to.",
        )
    return Check(name="Results folder", status="ok", detail=str(folder))


# Finding local AI servers. Each common server has its usual port on this computer, and they all
# answer the same OpenAI-style question ("which models do you have?") at /v1/models, so
# anything that answers there is a server localtune can benchmark.

KNOWN_SERVERS = {
    "Ollama": 11434,
    "llama-server": 8080,
    "LM Studio": 1234,
    "vLLM": 8000,
}


def find_local_servers(
    servers: dict[str, int] = KNOWN_SERVERS, host: str = "127.0.0.1", timeout_s: float = 0.5
) -> list[str]:
    found = []
    for name, port in servers.items():
        url = f"http://{host}:{port}"
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=timeout_s) as response:
                if response.status == 200:
                    found.append(f"{name} at {url}")
        except (urllib.error.URLError, OSError, ValueError):
            continue
    return found


# Ready means nothing failed and there's an engine "localtune run" can use: llama-bench, or a
# running local AI server (for "-b api"). When llama.cpp is the missing piece on a computer with
# a working GPU, install.py can fill it in.

ENGINE_CHECKS = ("llama-bench", "Local AI server")


def needs_llama_cpp(checks: list[Check]) -> bool:
    status = {check.name: check.status for check in checks}
    return status.get("NVIDIA GPU") == "ok" and status.get("llama-bench") != "ok"


def is_ready(checks: list[Check]) -> bool:
    if any(check.status == "fail" for check in checks):
        return False
    return any(check.name in ENGINE_CHECKS and check.status == "ok" for check in checks)


# Running every check on this computer, in the order they're shown.


def free_disk_gb(folder: Path) -> float:
    while not folder.exists():
        folder = folder.parent
    return shutil.disk_usage(folder).free / 1024**3


def run_checks() -> list[Check]:
    nvidia_smi = find_nvidia_smi()
    folder = home_dir()
    return [
        check_gpu(detect_hardware(), nvidia_smi is not None),
        check_power(read_sample(nvidia_smi) if nvidia_smi else None),
        check_tool("llama-bench", find_tool("llama-bench")),
        check_tool("llama-server", find_tool("llama-server")),
        check_servers(find_local_servers()),
        check_disk(free_disk_gb(folder)),
        check_results_folder(folder),
    ]
