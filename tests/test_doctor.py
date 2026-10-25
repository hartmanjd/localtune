# Tests for the readiness checks. Each check is given made-up findings (a GPU or none, a tool
# found or not, a little disk space or a lot), so every case is tried without the real setup.
# The server search is tested against a tiny pretend server started inside the test.

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from localtune import doctor
from localtune.doctor import Check
from localtune.gpu_monitor import GpuSample
from localtune.hardware import GpuInfo, HardwareInfo

PC = HardwareInfo(
    os="Linux 6.6",
    is_wsl=True,
    cpu_name="Test CPU",
    cpu_threads=16,
    ram_gb=32.0,
    gpus=[GpuInfo(index=0, name="Test GPU", vram_gb=16.0, driver_version="580.00")],
)


def sample(power_w):
    return GpuSample(time_s=0, power_w=power_w, vram_used_gb=1, utilization_pct=0, temperature_c=40)


def test_gpu_found():
    check = doctor.check_gpu(PC, nvidia_smi_found=True)
    assert check.status == "ok"
    assert "Test GPU (16.0 GB)" in check.detail


def test_no_nvidia_smi_fails_with_a_fix():
    check = doctor.check_gpu(PC, nvidia_smi_found=False)
    assert check.status == "fail"
    assert check.fix


def test_nvidia_smi_without_gpus_fails():
    assert doctor.check_gpu(PC.model_copy(update={"gpus": []}), True).status == "fail"


def test_power_readings():
    assert doctor.check_power(sample(35.4)).detail == "35 W now"
    assert doctor.check_power(sample(None)).status == "warn"
    assert doctor.check_power(None).status == "warn"


def test_missing_tool_offers_the_install():
    assert doctor.check_tool("llama-bench", "/usr/bin/llama-bench").status == "ok"
    check = doctor.check_tool("llama-bench", None)
    assert check.status == "warn"
    assert "install" in check.fix


def test_low_disk_space_warns():
    assert doctor.check_disk(100).status == "ok"
    assert doctor.check_disk(5).status == "warn"


def test_results_folder(tmp_path):
    assert doctor.check_results_folder(tmp_path / "localtune").status == "ok"
    blocked = tmp_path / "a-file"
    blocked.write_text("")
    assert doctor.check_results_folder(blocked / "localtune").status == "fail"


def ok(name):
    return Check(name=name, status="ok", detail="")


def test_ready_needs_llama_bench():
    assert doctor.is_ready([ok("NVIDIA GPU"), ok("llama-bench")])
    assert not doctor.is_ready([ok("NVIDIA GPU"), ok("Local AI server")])
    assert not doctor.is_ready([ok("NVIDIA GPU"), ok("Disk space")])


def test_llama_cpp_is_offered_only_with_a_working_gpu():
    missing = Check(name="llama-bench", status="warn", detail="")
    no_gpu = Check(name="NVIDIA GPU", status="fail", detail="")
    assert doctor.needs_llama_cpp([ok("NVIDIA GPU"), missing])
    assert not doctor.needs_llama_cpp([ok("NVIDIA GPU"), ok("llama-bench")])
    assert not doctor.needs_llama_cpp([no_gpu, missing])


def test_any_failure_means_not_ready():
    failed = Check(name="NVIDIA GPU", status="fail", detail="")
    assert not doctor.is_ready([failed, ok("llama-bench")])


@pytest.fixture
def pretend_server():
    class Models(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_finds_a_running_server(pretend_server):
    found = doctor.find_local_servers({"Ollama": pretend_server, "vLLM": 9})
    assert found == [f"Ollama at http://127.0.0.1:{pretend_server}"]
