# Tests for sharing runs. Runs are saved into a temporary localtune folder, and "sent" to a
# small web server started on this computer just for the test, so nothing leaves the machine.

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from typer.testing import CliRunner

from localtune import cli
from localtune.cost import ElectricityPrice
from localtune.gpu_monitor import GpuSummary
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.results import (
    RunResult,
    RunSettings,
    Speed,
    energy_and_cost,
    load_results,
    save_result,
)
from localtune.share import not_yet_shared, public_record

runner = CliRunner()
ZIP_PRICE = ElectricityPrice(per_kwh=0.15, kind="zip", source="zip 43210 (OH), NREL")


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    monkeypatch.delenv("LOCALTUNE_SHARE_URL", raising=False)


# A run with things in it that must not be shared: a folder path with a user name in its error
# message, and an electricity price looked up by zip code.


def private_run() -> RunResult:
    gpu = GpuSummary(
        duration_s=600,
        sample_count=1200,
        avg_power_w=300,
        peak_power_w=320,
        energy_wh=50,
        baseline_vram_gb=1.0,
        peak_vram_gb=9.0,
        added_vram_gb=8.0,
        avg_utilization_pct=95,
        peak_temperature_c=70,
    )
    speed = Speed(prompt_tokens=512, generated_tokens=128, prefill_tok_s=3000, decode_tok_s=90)
    return RunResult(
        rig_id="test-rig",
        status="crashed",
        error=r"C:\Users\justi\models\x.gguf: failed",
        settings=RunSettings(engine="llama.cpp", model="qwen3:14b", quant="Q4_K_M"),
        hardware=HardwareInfo(
            os="Windows-11",
            is_wsl=False,
            cpu_name="Test CPU",
            cpu_threads=16,
            ram_gb=32.0,
            gpus=[GpuInfo(index=0, name="Test GPU", vram_gb=16.0, driver_version="580.00")],
        ),
        speed=speed,
        gpu=gpu,
        energy=energy_and_cost(gpu, speed, ZIP_PRICE),
    )


def test_a_shared_run_leaves_out_paths_location_and_time_of_day():
    record = public_record(private_run())
    text = json.dumps(record)
    assert "justi" not in text
    assert "43210" not in text
    assert "price" not in record["energy"] and "cost" not in record["energy"]
    assert record["energy"]["kwh"] == pytest.approx(0.05)
    assert len(record["created_at"]) == len("2026-10-27")
    assert record["rig_id"] == "test-rig"
    assert record["speed"]["decode_tok_s"] == 90


class Collector(BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        Collector.received.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Collector.received = []
    httpd = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/runs"
    httpd.shutdown()


def test_share_without_a_server_only_writes_the_preview(tmp_path):
    save_result(private_run())
    result = runner.invoke(cli.app, ["share"])
    assert result.exit_code == 0
    assert "nothing was sent" in result.output
    preview = next((tmp_path / "share").glob("runs-*.json"))
    assert len(json.loads(preview.read_text())["runs"]) == 1
    assert len(not_yet_shared(load_results())) == 1


def test_share_sends_only_after_a_yes(server):
    save_result(private_run())
    result = runner.invoke(cli.app, ["share", "--to", server], input="n\n")
    assert "Nothing was sent." in result.output
    assert Collector.received == []

    result = runner.invoke(cli.app, ["share", "--to", server], input="y\n")
    assert result.exit_code == 0
    assert "Shared 1 runs" in result.output
    assert Collector.received[0]["runs"][0]["rig_id"] == "test-rig"
    assert "justi" not in json.dumps(Collector.received)


def test_each_run_is_shared_once(server):
    save_result(private_run())
    runner.invoke(cli.app, ["share", "--to", server, "--yes"])
    result = runner.invoke(cli.app, ["share", "--to", server, "--yes"])
    assert "No new runs to share." in result.output
    assert len(Collector.received) == 1
