# Shared by every test: the reading of what else the computer is doing before a run
# (background.py) is replaced by a quiet computer, so tests don't wait for real readings or
# depend on what the test machine happens to be doing.

import pytest

from localtune import cli
from localtune.background import Background
from localtune.runners import base

QUIET = Background(
    gpu_utilization_pct=0, gpu_vram_used_gb=0.5, gpu_power_w=20, cpu_load_pct=3, ram_used_gb=8
)


@pytest.fixture(autouse=True)
def quiet_computer(monkeypatch):
    monkeypatch.setattr(base, "read_background", lambda gpu_index=0: QUIET)
    monkeypatch.setattr(cli, "read_background", lambda gpu_index=0: QUIET)
