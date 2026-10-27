"""Share saved benchmark runs with localtune's public dataset, only when asked."""

# Every run localtune saves (results.py) stays on this computer. "localtune share" (cli.py) is
# the only way any of it leaves, and only after the person has seen exactly what would be sent
# and said yes. Nothing is ever sent in the background. Shared runs become part of the public
# dataset the prediction model learns from, so more people's runs mean better predictions for
# GPUs and models nobody has tested yet.
#
# A shared run keeps only what the dataset needs: the GPU, CPU, RAM and OS, the model and the
# settings, the speeds, what the GPU did, and the energy used. It is linked to other runs from
# the same computer only by rig_id, the random ID localtune made for it, which says nothing
# about who or where anyone is. Left out are:
#   - error messages, which can contain folder paths with a person's user name;
#   - the electricity price and its source, which can contain a zip code, and the cost worked
#     out from it (the energy is kept, so cost can be worked out again for any price);
#   - the exact time of the run; only the day is kept.
# The computer's name, user name, IP address and prompts are never saved in a run at all.

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from localtune.results import SCHEMA_VERSION, RunResult, home_dir

LEFT_OUT = {
    "error": True,
    "energy": {"cost", "cost_per_million_tokens", "price"},
}


def public_record(run: RunResult) -> dict:
    record = run.model_dump(mode="json", exclude=LEFT_OUT)
    record["created_at"] = run.created_at.date().isoformat()
    return record


# Which runs were already shared is kept in a small file next to the results, one run_id per
# line, so each run is offered only once.


def shared_file() -> Path:
    return home_dir() / "shared_runs"


def shared_ids() -> set[str]:
    path = shared_file()
    return set(path.read_text().split()) if path.exists() else set()


def not_yet_shared(runs: list[RunResult]) -> list[RunResult]:
    done = shared_ids()
    return [run for run in runs if run.run_id not in done]


def mark_shared(runs: list[RunResult]) -> None:
    path = shared_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.writelines(run.run_id + "\n" for run in runs)


# What would be sent is first written to a file in the localtune folder, exactly as it would
# go out, so it can be read in full before anything is sent (or kept, or sent by hand).


def payload(runs: list[RunResult]) -> dict:
    return {"schema_version": SCHEMA_VERSION, "runs": [public_record(run) for run in runs]}


def write_preview(body: dict) -> Path:
    path = home_dir() / "share" / f"runs-{datetime.now():%Y%m%d-%H%M%S}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


# Where shared runs go: the address given with --to, or the LOCALTUNE_SHARE_URL setting. There
# is no address built in until localtune's own server for the dataset is running.


class ShareError(Exception):
    pass


def share_url(given: str | None) -> str | None:
    return given or os.environ.get("LOCALTUNE_SHARE_URL") or None


def send(url: str, body: dict, timeout_s: float = 30) -> None:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s):
            pass
    except urllib.error.HTTPError as error:
        raise ShareError(f"the server said no ({error.code} {error.reason})") from error
    except (urllib.error.URLError, OSError) as error:
        raise ShareError(f"couldn't reach {url} ({error})") from error
