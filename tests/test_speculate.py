# Tests for measuring speculative decoding. No model is loaded: a made-up llama-server answers
# each request with timings like the real one's, so the grid, the numbers kept and the tables
# can be checked by hand.

import time

import pytest
from typer.testing import CliRunner

from localtune import cli, speculate
from localtune.gguf import ModelLayout
from localtune.gpu_monitor import GpuMonitor, GpuSample
from localtune.hardware import GpuInfo, HardwareInfo
from localtune.runners.base import BackendError
from localtune.speculate import (
    ServerOptions,
    Setup,
    SpeculateRun,
    best_setup,
    built_in_cases,
    can_draft_for,
    load_runs,
    measure_setup,
    notes,
    overall,
    pick_drafts,
    room_for,
    save_runs,
    server_command,
    setups,
    with_baselines,
)

GPU = GpuInfo(index=0, name="NVIDIA GeForce RTX 4070 Ti SUPER", vram_gb=16.0, driver_version="1")
runner = CliRunner()


@pytest.fixture(autouse=True)
def temporary_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))


def a_layout(tokenizer="gpt2", vocab=151_936):
    return ModelLayout(
        layers=40, experts=None, expert_share=0.0, tokenizer=tokenizer, vocab_size=vocab
    )


# The grid of setups.


def test_the_first_setup_has_no_speculation_and_every_value_is_combined():
    drafts = [("qwen3:0.6b", "/m/small"), ("qwen3:1.7b", "/m/medium")]
    found = setups(drafts, ["draft-simple", "ngram-mod"], [4, 16], [None], [0.0, 0.5])
    assert not found[0].speculates
    with_draft = [s for s in found if s.draft]
    assert len(with_draft) == 2 * 2 * 2
    assert {(s.n_max, s.p_min) for s in with_draft} == {(4, 0.0), (4, 0.5), (16, 0.0), (16, 0.5)}
    assert found[-1] == Setup(spec_type="ngram-mod")
    assert len(found) == 1 + 8 + 1


def test_a_type_joined_with_a_plus_needs_a_draft_and_is_checked_part_by_part():
    assert speculate.needs_draft("draft-simple+ngram-mod")
    assert not speculate.needs_draft("ngram-mod")
    assert speculate.known_type("draft-simple+ngram-mod")
    assert not speculate.known_type("draft-simple+guessing")


def test_setups_are_named_by_what_sets_them_apart():
    assert Setup().label == "no speculation"
    assert Setup(spec_type="ngram-mod").label == "ngram-mod"
    draft = Setup(spec_type="draft-simple", draft="qwen3:0.6b", n_max=16, p_min=0.5)
    assert draft.label == "qwen3:0.6b, n-max 16, p-min 0.5"


# Choosing drafts from the models already downloaded.


def test_a_draft_must_cut_text_the_same_way():
    target = a_layout()
    assert can_draft_for(a_layout(vocab=151_936 + 100), target)
    assert not can_draft_for(a_layout(vocab=128_256), target)
    assert not can_draft_for(a_layout(tokenizer="llama"), target)
    assert not can_draft_for(a_layout(tokenizer=None), target)


def test_the_smallest_models_of_the_same_family_are_picked(tmp_path):
    sizes = {"target": 9000, "tiny": 500, "small": 1400, "half": 5000, "other": 400}
    files = {}
    for name, size in sizes.items():
        files[name] = tmp_path / name
        files[name].write_bytes(b"x" * size)
    layouts = {name: a_layout() for name in sizes} | {"other": a_layout(vocab=32_000)}
    picked = pick_drafts(
        files["target"],
        layouts["target"],
        [(name, file) for name, file in files.items()],
        lambda file: layouts[file.name],
    )
    assert [name for name, _ in picked] == ["tiny", "small"]


# The cases.


def test_both_tasks_read_the_same_notes_and_the_notes_never_change():
    assert notes(2048) == notes(2048)
    assert notes(2048).count("\n") + 1 == 2048 // speculate.TOKENS_PER_NOTE
    cases = built_in_cases(["copy", "fresh"], [2048, 16_384], [0.0, 0.8])
    assert len(cases) == 8
    copy, fresh = cases[0], cases[4]
    assert (copy.task, fresh.task) == ("copy", "fresh")
    assert copy.prompt.split("\n\n")[0] == fresh.prompt.split("\n\n")[0]
    assert "word for word" in copy.prompt and "new story" in fresh.prompt


def test_the_server_gets_room_for_the_longest_prompt_and_its_answer():
    cases = built_in_cases(["copy"], [2048, 16_384], [0.0])
    room = room_for(cases, 256)
    assert room % 1024 == 0
    assert room >= 16_384 * 1.25 + 256


# Starting llama-server.


def test_a_draft_setup_is_started_with_the_flags_this_llama_cpp_knows(tmp_path):
    setup = Setup(spec_type="draft-simple+ngram-mod", draft="tiny", draft_file="/m/tiny", n_max=8)
    options = ServerOptions(
        context=4096, kv_cache="q8_0", draft_kv_cache="q4_0", llama_args=["--threads", "6"]
    )
    command = server_command("llama-server", tmp_path / "m.gguf", setup, options, 9000)
    flag = lambda name: command[command.index(name) + 1]  # noqa: E731
    assert flag("--spec-draft-model") == "/m/tiny"
    assert flag("--spec-draft-ngl") == "999"
    assert flag("--spec-type") == "draft-simple,ngram-mod"
    assert flag("--spec-draft-n-max") == "8"
    assert flag("--spec-draft-type-k") == "q4_0"
    assert flag("-ctk") == "q8_0"
    assert "--spec-draft-p-min" not in command
    assert command[-2:] == ["--threads", "6"]


def test_the_plain_setup_has_no_speculation_flags(tmp_path):
    command = server_command(
        "llama-server", tmp_path / "m.gguf", Setup(), ServerOptions(context=4096), 9000
    )
    assert not [part for part in command if part.startswith("--spec")]
    assert command[command.index("-ngl") + 1] == "999"


# One setup, against a made-up server.


class Process:
    returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def a_monitor(peak_gb):
    count = iter(range(100_000))

    def read():
        t = next(count)
        return {
            0: GpuSample(
                time_s=t,
                power_w=100.0,
                vram_used_gb=peak_gb if t else 0.5,
                utilization_pct=10.0,
                temperature_c=50.0,
            )
        }

    return lambda gpu_index: GpuMonitor(interval_s=0.01, cards_reader=read)


def a_server(speed=40.0, kept=0.75, fail_on=None):
    asked = []

    def post(url, body, timeout_s=30):
        asked.append(body)
        time.sleep(0.01)
        prompt = body["messages"][0]["content"]
        if fail_on and fail_on in prompt:
            raise OSError("connection reset")
        timings = {
            "prompt_n": 2000,
            "prompt_ms": 1000.0,
            "prompt_per_second": 2000.0,
            "predicted_n": body["max_tokens"],
            "predicted_ms": 1000.0 * body["max_tokens"] / (speed + body["seed"]),
            "predicted_per_second": speed + body["seed"],
        }
        if kept is not None:
            timings |= {"draft_n": 200, "draft_n_accepted": int(200 * kept)}
        return {"timings": timings}

    return post, asked


def measured(tmp_path, post, ready=lambda *args: None, peak_gb=12.0, monitor=None, cases=None):
    commands = []

    def start(command, **kwargs):
        commands.append(command)
        return Process()

    rows = measure_setup(
        "llama-server",
        tmp_path / "model.gguf",
        Setup(spec_type="draft-simple", draft="tiny", draft_file="/m/tiny", n_max=8),
        cases or built_in_cases(["copy", "fresh"], [2048], [0.0]),
        ServerOptions(context=4096),
        [GPU],
        gen_tokens=128,
        repeats=3,
        start=start,
        monitor=monitor or a_monitor(peak_gb),
        ready=ready,
        post=post,
        props=lambda url, timeout: {"build_info": "b11669-abc"},
    )
    return rows, commands


def test_a_setup_answers_every_case_and_keeps_the_middle_speed(tmp_path):
    post, asked = a_server(speed=40.0, kept=0.75)
    rows, commands = measured(tmp_path, post)
    assert len(commands) == 1
    assert len(asked) == 1 + 2 * 3  # one to warm up, then three per case
    assert [body["seed"] for body in asked[1:4]] == [0, 1, 2]
    assert all(body["cache_prompt"] is False and body["stream"] is False for body in asked)
    row = rows[0]
    assert row["decode_runs"] == [40.0, 41.0, 42.0]
    assert row["decode_tok_s"] == 41.0
    assert (row["draft_tokens"], row["draft_accepted"]) == (600, 450)
    assert row["generated_tokens"] == 128
    assert row["llama_cpp"] == "b11669-abc"
    assert row["peak_vram_gb"] == 12.0 and row["free_gb"] == 4.0


def test_a_setup_that_cant_start_fails_every_case_and_says_why(tmp_path):
    def ready(*args):
        raise BackendError("llama-server didn't start: out of memory", status="out_of_memory")

    rows, _ = measured(tmp_path, a_server()[0], ready=ready)
    assert [row["status"] for row in rows] == ["out_of_memory", "out_of_memory"]
    assert "out of memory" in rows[0]["error"]


def test_one_case_failing_doesnt_stop_the_others(tmp_path):
    post, _ = a_server(fail_on="new story")
    rows, _ = measured(tmp_path, post)
    assert rows[0]["decode_tok_s"] == 41.0
    assert rows[1]["status"] == "crashed"


def test_a_server_without_guesses_reports_none(tmp_path):
    rows, _ = measured(tmp_path, a_server(kept=None)[0])
    assert rows[0]["draft_tokens"] is None and rows[0]["draft_accepted"] is None


# Reading a session.


def a_run(setup, task="copy", speed=40.0, free_gb=4.0, guessed=None, kept=None, **more):
    return SpeculateRun(
        session_id="s",
        rig_id="rig",
        model="qwen3:14b",
        setup=setup,
        task=task,
        context=2048,
        decode_tok_s=speed,
        decode_runs=[speed - 1, speed, speed + 1],
        peak_vram_gb=16.0 - free_gb,
        free_gb=free_gb,
        draft_tokens=guessed,
        draft_accepted=kept,
        **more,
    )


PLAIN = Setup()
DRAFT = Setup(spec_type="draft-simple", draft="qwen3:0.6b", draft_file="/m/tiny", n_max=8)
LOOKUP = Setup(spec_type="ngram-mod")


def a_session():
    return with_baselines(
        [
            a_run(PLAIN, "copy", 40.0),
            a_run(PLAIN, "fresh", 40.0),
            a_run(DRAFT, "copy", 80.0, free_gb=3.0, guessed=400, kept=360),
            a_run(DRAFT, "fresh", 45.0, free_gb=3.0, guessed=400, kept=160),
            a_run(LOOKUP, "copy", 60.0, guessed=300, kept=270),
            a_run(LOOKUP, "fresh", 40.0),
        ]
    )


def test_speed_up_is_against_the_plain_setup_on_the_same_case():
    runs = a_session()
    assert runs[0].speedup == 1.0
    assert runs[2].speedup == 2.0
    assert runs[3].speedup == 1.125
    assert runs[2].acceptance == 0.9
    assert runs[5].acceptance is None


def test_a_setups_overall_speed_up_is_the_geometric_mean():
    runs = a_session()
    assert overall(runs[2:4]) == pytest.approx((2.0 * 1.125) ** 0.5)
    setup, gain = best_setup(runs)
    assert setup == DRAFT
    assert gain == pytest.approx(1.5)


def test_a_setup_that_left_the_gpu_too_full_cant_be_the_best():
    runs = with_baselines(
        [
            a_run(PLAIN, speed=40.0),
            a_run(DRAFT, speed=80.0, free_gb=0.4),
            a_run(LOOKUP, speed=50.0),
        ]
    )
    assert runs[1].spilled
    assert best_setup(runs)[0] == LOOKUP


def test_runs_are_saved_apart_from_results_and_read_back(tmp_path):
    path = save_runs(a_session())
    assert path == tmp_path / "speculate.jsonl"
    assert not (tmp_path / "results.jsonl").exists()
    back = load_runs()
    assert len(back) == 6
    assert back[2].setup == DRAFT and back[2].speedup == 2.0


# The command.


def on_this_pc(monkeypatch, tmp_path, drafts=("qwen3:0.6b",)):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"x" * 9000)
    files = {}
    for name in drafts:
        files[name] = tmp_path / name.replace(":", "-")
        files[name].write_bytes(b"x" * 500)
    pc = HardwareInfo(os="Linux", is_wsl=False, cpu_name=None, cpu_threads=8, ram_gb=32, gpus=[GPU])
    monkeypatch.setattr(cli, "find_tool", lambda name: "/llama/llama-server")
    monkeypatch.setattr(cli, "read_layout", lambda path: a_layout())
    monkeypatch.setattr(cli, "detect_hardware", lambda: pc)
    monkeypatch.setattr(cli, "rig_id", lambda: "rig")
    monkeypatch.setattr(cli, "starting_split", lambda *args: {})
    monkeypatch.setattr(cli, "downloaded_models", lambda: list(files))
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: files.get(name))
    seen = []

    def measure(llama_server, model_file, setup, cases, options, gpus, gen_tokens, repeats):
        seen.append((setup, cases, options))
        rows = []
        for case in cases:
            speed = 40.0
            guessed = kept = None
            if setup.speculates:
                speed = 72.0 if case.task == "copy" else 42.0
                guessed, kept = 400, 340 if case.task == "copy" else 150
            rows.append(
                {
                    "decode_tok_s": speed,
                    "decode_runs": [speed - 1, speed, speed + 1],
                    "prompt_tokens": case.context,
                    "draft_tokens": guessed,
                    "draft_accepted": kept,
                    "peak_vram_gb": 11.0 if setup.draft else 10.0,
                    "free_gb": 5.0 if setup.draft else 6.0,
                }
            )
        return rows

    monkeypatch.setattr(cli, "measure_setup", measure)
    return model, seen


def test_the_command_with_only_a_model_picks_a_draft_and_runs_the_whole_grid(monkeypatch, tmp_path):
    model, seen = on_this_pc(monkeypatch, tmp_path)
    result = runner.invoke(cli.app, ["speculate", str(model)])
    assert result.exit_code == 0, result.output
    labels = [setup.label for setup, _, _ in seen]
    assert labels == ["no speculation", "qwen3:0.6b, n-max 4", "qwen3:0.6b, n-max 16", "ngram-mod"]
    assert len(seen[0][1]) == 8
    assert "4 setups" in result.output and "96 answers" in result.output
    assert "1.80x" in result.output and "85%" in result.output
    assert "+1.0 GB" in result.output
    assert "Measured:" in result.output and "--spec-draft-model" in result.output
    assert len(load_runs()) == 32


def test_every_list_option_multiplies_the_grid(monkeypatch, tmp_path):
    model, seen = on_this_pc(monkeypatch, tmp_path)
    args = ["--type", "draft-simple", "--n-max", "2,4,8", "--p-min", "0,0.5"]
    args += ["--context", "1024", "--temperature", "0", "--task", "copy", "--repeats", "5"]
    args += ["--llama-args", "--threads 6"]
    result = runner.invoke(cli.app, ["speculate", str(model), *args])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1 + 6
    assert [len(cases) for _, cases, _ in seen] == [1] * 7
    assert seen[0][2].llama_args == ["--threads", "6"]
    assert "35 answers" in result.output


def test_without_a_draft_downloaded_only_the_lookup_is_tried(monkeypatch, tmp_path):
    model, seen = on_this_pc(monkeypatch, tmp_path, drafts=())
    result = runner.invoke(cli.app, ["speculate", str(model)])
    assert result.exit_code == 0, result.output
    assert "--draft" in result.output
    assert [setup.label for setup, _, _ in seen] == ["no speculation", "ngram-mod"]


def test_your_own_prompt_files_replace_the_built_in_tasks(monkeypatch, tmp_path):
    model, seen = on_this_pc(monkeypatch, tmp_path)
    mine = tmp_path / "refactor.txt"
    mine.write_text("Rename x to count in this function: def f(x): return x + 1")
    result = runner.invoke(
        cli.app, ["speculate", str(model), "--prompt-file", str(mine), "--temperature", "0"]
    )
    assert result.exit_code == 0, result.output
    cases = seen[0][1]
    assert [(case.task, case.context) for case in cases] == [("refactor", None)]
    assert "Rename x" in cases[0].prompt


def test_nothing_clearly_faster_is_said_plainly(monkeypatch, tmp_path):
    model, _ = on_this_pc(monkeypatch, tmp_path)
    slow = lambda *args: [  # noqa: E731
        {"decode_tok_s": 40.0, "decode_runs": [40.0], "peak_vram_gb": 10.0, "free_gb": 6.0}
        for _ in args[3]
    ]
    monkeypatch.setattr(cli, "measure_setup", slow)
    result = runner.invoke(cli.app, ["speculate", str(model)])
    assert result.exit_code == 0, result.output
    assert "no setup made this model clearly faster" in result.output


def test_mistakes_are_explained_before_anything_is_loaded(monkeypatch, tmp_path):
    model, seen = on_this_pc(monkeypatch, tmp_path)
    for args, said in (
        (["--type", "guessing"], "no speculation type called guessing"),
        (["--n-max", "four"], "separated by commas"),
        (["--task", "poem"], "--task takes"),
        (["--kv-cache", "q2"], "Unknown KV cache type"),
    ):
        result = runner.invoke(cli.app, ["speculate", str(model), *args])
        assert result.exit_code == 1
        assert said in result.output
    assert seen == []
