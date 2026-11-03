# Tests for sample answers: the questions, marking answers right or wrong, picking models, and a
# whole run of "localtune samples" against a pretend Ollama, so no network or model is needed.

import io
import json

import pytest
from typer.testing import CliRunner

from localtune import cli, samples
from localtune.samples import (
    Question,
    SampleAnswer,
    ask,
    final_number,
    load_answers,
    load_questions,
    mark,
    models_that_fit,
    prompt_for,
    score,
    without_thinking,
)

runner = CliRunner()


def test_questions_go_from_level_1_to_20():
    questions = load_questions()
    assert [q.level for q in questions] == list(range(1, 21))
    assert len({q.id for q in questions}) == 20
    for q in questions:
        assert (q.answer is None) == (q.kind == "read")


def test_a_number_question_asks_for_an_answer_line():
    number = Question(level=3, id="n", kind="number", answer=14, ask="How much?")
    words = Question(level=1, id="w", kind="read", ask="Write a poem.")
    assert prompt_for(number).endswith("Answer: <the number>")
    assert prompt_for(words) == "Write a poem."


@pytest.mark.parametrize(
    ("text", "number"),
    [
        ("3 x 4 = 12, plus 2.\nAnswer: 14", 14),
        ("**Answer:** $1,250.50", 1250.5),
        ("It takes 6 minutes, not 120.\n\nAnswer: 6", 6),
        ("The machines need 6 minutes.", 6),
        ("No idea.", None),
    ],
)
def test_finds_the_final_number(text, number):
    assert final_number(text) == number


def test_marks_answers():
    number = Question(level=8, id="flour", kind="number", answer=750, ask="?")
    word = Question(level=12, id="kg", kind="contains", answer=["same", "equal"], ask="?")
    words = Question(level=1, id="w", kind="read", ask="?")
    assert mark(number, "300 / 4 x 10\nAnswer: 750") is True
    assert mark(number, "Answer: 700") is False
    assert mark(word, "They weigh the SAME.") is True
    assert mark(word, "Steel, obviously.") is False
    assert mark(words, "anything") is None


def test_thinking_is_left_out_of_the_answer():
    assert without_thinking("<think>hmm, 3 x 4</think>\n\nAnswer: 14") == "Answer: 14"
    assert without_thinking("<think>never finished") == ""


def test_picks_popular_models_that_fit():
    picked = models_that_fit(memory_gb=6, how_many=5)
    assert len(picked) == 5
    assert all(m.files[0].size_bytes <= 6e9 for m in picked)
    assert [m.popularity_rank for m in picked] == sorted(m.popularity_rank for m in picked)


def test_scores_the_checked_answers():
    def answer(level, correct):
        return SampleAnswer(
            model="m", question=str(level), level=level, kind="x", answer="", correct=correct
        )

    found = [answer(1, None), answer(3, True), answer(10, False), answer(14, True)]
    assert score(found) == (2, 3, 14)


# A pretend Ollama: answers each request by its address, and remembers what was sent.


class Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def pretend_ollama(monkeypatch, thinking=False, content="Answer: 14"):
    sent = []

    def urlopen(request, timeout=None):
        body = json.loads(request.data) if request.data else {}
        path = request.full_url.split("11434", 1)[1]
        sent.append((path, request.get_method(), body))
        if path == "/api/show":
            reply = {"capabilities": ["completion", "thinking"] if thinking else ["completion"]}
        elif path == "/api/chat":
            reply = {
                "message": {"content": content},
                "eval_count": 40,
                "total_duration": 2_500_000_000,
                "done_reason": "stop",
            }
        else:
            reply = {}
        return Reply(json.dumps(reply).encode())

    monkeypatch.setattr(samples.urllib.request, "urlopen", urlopen)
    return sent


def test_asks_with_a_fixed_seed(monkeypatch):
    sent = pretend_ollama(monkeypatch, content="<think>12 + 2</think>Answer: 14")
    q = Question(level=3, id="notebooks", kind="number", answer=14, ask="How much?")
    got = ask("http://127.0.0.1:11434", "qwen3:8b", q, thinks=True)
    path, _, body = sent[0]
    assert path == "/api/chat"
    assert body["options"]["seed"] == samples.SEED
    assert body["think"] is True
    assert got.answer == "Answer: 14"
    assert got.correct is True
    assert got.seconds == 2.5


def test_samples_asks_every_question_and_carries_on(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    sent = pretend_ollama(monkeypatch)
    monkeypatch.setattr(cli, "is_running", lambda url: True)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: tmp_path / "blob")
    result = runner.invoke(cli.app, ["samples", "qwen3:8b"])
    assert result.exit_code == 0, result.output
    assert len(load_answers()) == 20
    assert "qwen3:8b: " in result.output
    assert not any(path == "/api/delete" for path, _, _ in sent)

    asked = sum(1 for path, _, _ in sent if path == "/api/chat")
    runner.invoke(cli.app, ["samples", "qwen3:8b"])
    assert sum(1 for path, _, _ in sent if path == "/api/chat") == asked
    assert len(load_answers()) == 20


def test_samples_deletes_only_what_it_downloaded(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALTUNE_HOME", str(tmp_path))
    sent = pretend_ollama(monkeypatch)
    monkeypatch.setattr(cli, "is_running", lambda url: True)
    monkeypatch.setattr(cli, "find_ollama_model", lambda name: None)
    monkeypatch.setattr(cli, "download_for_samples", lambda url, model, place: True)
    result = runner.invoke(cli.app, ["samples", "gemma3:1b"])
    assert result.exit_code == 0, result.output
    deleted = [body["model"] for path, method, body in sent if path == "/api/delete"]
    assert deleted == ["gemma3:1b"]
